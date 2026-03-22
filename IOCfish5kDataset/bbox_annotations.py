"""
Estimate bounding boxes from point annotations.

Two methods:
  --method watershed   marker-controlled watershed on RGB (default, no GPU)
  --method sam2        SAM 2 single-point prompting  (needs GPU + checkpoint)

Draws bboxes + center dots and saves results to images/bbox_annotated/.
Run from inside the dataset folder or adjust BASE_DIR.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# ── SAM 2 path setup (deferred import) ─────────────────────────────────────────
# Add the GECO2 root (parent of the sam2 repo dir) so that `import sam2`
# resolves to sam2/ and `sam2.sam2` resolves to sam2/sam2/ (the package).
SAM2_ROOT = Path(__file__).resolve().parent.parent
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
IMG_DIR  = BASE_DIR / "images"
ANN_DIR  = BASE_DIR / "annotations"
OUT_DIR  = IMG_DIR  / "bbox_annotated"

# ── visual settings ────────────────────────────────────────────────────────────
DOT_RADIUS     = 4
DOT_COLOR      = (0, 0, 255)      # red  (BGR)
BBOX_COLOR     = (0, 255, 0)      # green (BGR)
BBOX_THICKNESS = 2
TEXT_COLOR     = (0, 255, 255)     # yellow (BGR)
BBOX_PAD_FRAC  = 0.10             # 10 % padding around watershed bbox

# ── SAM 2 settings ─────────────────────────────────────────────────────────────
SAM2_HF_MODEL_ID = "facebook/sam2-hiera-base-plus"    # auto-downloaded
SAM2_MIN_MASK_PX  = 10       # ignore masks smaller than this many pixels


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_points(xml_path: Path) -> list[tuple[int, int]]:
    """Return list of (x, y) from a Pascal-VOC-style point annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    points = []
    for obj in root.findall("object"):
        pt = obj.find("point")
        if pt is not None:
            x = int(pt.findtext("x", "0"))
            y = int(pt.findtext("y", "0"))
            points.append((x, y))
    return points


def _fish_scale(points: list[tuple[int, int]], h: int, w: int) -> int:
    """Median nearest-neighbour distance – a proxy for typical fish size."""
    if len(points) <= 1:
        return min(h, w) // 10
    pts = np.array(points, dtype=np.float32)
    nn = []
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        d[i] = np.inf
        nn.append(d.min())
    return max(15, int(np.median(nn) * 0.7))


def _pad_box(bx, by, bw, bh, h, w, frac=BBOX_PAD_FRAC):
    """Add proportional padding and clamp to image bounds."""
    px = max(1, int(bw * frac))
    py = max(1, int(bh * frac))
    bx = max(0, bx - px)
    by = max(0, by - py)
    bw = min(bw + 2 * px, w - bx)
    bh = min(bh + 2 * py, h - by)
    return bx, by, bw, bh


def _ensure_center_inside(bx, by, bw, bh, cx, cy, h, w):
    """Shift the box so the annotation centre is guaranteed to be inside."""
    if cx < bx:
        bx = max(0, cx - 2)
        bw = min(bw, w - bx)
    elif cx >= bx + bw:
        bw = min(cx - bx + 3, w - bx)
    if cy < by:
        by = max(0, cy - 2)
        bh = min(bh, h - by)
    elif cy >= by + bh:
        bh = min(cy - by + 3, h - by)
    return bx, by, bw, bh


# ── SAM 2 point-prompting ──────────────────────────────────────────────────────

def build_sam2_predictor(device: str = "cuda"):
    """Build and return a SAM2ImagePredictor (downloads weights on first run)."""
    import torch
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available – falling back to CPU (will be slow).")
        device = "cpu"

    model = build_sam2_hf(SAM2_HF_MODEL_ID, device=device)
    return SAM2ImagePredictor(model)


def compute_bboxes_sam2(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) bboxes via SAM 2 single-point prompting."""
    h, w = img_bgr.shape[:2]
    if not points:
        return []

    # SAM 2 expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    scale = _fish_scale(points, h, w)
    bboxes: list[tuple[int, int, int, int]] = []

    for cx, cy in points:
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))

        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx_c, cy_c]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )

        # Pick the mask with the highest predicted IoU
        best_idx = scores.argmax()
        mask = masks[best_idx]

        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is not None and len(coords) >= SAM2_MIN_MASK_PX:
            bx, by, bw, bh = cv2.boundingRect(coords)
        else:
            # Fallback: square box sized to the estimated scale
            half = scale // 2
            bx = max(0, cx_c - half)
            by = max(0, cy_c - half)
            bw = min(scale, w - bx)
            bh = min(scale, h - by)

        bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
        bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx_c, cy_c, h, w)
        bboxes.append((bx, by, bw, bh))

    predictor.reset_predictor()
    return bboxes


# ── watershed segmentation ─────────────────────────────────────────────────────

def compute_bboxes_watershed(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) bboxes estimated via watershed segmentation."""
    h, w = img_bgr.shape[:2]
    n = len(points)
    if n == 0:
        return []

    scale   = _fish_scale(points, h, w)
    max_dim = int(scale * 3)
    min_dim = max(8, scale // 4)

    # Smooth while preserving edges → better watershed result
    smooth = cv2.bilateralFilter(img_bgr, 9, 75, 75)

    # ── markers ────────────────────────────────────────────────────────────
    markers = np.zeros((h, w), dtype=np.int32)

    # Background label = 1 along the image border
    markers[0, :]  = 1
    markers[-1, :] = 1
    markers[:, 0]  = 1
    markers[:, -1] = 1

    # Additional background seeds on a grid, far from any fish centre
    center_map = np.full((h, w), 255, dtype=np.uint8)
    for cx, cy in points:
        center_map[np.clip(cy, 0, h - 1), np.clip(cx, 0, w - 1)] = 0
    dist_from_center = cv2.distanceTransform(center_map, cv2.DIST_L2, 5)

    grid_step = max(40, scale)
    for gy in range(grid_step // 2, h, grid_step):
        for gx in range(grid_step // 2, w, grid_step):
            if dist_from_center[gy, gx] > scale * 1.2:
                markers[gy, gx] = 1

    # Foreground: each fish centre gets its own label (2 … n+1)
    marker_r = max(2, scale // 10)
    for i, (cx, cy) in enumerate(points):
        cv2.circle(
            markers,
            (np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)),
            marker_r,
            i + 2,
            -1,
        )

    # ── watershed ──────────────────────────────────────────────────────────
    cv2.watershed(smooth, markers)

    # ── extract bboxes per label ───────────────────────────────────────────
    bboxes: list[tuple[int, int, int, int]] = []
    for i, (cx, cy) in enumerate(points):
        cx_c = np.clip(cx, 0, w - 1)
        cy_c = np.clip(cy, 0, h - 1)
        label  = i + 2
        region = (markers == label).astype(np.uint8)
        coords = cv2.findNonZero(region)

        if coords is not None and len(coords) >= min_dim * min_dim // 4:
            bx, by, bw, bh = cv2.boundingRect(coords)

            # ── constrain to max size, centred on annotation ──────────────
            if bw > max_dim:
                bx = max(0, cx_c - max_dim // 2)
                bw = min(max_dim, w - bx)
            if bh > max_dim:
                by = max(0, cy_c - max_dim // 2)
                bh = min(max_dim, h - by)

            # ── enforce min size ──────────────────────────────────────────
            if bw < min_dim:
                bx = max(0, cx_c - min_dim // 2)
                bw = min(min_dim, w - bx)
            if bh < min_dim:
                by = max(0, cy_c - min_dim // 2)
                bh = min(min_dim, h - by)
        else:
            # Fallback: square box sized to the estimated scale
            half = scale // 2
            bx = max(0, cx_c - half)
            by = max(0, cy_c - half)
            bw = min(scale, w - bx)
            bh = min(scale, h - by)

        bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
        bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx_c, cy_c, h, w)
        bboxes.append((bx, by, bw, bh))

    return bboxes


# ── drawing + I/O ──────────────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    xml_path: Path,
    out_path: Path,
    predictor=None,
) -> int:
    """Estimate bboxes, draw them with centre dots, and save the result.

    If *predictor* is given, uses SAM 2; otherwise falls back to watershed.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    points = parse_points(xml_path)
    if not points:
        return 0

    if predictor is not None:
        bboxes = compute_bboxes_sam2(img, points, predictor)
    else:
        bboxes = compute_bboxes_watershed(img, points)

    vis = img.copy()
    for (cx, cy), (bx, by, bw, bh) in zip(points, bboxes):
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), BBOX_COLOR, BBOX_THICKNESS)
        cv2.circle(vis, (cx, cy), DOT_RADIUS, DOT_COLOR, -1)
        cv2.circle(vis, (cx, cy), DOT_RADIUS + 1, (255, 255, 255), 1)

    label = f"count: {len(points)}"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(points)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate bboxes from point annotations.")
    parser.add_argument(
        "--method",
        choices=["sam2", "watershed"],
        default="sam2",
        help="Segmentation backend (default: sam2)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for SAM 2 (default: cuda)",
    )
    args = parser.parse_args()

    img_files = sorted(IMG_DIR.glob("*.jpg"))
    if not img_files:
        print(f"No .jpg files found in {IMG_DIR}")
        return

    predictor = None
    if args.method == "sam2":
        print("Loading SAM 2 model …")
        predictor = build_sam2_predictor(device=args.device)
        print("SAM 2 model ready.")

    print(f"Found {len(img_files)} images.  Method: {args.method}.")
    print(f"Saving bbox-annotated copies to {OUT_DIR} …")
    skipped = 0
    for img_path in img_files:
        xml_path = ANN_DIR / (img_path.stem + ".xml")
        if not xml_path.exists():
            skipped += 1
            continue
        out_path = OUT_DIR / img_path.name
        count = process_image(img_path, xml_path, out_path, predictor=predictor)
        print(f"  {img_path.name}  →  {count} bboxes", flush=True)

    if skipped:
        print(f"\nSkipped {skipped} image(s) with no matching annotation file.")
    print("Done.")


if __name__ == "__main__":
    main()
