"""
Estimate bounding boxes from point annotations.

Three methods:
  --method watershed   marker-controlled watershed on RGB (no GPU)
  --method sam2        SAM 2 single-point prompting  (needs GPU + checkpoint)
  --method combined    SAM 2 + GECO2 – keeps the more confident bbox per point
                       (needs GPU + GECO2 checkpoint)

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
XML_DIR  = OUT_DIR  / "xml"

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

# ── GECO2 settings ─────────────────────────────────────────────────────────────
GECO2_IMAGE_SIZE  = 1024
GECO2_EMB_DIM     = 256
GECO2_REDUCTION   = 16
GECO2_KERNEL_DIM  = 1
GECO2_NUM_OBJECTS = 3        # number of exemplar boxes for GECO2
GECO2_NMS_THR     = 0.5
GECO2_SCORE_DIV   = 0.11     # thr = 1 / SCORE_DIV (from inference.py)


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
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available – falling back to CPU (will be slow).")
        device = "cpu"

    sam2_configs_dir = str(SAM2_ROOT / "sam2" / "sam2_configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=sam2_configs_dir, version_base="1.2"):
        model = build_sam2_hf(SAM2_HF_MODEL_ID, device=device)
    return SAM2ImagePredictor(model)


def compute_bboxes_sam2(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Return (x, y, w, h) bboxes and pIoU confidence scores via SAM 2."""
    h, w = img_bgr.shape[:2]
    if not points:
        return [], []

    # SAM 2 expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    scale = _fish_scale(points, h, w)
    bboxes: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []

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
        confidence = float(scores[best_idx])
        mask = masks[best_idx]

        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is not None and len(coords) >= SAM2_MIN_MASK_PX:
            bx, by, bw, bh = cv2.boundingRect(coords)
        else:
            confidence = 0.0
            # Fallback: square box sized to the estimated scale
            half = scale // 2
            bx = max(0, cx_c - half)
            by = max(0, cy_c - half)
            bw = min(scale, w - bx)
            bh = min(scale, h - by)

        bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
        bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx_c, cy_c, h, w)
        bboxes.append((bx, by, bw, bh))
        confidences.append(confidence)

    predictor.reset_predictor()
    return bboxes, confidences


# ── watershed segmentation ─────────────────────────────────────────────────────

def compute_bboxes_watershed(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Return (x, y, w, h) bboxes and scores via watershed segmentation."""
    h, w = img_bgr.shape[:2]
    n = len(points)
    if n == 0:
        return [], []

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

    return bboxes, [0.0] * len(bboxes)


# ── GECO2 counting-model prompting ─────────────────────────────────────────────

def build_geco2_model(checkpoint_path: str, device: str = "cuda"):
    """Build and return the GECO2 counting model (CNT)."""
    import torch
    from models.counter_infer import build_model

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available \u2013 falling back to CPU (will be slow).")
        device = "cpu"

    class _Args:
        image_size = GECO2_IMAGE_SIZE
        emb_dim = GECO2_EMB_DIM
        reduction = GECO2_REDUCTION
        kernel_dim = GECO2_KERNEL_DIM
        num_objects = GECO2_NUM_OBJECTS
        zero_shot = False

    model = build_model(_Args())

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Strip DataParallel 'module.' prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def _preprocess_for_geco2(
    img_bgr: np.ndarray,
    exemplar_bboxes_xyxy: list[tuple[int, int, int, int]],
    image_size: int = GECO2_IMAGE_SIZE,
):
    """Resize + pad image for GECO2 and scale exemplar boxes.

    Returns (img_tensor, exemplar_tensor, scaling_factor, (pad_w, pad_h)).
    """
    import torch

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    _, oh, ow = img_t.shape
    longer = max(oh, ow)
    sf = image_size / longer

    ex_t = torch.tensor(exemplar_bboxes_xyxy, dtype=torch.float32)
    scaled_ex = ex_t * sf
    a_dim = (
        (scaled_ex[:, 2] - scaled_ex[:, 0]).mean()
        + (scaled_ex[:, 3] - scaled_ex[:, 1]).mean()
    ) / 2
    sf = min(1.0, 80.0 / max(a_dim.item(), 1e-6)) * sf

    resized = torch.nn.functional.interpolate(
        img_t.unsqueeze(0), scale_factor=sf, mode="bilinear", align_corners=False,
    )
    pad_h = max(0, image_size - resized.shape[2])
    pad_w = max(0, image_size - resized.shape[3])
    padded = torch.nn.functional.pad(
        resized, (0, pad_w, 0, pad_h), mode="constant", value=0,
    )[0]

    exemplar_boxes = ex_t * sf
    return padded, exemplar_boxes, sf, (pad_w, pad_h)


def _match_detections_to_points(
    points: list[tuple[int, int]],
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    h: int,
    w: int,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Assign the best detected box to each annotation point."""
    result_bboxes: list[tuple[int, int, int, int]] = []
    result_scores: list[float] = []

    for cx, cy in points:
        best_idx = -1
        best_score = -1.0

        if len(boxes_xyxy) > 0:
            # Prefer the highest-scoring box that contains the point
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                if x1 <= cx <= x2 and y1 <= cy <= y2 and scores[i] > best_score:
                    best_idx = i
                    best_score = float(scores[i])

            # Fallback: closest box centre
            if best_idx == -1:
                centres = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2
                dists = np.hypot(centres[:, 0] - cx, centres[:, 1] - cy)
                best_idx = int(dists.argmin())
                best_score = float(scores[best_idx])

        if best_idx == -1:
            result_bboxes.append((0, 0, 0, 0))
            result_scores.append(0.0)
        else:
            x1, y1, x2, y2 = boxes_xyxy[best_idx]
            bx = max(0, int(x1))
            by = max(0, int(y1))
            bw = min(max(1, int(x2 - x1)), w - bx)
            bh = min(max(1, int(y2 - y1)), h - by)
            result_bboxes.append((bx, by, bw, bh))
            result_scores.append(best_score)

    return result_bboxes, result_scores


def compute_bboxes_geco2(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    exemplar_bboxes_xywh: list[tuple[int, int, int, int]],
    geco2_model,
    device: str = "cuda",
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Run GECO2 inference and match detections to annotation points.

    *exemplar_bboxes_xywh*: 3 exemplar boxes in (x, y, w, h) format
    (typically the top-scoring SAM 2 boxes for this image).
    """
    import torch
    from torchvision import ops

    h, w = img_bgr.shape[:2]
    if not points:
        return [], []

    # Convert exemplars xywh -> xyxy
    ex_xyxy = [(x, y, x + bw, y + bh) for x, y, bw, bh in exemplar_bboxes_xywh]

    img_tensor, ex_tensor, sf, (pad_w, pad_h) = _preprocess_for_geco2(
        img_bgr, ex_xyxy,
    )
    img_tensor = img_tensor.unsqueeze(0).to(device)
    ex_tensor = ex_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ref, _cent, _coord, _masks = geco2_model(img_tensor, ex_tensor)

    # Post-process (mirrors GECO2/inference.py)
    idx = 0
    pred_boxes = outputs[idx]["pred_boxes"]
    if pred_boxes.numel() == 0:
        return [(0, 0, 0, 0)] * len(points), [0.0] * len(points)

    v = outputs[idx]["box_v"]
    thr = 1.0 / GECO2_SCORE_DIV
    v_thr = v.max() / thr
    sel = v > v_thr
    if sel.sum() == 0:
        return [(0, 0, 0, 0)] * len(points), [0.0] * len(points)

    keep = ops.nms(pred_boxes[sel], v[sel], GECO2_NMS_THR)
    boxes = torch.clamp(pred_boxes[sel][keep], 0, 1)
    det_scores = outputs[idx]["scores"][sel][keep]

    # Remove detections in the padded area
    img_sz = float(img_tensor.shape[-1])
    maxw_ = img_sz - pad_w
    maxh_ = img_sz - pad_h
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    valid = (ctr[:, 0] * img_sz < maxw_) & (ctr[:, 1] * img_sz < maxh_)
    boxes = boxes[valid]
    det_scores = det_scores[valid]

    # Normalised [0, 1] -> original pixel coordinates
    boxes_px = (boxes * img_sz / sf).cpu().numpy()
    scores_np = det_scores.cpu().numpy()

    return _match_detections_to_points(points, boxes_px, scores_np, h, w)


def compute_bboxes_combined(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    sam2_predictor,
    geco2_model,
    device: str = "cuda",
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Run SAM 2 and GECO2, keeping the more confident bbox per point."""
    if not points:
        return [], []

    # 1. SAM 2 bboxes + confidence
    sam2_bboxes, sam2_scores = compute_bboxes_sam2(img_bgr, points, sam2_predictor)

    # 2. Select top exemplars for GECO2
    n_ex = min(GECO2_NUM_OBJECTS, len(sam2_bboxes))
    ranked = sorted(
        range(len(sam2_scores)), key=lambda i: sam2_scores[i], reverse=True,
    )
    ex_idx = ranked[:n_ex]
    while len(ex_idx) < GECO2_NUM_OBJECTS:
        ex_idx.append(ex_idx[-1])
    exemplars = [sam2_bboxes[i] for i in ex_idx]

    # 3. GECO2 bboxes + confidence
    geco2_bboxes, geco2_scores = compute_bboxes_geco2(
        img_bgr, points, exemplars, geco2_model, device,
    )

    # 4. Per-point selection: higher confidence wins
    final_bboxes: list[tuple[int, int, int, int]] = []
    final_scores: list[float] = []
    for i in range(len(points)):
        if geco2_scores[i] > sam2_scores[i]:
            final_bboxes.append(geco2_bboxes[i])
            final_scores.append(geco2_scores[i])
        else:
            final_bboxes.append(sam2_bboxes[i])
            final_scores.append(sam2_scores[i])

    return final_bboxes, final_scores


# ── XML export ─────────────────────────────────────────────────────────────────

def write_bbox_xml(
    xml_path: Path,
    img_path: Path,
    img_shape: tuple[int, int, int],
    points: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    confidences: list[float] | None = None,
) -> None:
    """Write bbox + centre annotations in Pascal-VOC-style XML."""
    h, w, d = img_shape

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = img_path.parent.name
    ET.SubElement(root, "filename").text = img_path.name
    ET.SubElement(root, "path").text = str(img_path)

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(d)

    ET.SubElement(root, "segmented").text = "0"

    for i, ((cx, cy), (bx, by, bw, bh)) in enumerate(zip(points, bboxes)):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "fish"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        pt = ET.SubElement(obj, "point")
        ET.SubElement(pt, "x").text = str(cx)
        ET.SubElement(pt, "y").text = str(cy)

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bx)
        ET.SubElement(bndbox, "ymin").text = str(by)
        ET.SubElement(bndbox, "xmax").text = str(bx + bw)
        ET.SubElement(bndbox, "ymax").text = str(by + bh)

        if confidences is not None:
            ET.SubElement(obj, "confidence").text = f"{confidences[i]:.4f}"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(xml_path, encoding="unicode", xml_declaration=True)


# ── XML parsing (bbox annotations) ─────────────────────────────────────────────

def parse_bbox_xml(
    xml_path: Path,
) -> list[tuple[tuple[int, int], tuple[int, int, int, int]]]:
    """Parse a bbox-annotated XML and return [(center, bbox), ...].

    Each entry is ((cx, cy), (xmin, ymin, xmax, ymax)).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []
    for obj in root.findall("object"):
        pt = obj.find("point")
        bb = obj.find("bndbox")
        if pt is None or bb is None:
            continue
        cx = int(pt.findtext("x", "0"))
        cy = int(pt.findtext("y", "0"))
        xmin = int(bb.findtext("xmin", "0"))
        ymin = int(bb.findtext("ymin", "0"))
        xmax = int(bb.findtext("xmax", "0"))
        ymax = int(bb.findtext("ymax", "0"))
        results.append(((cx, cy), (xmin, ymin, xmax, ymax)))
    return results


# ── Phase 1: compute bboxes → write XML ────────────────────────────────────────

def generate_bbox_xml(
    img_path: Path,
    ann_xml_path: Path,
    method: str = "sam2",
    sam2_predictor=None,
    geco2_model=None,
    device: str = "cuda",
) -> int:
    """Compute bboxes for one image and write the result as XML.

    Returns the number of objects written (0 if skipped).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    points = parse_points(ann_xml_path)
    if not points:
        return 0

    if method == "combined" and sam2_predictor is not None and geco2_model is not None:
        bboxes, scores = compute_bboxes_combined(
            img, points, sam2_predictor, geco2_model, device,
        )
    elif method == "sam2" and sam2_predictor is not None:
        bboxes, scores = compute_bboxes_sam2(img, points, sam2_predictor)
    else:
        bboxes, scores = compute_bboxes_watershed(img, points)

    xml_out = XML_DIR / (img_path.stem + ".xml")
    write_bbox_xml(xml_out, img_path, img.shape, points, bboxes, confidences=scores)
    return len(points)


# ── Phase 2: read XML → draw annotated image ───────────────────────────────────

def draw_annotated_image(img_path: Path, bbox_xml_path: Path, out_path: Path) -> int:
    """Read a bbox XML and draw centres + bboxes on the image.

    Returns the number of objects drawn (0 if skipped).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    entries = parse_bbox_xml(bbox_xml_path)
    if not entries:
        return 0

    vis = img.copy()
    for (cx, cy), (xmin, ymin, xmax, ymax) in entries:
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), BBOX_COLOR, BBOX_THICKNESS)
        cv2.circle(vis, (cx, cy), DOT_RADIUS, DOT_COLOR, -1)
        cv2.circle(vis, (cx, cy), DOT_RADIUS + 1, (255, 255, 255), 1)

    label = f"count: {len(entries)}"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(entries)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate bboxes from point annotations.")
    parser.add_argument(
        "--method",
        choices=["sam2", "watershed", "combined"],
        default="combined",
        help="Segmentation backend (default: combined – SAM 2 + GECO2)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for SAM 2 / GECO2 (default: cuda)",
    )
    parser.add_argument(
        "--geco2_checkpoint",
        type=str,
        default=None,
        help="Path to GECO2 model checkpoint (.pth). Required for combined.",
    )
    args = parser.parse_args()

    img_files = sorted(IMG_DIR.glob("*.jpg"))
    if not img_files:
        print(f"No .jpg files found in {IMG_DIR}")
        return

    sam2_predictor = None
    geco2_model = None

    if args.method in ("combined", "sam2"):
        # Build GECO2 *before* SAM 2 to avoid Hydra-init conflicts
        if args.method == "combined":
            if args.geco2_checkpoint is None:
                print("Error: --geco2_checkpoint is required for the combined method.")
                return
            print("Loading GECO2 model …")
            geco2_model = build_geco2_model(args.geco2_checkpoint, device=args.device)
            print("GECO2 model ready.")

        print("Loading SAM 2 model …")
        sam2_predictor = build_sam2_predictor(device=args.device)
        print("SAM 2 model ready.")

    print(f"Found {len(img_files)} images.  Method: {args.method}.")

    # ── Phase 1: compute bboxes and write XML ──────────────────────────────
    print(f"Phase 1 – writing bbox XML to {XML_DIR} …")
    skipped = 0
    for img_path in img_files:
        ann_xml = ANN_DIR / (img_path.stem + ".xml")
        if not ann_xml.exists():
            skipped += 1
            continue
        count = generate_bbox_xml(
            img_path, ann_xml,
            method=args.method,
            sam2_predictor=sam2_predictor,
            geco2_model=geco2_model,
            device=args.device,
        )
        print(f"  {img_path.name}  →  {count} bboxes (xml)", flush=True)

    if skipped:
        print(f"  Skipped {skipped} image(s) with no matching annotation file.")

    # ── Phase 2: read XML and draw annotated images ────────────────────────
    print(f"Phase 2 – drawing annotated images to {OUT_DIR} …")
    drawn = 0
    for img_path in img_files:
        bbox_xml = XML_DIR / (img_path.stem + ".xml")
        if not bbox_xml.exists():
            continue
        out_path = OUT_DIR / img_path.name
        count = draw_annotated_image(img_path, bbox_xml, out_path)
        print(f"  {img_path.name}  →  {count} annotations drawn", flush=True)
        drawn += 1

    print(f"Done. {drawn} annotated image(s) saved.")


if __name__ == "__main__":
    main()
