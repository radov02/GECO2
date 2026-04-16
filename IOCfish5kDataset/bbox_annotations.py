"""
Estimate bounding boxes from point annotations.

Available methods:
  watershed  - marker-controlled watershed on RGB, no GPU needed
  sam2       - SAM 2 with single-point prompts, needs GPU
  combined   - SAM 2 + GECO2, picks the more confident bbox per point
  rgbd       - SAM 2 on RGB and depth colormap independently, fuses results

Default folder structure (inside IOCfish5kDataset/):

  point_annotations/          <-- input
    images/   ####.jpg           RGB images
    color/    ####_depth.jpg     depth colormaps
    xml/      ####.xml           point-only annotations

  annotated_images/           <-- output
    images/   ####.jpg           copied from point_annotations/images/
    color/    ####_depth.jpg     copied from point_annotations/color/
    xml/      ####.xml           new XMLs with bounding boxes

When a bbox XML is written, the corresponding image and depth file
are copied into annotated_images/ so that inputs and outputs stay
together. All paths can be overridden via CLI arguments.
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# Add the GECO2 root so that "import sam2" resolves to sam2/ inside the repo.
SAM2_ROOT = Path(__file__).resolve().parent.parent
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

# Default paths relative to the dataset folder.
# Input lives under point_annotations/, output under annotated_images/.
BASE_DIR       = Path(__file__).parent
IN_DIR         = BASE_DIR / "point_annotations"
IMG_DIR        = IN_DIR / "images"
ANN_DIR        = IN_DIR / "xml"
DEPTH_DIR      = IN_DIR / "color"
OUT_DIR        = BASE_DIR / "annotated_images"
OUT_IMG_DIR    = OUT_DIR / "images"
OUT_XML_DIR    = OUT_DIR / "xml"
OUT_DEPTH_DIR  = OUT_DIR / "color"

# Visualization colors (BGR) and line settings.
DOT_RADIUS     = 4
DOT_COLOR      = (0, 0, 255)      # red
BBOX_COLOR     = (0, 255, 0)      # green
BBOX_THICKNESS = 2
TEXT_COLOR     = (0, 255, 255)    # yellow
BBOX_PAD_FRAC  = 0.10            # 10% padding added around each bbox

# SAM 2 model and mask settings.
SAM2_HF_MODEL_ID  = "facebook/sam2-hiera-base-plus"  # downloaded on first run
SAM2_MIN_MASK_PX  = 10   # masks smaller than this are treated as failures

# RGB-D fusion thresholds.
RGBD_IOU_AGREE_THR   = 0.25  # masks with IoU above this are considered agreeing
RGBD_SCORE_DOMINANCE = 1.5   # one modality dominates if its score is this times higher
RGBD_DEPTH_COMPACT_W = 0.8   # prefer depth mask if it is smaller and at least this fraction of rgb score

# GECO2 model settings.
GECO2_IMAGE_SIZE  = 1024
GECO2_EMB_DIM     = 256
GECO2_REDUCTION   = 16
GECO2_KERNEL_DIM  = 1
GECO2_NUM_OBJECTS = 3     # number of exemplar boxes passed to GECO2
GECO2_NMS_THR     = 0.5
GECO2_SCORE_DIV   = 0.11  # score threshold is 1 / SCORE_DIV (matches inference.py)


def parse_points(xml_path: Path) -> list[tuple[int, int]]:
    """Read center-point annotations from a Pascal-VOC XML file."""
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
    """Estimate a typical fish size as the median nearest-neighbor distance."""
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
    """Expand bbox by a fraction of its size and clamp to image bounds."""
    px = max(1, int(bw * frac))
    py = max(1, int(bh * frac))
    bx = max(0, bx - px)
    by = max(0, by - py)
    bw = min(bw + 2 * px, w - bx)
    bh = min(bh + 2 * py, h - by)
    return bx, by, bw, bh


def _ensure_center_inside(bx, by, bw, bh, cx, cy, h, w):
    """Shift/expand the box if the annotated center point falls outside it."""
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


def build_sam2_predictor(device: str = "cuda"):
    """Load SAM 2 and return a SAM2ImagePredictor. Downloads weights on first run."""
    import torch
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (will be slow).")
        device = "cpu"

    sam2_configs_dir = str(SAM2_ROOT / "sam2" / "sam2_configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=sam2_configs_dir, version_base="1.2"):
        model = build_sam2_hf(SAM2_HF_MODEL_ID, device=device)
    return SAM2ImagePredictor(model)


def _sam2_predict_per_point(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[float],
    list[np.ndarray | None],
]:
    """
    Run SAM 2 on each center point and return bboxes, pIoU scores, and masks.
    Masks are None for points where SAM 2 returned a too-small or empty mask.
    """
    h, w = img_bgr.shape[:2]
    if not points:
        return [], [], []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    scale = _fish_scale(points, h, w)
    bboxes: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []
    raw_masks: list[np.ndarray | None] = []

    for cx, cy in points:
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))

        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx_c, cy_c]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )

        best_idx = scores.argmax()
        confidence = float(scores[best_idx])
        mask = masks[best_idx]

        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is not None and len(coords) >= SAM2_MIN_MASK_PX:
            bx, by, bw, bh = cv2.boundingRect(coords)
            raw_masks.append(mask.astype(bool))
        else:
            confidence = 0.0
            half = scale // 2
            bx = max(0, cx_c - half)
            by = max(0, cy_c - half)
            bw = min(scale, w - bx)
            bh = min(scale, h - by)
            raw_masks.append(None)

        bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
        bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx_c, cy_c, h, w)
        bboxes.append((bx, by, bw, bh))
        confidences.append(confidence)

    predictor.reset_predictor()
    return bboxes, confidences, raw_masks


def compute_bboxes_sam2(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Run SAM 2 on the RGB image and return (x, y, w, h) bboxes with pIoU scores."""
    bboxes, confidences, _ = _sam2_predict_per_point(img_bgr, points, predictor)
    return bboxes, confidences


def compute_bboxes_watershed(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Estimate bboxes using marker-controlled watershed. No GPU required."""
    h, w = img_bgr.shape[:2]
    n = len(points)
    if n == 0:
        return [], []

    scale   = _fish_scale(points, h, w)
    max_dim = int(scale * 3)
    min_dim = max(8, scale // 4)

    # Bilateral filter smooths noise while keeping edges sharp.
    smooth = cv2.bilateralFilter(img_bgr, 9, 75, 75)

    markers = np.zeros((h, w), dtype=np.int32)

    # Label the image border as background (label 1).
    markers[0, :]  = 1
    markers[-1, :] = 1
    markers[:, 0]  = 1
    markers[:, -1] = 1

    # Add background seeds on a grid, but only far from any annotation point.
    center_map = np.full((h, w), 255, dtype=np.uint8)
    for cx, cy in points:
        center_map[np.clip(cy, 0, h - 1), np.clip(cx, 0, w - 1)] = 0
    dist_from_center = cv2.distanceTransform(center_map, cv2.DIST_L2, 5)

    grid_step = max(40, scale)
    for gy in range(grid_step // 2, h, grid_step):
        for gx in range(grid_step // 2, w, grid_step):
            if dist_from_center[gy, gx] > scale * 1.2:
                markers[gy, gx] = 1

    # Each fish center gets its own foreground label starting at 2.
    marker_r = max(2, scale // 10)
    for i, (cx, cy) in enumerate(points):
        cv2.circle(
            markers,
            (np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)),
            marker_r,
            i + 2,
            -1,
        )

    cv2.watershed(smooth, markers)

    bboxes: list[tuple[int, int, int, int]] = []
    for i, (cx, cy) in enumerate(points):
        cx_c = np.clip(cx, 0, w - 1)
        cy_c = np.clip(cy, 0, h - 1)
        label  = i + 2
        region = (markers == label).astype(np.uint8)
        coords = cv2.findNonZero(region)

        if coords is not None and len(coords) >= min_dim * min_dim // 4:
            bx, by, bw, bh = cv2.boundingRect(coords)

            # Clamp to a reasonable max size centered on the annotation.
            if bw > max_dim:
                bx = max(0, cx_c - max_dim // 2)
                bw = min(max_dim, w - bx)
            if bh > max_dim:
                by = max(0, cy_c - max_dim // 2)
                bh = min(max_dim, h - by)

            # Enforce a minimum size so tiny regions are still usable.
            if bw < min_dim:
                bx = max(0, cx_c - min_dim // 2)
                bw = min(min_dim, w - bx)
            if bh < min_dim:
                by = max(0, cy_c - min_dim // 2)
                bh = min(min_dim, h - by)
        else:
            # Watershed failed for this point, fall back to a square box.
            half = scale // 2
            bx = max(0, cx_c - half)
            by = max(0, cy_c - half)
            bw = min(scale, w - bx)
            bh = min(scale, h - by)

        bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
        bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx_c, cy_c, h, w)
        bboxes.append((bx, by, bw, bh))

    return bboxes, [0.0] * len(bboxes)


def build_geco2_model(checkpoint_path: str, device: str = "cuda"):
    """Load GECO2 from a checkpoint and return the model in eval mode."""
    import torch
    from models.counter import build_model

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (will be slow).")
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
    # Remove DataParallel "module." prefix if the checkpoint was saved that way.
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
    """Resize and pad the image to GECO2 input size, scaling exemplar boxes accordingly.

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
    """Match each annotation point to the best GECO2 detection box."""
    result_bboxes: list[tuple[int, int, int, int]] = []
    result_scores: list[float] = []

    for cx, cy in points:
        best_idx = -1
        best_score = -1.0

        if len(boxes_xyxy) > 0:
            # First try to find the highest-scoring box that contains the point.
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                if x1 <= cx <= x2 and y1 <= cy <= y2 and scores[i] > best_score:
                    best_idx = i
                    best_score = float(scores[i])

            # If no box contains the point, take the nearest box center.
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
    """
    Run GECO2 and assign one detection box per annotation point.
    exemplar_bboxes_xywh should be 3 representative fish boxes (x, y, w, h),
    typically picked as the highest-scoring SAM 2 results for this image.
    """
    import torch
    from torchvision import ops

    h, w = img_bgr.shape[:2]
    if not points:
        return [], []

    # GECO2 expects exemplars in xyxy format.
    ex_xyxy = [(x, y, x + bw, y + bh) for x, y, bw, bh in exemplar_bboxes_xywh]

    img_tensor, ex_tensor, sf, (pad_w, pad_h) = _preprocess_for_geco2(
        img_bgr, ex_xyxy,
    )
    img_tensor = img_tensor.unsqueeze(0).to(device)
    ex_tensor = ex_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ref, _cent, _coord, _masks = geco2_model(img_tensor, ex_tensor)

    # Post-process detections, mirroring the logic in GECO2/inference.py.
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

    # Discard detections that fall in the padded (empty) area of the image.
    img_sz = float(img_tensor.shape[-1])
    maxw_ = img_sz - pad_w
    maxh_ = img_sz - pad_h
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    valid = (ctr[:, 0] * img_sz < maxw_) & (ctr[:, 1] * img_sz < maxh_)
    boxes = boxes[valid]
    det_scores = det_scores[valid]

    # Convert from normalized [0, 1] coordinates back to pixel space.
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
    """Run both SAM 2 and GECO2, then keep the higher-confidence bbox per point."""
    if not points:
        return [], []

    sam2_bboxes, sam2_scores = compute_bboxes_sam2(img_bgr, points, sam2_predictor)

    # Pick the top-scoring SAM 2 boxes as exemplars for GECO2.
    n_ex = min(GECO2_NUM_OBJECTS, len(sam2_bboxes))
    ranked = sorted(
        range(len(sam2_scores)), key=lambda i: sam2_scores[i], reverse=True,
    )
    ex_idx = ranked[:n_ex]
    while len(ex_idx) < GECO2_NUM_OBJECTS:
        ex_idx.append(ex_idx[-1])
    exemplars = [sam2_bboxes[i] for i in ex_idx]

    geco2_bboxes, geco2_scores = compute_bboxes_geco2(
        img_bgr, points, exemplars, geco2_model, device,
    )

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


def _mask_to_bbox_xywh(
    mask: np.ndarray, cx: int, cy: int, h: int, w: int, scale: int,
) -> tuple[int, int, int, int]:
    """Convert a binary mask to (x, y, w, h). Falls back to a square box if the mask is empty."""
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is not None and len(coords) >= SAM2_MIN_MASK_PX:
        bx, by, bw, bh = cv2.boundingRect(coords)
    else:
        half = scale // 2
        bx = max(0, cx - half)
        by = max(0, cy - half)
        bw = min(scale, w - bx)
        bh = min(scale, h - by)
    bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
    bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx, cy, h, w)
    return bx, by, bw, bh


def _fuse_single_point(
    rgb_mask: np.ndarray | None,
    rgb_score: float,
    depth_mask: np.ndarray | None,
    depth_score: float,
    cx: int,
    cy: int,
    h: int,
    w: int,
    scale: int,
) -> tuple[tuple[int, int, int, int], float]:
    """
    Fuse the RGB and depth SAM 2 masks for one point into a single bbox.

    When both masks are valid:
      - High IoU means they agree, so we use the intersection (tighter box).
      - If one score clearly dominates, we trust that modality.
      - If depth produces a smaller, reasonably confident mask, we prefer it
        since depth boundaries tend to be cleaner at object edges.
      - Otherwise we take whichever score is higher.
    If only one mask is valid we use it directly.
    If neither is valid we fall back to a square box sized by fish scale.
    """
    rgb_ok = rgb_mask is not None and rgb_mask.sum() >= SAM2_MIN_MASK_PX
    depth_ok = depth_mask is not None and depth_mask.sum() >= SAM2_MIN_MASK_PX

    if rgb_ok and depth_ok:
        intersection = rgb_mask & depth_mask
        union = rgb_mask | depth_mask
        union_px = union.sum()
        iou = float(intersection.sum()) / max(float(union_px), 1.0)

        # Both modalities agree on this region, use the intersection for a tighter box.
        if iou >= RGBD_IOU_AGREE_THR and intersection.sum() >= SAM2_MIN_MASK_PX:
            conf = max(rgb_score, depth_score) * (0.5 + 0.5 * iou)
            return _mask_to_bbox_xywh(intersection, cx, cy, h, w, scale), conf

        # One modality is much more confident, trust it.
        if rgb_score > depth_score * RGBD_SCORE_DOMINANCE:
            return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score
        if depth_score > rgb_score * RGBD_SCORE_DOMINANCE:
            return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

        # Depth mask is more compact and reasonably confident, prefer it.
        rgb_area = float(rgb_mask.sum())
        depth_area = float(depth_mask.sum())
        if depth_area < rgb_area and depth_score >= rgb_score * RGBD_DEPTH_COMPACT_W:
            return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

        # No strong preference, just use the higher score.
        if rgb_score >= depth_score:
            return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score
        else:
            return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

    elif rgb_ok:
        return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score

    elif depth_ok:
        return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

    else:
        # No usable mask from either modality.
        half = scale // 2
        bx = max(0, cx - half)
        by = max(0, cy - half)
        bw = min(scale, w - bx)
        bh = min(scale, h - by)
        bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
        bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx, cy, h, w)
        return (bx, by, bw, bh), 0.0


def compute_bboxes_rgbd(
    img_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """
    Run SAM 2 on the RGB image and separately on the depth colormap,
    then fuse the resulting masks per annotation point.
    depth_bgr must be the colorized depth image at the same (or similar) resolution.
    """
    h, w = img_bgr.shape[:2]
    if not points:
        return [], []

    dh, dw = depth_bgr.shape[:2]
    if (dh, dw) != (h, w):
        depth_bgr = cv2.resize(depth_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    rgb_bboxes, rgb_scores, rgb_masks = _sam2_predict_per_point(
        img_bgr, points, predictor,
    )
    depth_bboxes, depth_scores, depth_masks = _sam2_predict_per_point(
        depth_bgr, points, predictor,
    )

    scale = _fish_scale(points, h, w)
    fused_bboxes: list[tuple[int, int, int, int]] = []
    fused_scores: list[float] = []

    for i, (cx, cy) in enumerate(points):
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))

        bbox, conf = _fuse_single_point(
            rgb_masks[i], rgb_scores[i],
            depth_masks[i], depth_scores[i],
            cx_c, cy_c, h, w, scale,
        )
        fused_bboxes.append(bbox)
        fused_scores.append(conf)

    return fused_bboxes, fused_scores


def write_bbox_xml(
    xml_path: Path,
    img_path: Path,
    img_shape: tuple[int, int, int],
    points: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    confidences: list[float] | None = None,
) -> None:
    """Save center points and bounding boxes to a Pascal-VOC-style XML file."""
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


def parse_bbox_xml(
    xml_path: Path,
) -> list[tuple[tuple[int, int], tuple[int, int, int, int]]]:
    """Read bounding boxes from an annotated XML. Returns [(cx, cy), (xmin, ymin, xmax, ymax)] per object."""
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


def generate_bbox_xml(
    img_path: Path,
    ann_xml_path: Path,
    method: str = "sam2",
    sam2_predictor=None,
    geco2_model=None,
    device: str = "cuda",
    depth_dir: Path | None = None,
    xml_out_dir: Path | None = None,
) -> int:
    """
    Compute bounding boxes for one image and write them to an XML file.
    depth_dir is required for the rgbd method (expects <stem>_depth.jpg files).
    Returns the number of annotations written, or 0 if the image was skipped.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    points = parse_points(ann_xml_path)
    if not points:
        return 0

    if method == "rgbd" and sam2_predictor is not None and depth_dir is not None:
        depth_path = depth_dir / (img_path.stem + "_depth.jpg")
        if depth_path.exists():
            depth_img = cv2.imread(str(depth_path))
        else:
            depth_img = None

        if depth_img is not None:
            bboxes, scores = compute_bboxes_rgbd(
                img, depth_img, points, sam2_predictor,
            )
        else:
            # No depth image found for this frame, fall back to RGB-only SAM 2.
            bboxes, scores = compute_bboxes_sam2(img, points, sam2_predictor)

    elif method == "combined" and sam2_predictor is not None and geco2_model is not None:
        bboxes, scores = compute_bboxes_combined(
            img, points, sam2_predictor, geco2_model, device,
        )
    elif method == "sam2" and sam2_predictor is not None:
        bboxes, scores = compute_bboxes_sam2(img, points, sam2_predictor)
    else:
        bboxes, scores = compute_bboxes_watershed(img, points)

    out_dir = xml_out_dir if xml_out_dir is not None else OUT_XML_DIR
    xml_out = out_dir / (img_path.stem + ".xml")
    write_bbox_xml(xml_out, img_path, img.shape, points, bboxes, confidences=scores)
    return len(points)


def draw_annotated_image(img_path: Path, bbox_xml_path: Path, out_path: Path) -> int:
    """Draw center dots and bounding boxes from a bbox XML onto the image and save it."""
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


def _copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy src to dst if src exists. Creates parent dirs as needed."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate bboxes from point annotations.",
        epilog=(
            "Default folder layout (all under IOCfish5kDataset/):\n"
            "  point_annotations/images/  ####.jpg        (input RGB)\n"
            "  point_annotations/color/   ####_depth.jpg  (input depth)\n"
            "  point_annotations/xml/     ####.xml        (input points)\n"
            "  annotated_images/images/   ####.jpg        (copied RGB)\n"
            "  annotated_images/color/    ####_depth.jpg  (copied depth)\n"
            "  annotated_images/xml/      ####.xml        (output bboxes)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["sam2", "watershed", "combined", "rgbd"],
        default="rgbd",
        help="Which method to use (default: rgbd)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use (default: cuda)",
    )
    parser.add_argument(
        "--geco2_checkpoint",
        type=str,
        default=None,
        help="Path to GECO2 checkpoint (.pth). Required for the combined method.",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=None,
        help="Input RGB images folder. Default: point_annotations/images/",
    )
    parser.add_argument(
        "--ann_dir",
        type=Path,
        default=None,
        help="Input point-annotation XMLs folder. Default: point_annotations/xml/",
    )
    parser.add_argument(
        "--depth_dir",
        type=Path,
        default=None,
        help=(
            "Input depth colormaps folder (<stem>_depth.jpg). "
            "Default: point_annotations/color/"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Root output folder. Default: annotated_images/. "
            "Sub-folders images/, color/, xml/ are created automatically."
        ),
    )
    parser.add_argument(
        "--vis_out_dir",
        type=Path,
        default=None,
        help="Optional folder for visualization images with drawn bboxes.",
    )
    args = parser.parse_args()

    # Resolve input directories.
    img_dir = args.img_dir.resolve() if args.img_dir else IMG_DIR
    ann_dir = args.ann_dir.resolve() if args.ann_dir else ANN_DIR

    # Resolve depth directory: explicit flag > color/ sibling of img_dir > default.
    if args.depth_dir is not None:
        depth_dir: Path | None = args.depth_dir.resolve()
    elif (img_dir.parent / "color").is_dir():
        depth_dir = img_dir.parent / "color"
    else:
        depth_dir = DEPTH_DIR if DEPTH_DIR.is_dir() else None

    # Resolve output directories.
    out_root = args.out_dir.resolve() if args.out_dir else OUT_DIR
    xml_out_dir   = out_root / "xml"
    out_img_dir   = out_root / "images"
    out_depth_dir = out_root / "color"
    vis_out_dir   = args.vis_out_dir.resolve() if args.vis_out_dir else None

    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        print(f"No .jpg files found in {img_dir}")
        return

    sam2_predictor = None
    geco2_model = None

    if args.method in ("combined", "sam2", "rgbd"):
        # GECO2 must be loaded before SAM 2 to avoid Hydra initialization conflicts.
        if args.method == "combined":
            if args.geco2_checkpoint is None:
                print("Error: --geco2_checkpoint is required for the combined method.")
                return
            print("Loading GECO2...")
            geco2_model = build_geco2_model(args.geco2_checkpoint, device=args.device)
            print("GECO2 ready.")

        if args.method == "rgbd" and depth_dir is None:
            print(
                "Warning: no depth folder found. "
                "Pass --depth_dir or place depth images in point_annotations/color/. "
                "Will fall back to RGB-only SAM 2 for every image."
            )

        print("Loading SAM 2...")
        sam2_predictor = build_sam2_predictor(device=args.device)
        print("SAM 2 ready.")

    n_depth = 0
    if depth_dir is not None:
        n_depth = len(list(depth_dir.glob("*_depth.jpg")))
    print(f"Found {len(img_files)} images, {n_depth} depth maps. Method: {args.method}.")

    # Ensure output sub-folders exist.
    for d in (xml_out_dir, out_img_dir, out_depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Writing bbox XMLs to {xml_out_dir}")
    skipped = 0
    processed = 0
    for img_path in img_files:
        ann_xml = ann_dir / (img_path.stem + ".xml")
        if not ann_xml.exists():
            skipped += 1
            continue

        count = generate_bbox_xml(
            img_path, ann_xml,
            method=args.method,
            sam2_predictor=sam2_predictor,
            geco2_model=geco2_model,
            device=args.device,
            depth_dir=depth_dir,
            xml_out_dir=xml_out_dir,
        )
        if count == 0:
            continue

        # Copy the source image and depth colormap alongside the new XML.
        _copy_if_exists(img_path, out_img_dir / img_path.name)
        if depth_dir is not None:
            depth_name = img_path.stem + "_depth.jpg"
            _copy_if_exists(depth_dir / depth_name, out_depth_dir / depth_name)

        processed += 1
        print(f"  {img_path.name}: {count} bboxes", flush=True)

    if skipped:
        print(f"  Skipped {skipped} image(s) with no matching annotation file.")
    print(f"Processed {processed} image(s). Output in {out_root}")

    # Optional: draw bboxes on images for visual inspection.
    if vis_out_dir is not None:
        vis_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Drawing annotated images to {vis_out_dir}")
        drawn = 0
        for img_path in img_files:
            bbox_xml = xml_out_dir / (img_path.stem + ".xml")
            if not bbox_xml.exists():
                continue
            out_path = vis_out_dir / img_path.name
            count = draw_annotated_image(img_path, bbox_xml, out_path)
            if count > 0:
                print(f"  {img_path.name}: {count} annotations", flush=True)
                drawn += 1
        print(f"  {drawn} visualization(s) saved.")


if __name__ == "__main__":
    main()
