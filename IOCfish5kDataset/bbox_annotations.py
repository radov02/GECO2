"""
Estimate bounding boxes from point annotations.

Available methods:
  watershed  - marker-controlled watershed on RGB, no GPU needed
  sam2       - SAM 2 with single-point prompts, needs GPU
  combined   - SAM 2 + GECO2, picks the more confident bbox per point
  rgbd       - SAM 2 on RGB and depth colormap independently, fuses results
               (default method — described in detail below)

RGBD method — step-by-step pipeline
====================================

1. **Per-point density estimation**
   A KD-tree is built over all annotation points.  For each point the
   nearest-neighbour distance is computed and compared against the median
   NN distance across the image.  Points that have >= DENSE_MIN_NEIGHBORS
   neighbours within (median_nn × DENSE_RADIUS_FACTOR) are classified as
   belonging to a dense cluster and receive a *smaller* local scale
   (nn_dist × DENSE_SCALE_SHRINK).  Isolated points keep the global
   fish-scale estimate (median NN distance × 0.7).

2. **Dual SAM 2 inference (RGB + depth)**
   SAM 2 (facebook/sam2-hiera-base-plus) is run *twice* — once on the
   original RGB image and once on the colourised depth map — using single-
   point prompts.  For dense-cluster points the RGB prompt additionally
   includes a tight *box hint* centred on the point (side = local scale)
   so that SAM 2 focuses on a small region instead of swallowing the
   entire swarm.  The **depth** run is always un-hinted, because depth
   maps naturally separate objects by distance: a tight hint there would
   artificially shrink the mask of a large object (turtle, shark, big
   fish) whose annotation point happens to sit inside a dense cluster of
   smaller neighbours.  Each run yields a per-point binary mask and a
   predicted-IoU confidence score.

3. **Per-point mask fusion (centring-first)**
   For every annotation point the RGB and depth masks are fused with the
   following precedence:
     1. **Containment** — a mask whose bbox contains the point wins over
        one that doesn't.
     2. **Hint override** — if RGB was box-hinted and the un-hinted depth
        mask is noticeably larger *and* still reasonably centred on the
        point, depth wins (rescues large objects that RGB's hint cut off).
     3. **Centring** — otherwise the mask on which the point is closer to
        the bbox centre wins.  A minimum margin (RGBD_CENTERING_DIFF) is
        required to unseat the other modality; if both are well centred,
        RGB wins.
     4. **Intersection** — if the two masks agree well (IoU ≥ 0.25) and
        the intersection still contains the point, use the intersection
        for a tighter box.
     5. **Score/compactness fall-through** — score dominance (≥ 1.5×),
        then "depth more compact at ≥ 0.8× RGB score", then the higher
        score outright (ties to RGB).
   If neither modality produces a usable mask (< 10 px) a square fallback
   box sized by the local scale is used.  Every bbox is padded by 10 %,
   shifted so the annotation centre falls inside, and capped at 95 % of
   the image dimension.

4. **Duplicate-bbox detection (post-processing)**
   SAM 2 sometimes maps many swarm points to the *same* large mask,
   producing near-identical bboxes.  A vectorised N × N pairwise IoU
   matrix is computed; connected components of bboxes with IoU ≥ 0.70
   that contain ≥ 3 members are detected via BFS.  For each such group
   the bboxes are replaced by small squares whose side equals the *median
   nearest-neighbour distance* among the group's annotation points.
   Any remaining bbox still exceeding the max-size cap is replaced by a
   box whose side equals the *median side length across all bboxes in the
   image*.

5. **GECO2 swarm refinement** (optional, enabled by default)
   Dense clusters are identified again (same KD-tree connected-component
   algorithm).  For each cluster that contains oversized SAM 2 bboxes
   (area > median cluster area × 3):
     a. Three exemplar bboxes closest to the cluster's median area are
        picked.  If any exemplar exceeds 1/8 of the image dimension the
        cluster is skipped (bad exemplars would mislead GECO2).
     b. GECO2 is run on the full image with those exemplars; the model
        predicts detection boxes and confidence scores.
     c. Each predicted box is matched to the nearest annotation point
        (preferring boxes that contain the point).
     d. Oversized SAM 2 bboxes are replaced by the GECO2 prediction;
        non-oversized bboxes are replaced only if GECO2 found a
        meaningfully tighter box (area < 80 % of SAM 2 area).

Default folder structure (inside IOCfish5kDataset/):

  point_annotations/          <-- input
    images/   ####.jpg           RGB images
    color/    ####_depth.jpg     depth colormaps
    xml/      ####.xml           point-only annotations

  auto_annotated_images/           <-- output
    images/   ####.jpg           copied from point_annotations/images/
    color/    ####_depth.jpg     copied from point_annotations/color/
    xml/      ####.xml           new XMLs with bounding boxes

When a bbox XML is written, the corresponding image and depth file
are copied into annotated_images/ so that inputs and outputs stay
together. All paths can be overridden via CLI arguments.

Usage examples
==============

tmux new -s mysession
conda activate geco_310
CUDA_VISIBLE_DEVICES=1,2 python bbox_annotations.py --num_gpus 2 --method rgbd --in_dir ~/GECO2/IOCfish5kDataset/badly_annotated/auto_annotated_images --out_dir ~/GECO2/IOCfish5kDataset/better_annotated_v3
Ctrl + b, then d (to detach and leave the process running in the background)
tmux attach -t mysession
cd /home/erik/GECO2/IOCfish5kDataset/ && zip -r better_annotated_v3.zip better_annotated_v3
# locally: scp -r -P 30688 -i C:\\Users\\radov\\.ssh\\id_ed25519 erik@proxy.vicos.si:~/GECO2/better_annotated_v3.zip /home/erik/Diploma-GECO2-with-Depth-information/GECO2/IOCfish5kDataset/_data


# 1. Default folders (point_annotations/ → auto_annotated_images/):
python bbox_annotations.py

# 2. Specify a single input root (must contain images/, xml/, color/ sub-folders)
#    and a single output root (images/, xml/, color/ are created automatically):
python bbox_annotations.py \
    --in_dir /path/to/my_input_folder \
    --out_dir /path/to/my_output_folder

# 3. Specify each input sub-folder individually:
python bbox_annotations.py \
    --img_dir /data/rgb_images \
    --ann_dir /data/xml_annotations \
    --depth_dir /data/depth_colormaps \
    --out_dir /data/output

# 4. Use a specific method and generate visualisation images:
python bbox_annotations.py \
    --method rgbd \
    --in_dir /path/to/input \
    --out_dir /path/to/output \
    --vis_out_dir /path/to/visualisations

# 5. Disable GECO2 swarm refinement (SAM2/RGBD only):
python bbox_annotations.py --no_geco2 --in_dir /path/to/input --out_dir /path/to/output

# 6. Use the watershed method (no GPU needed):
python bbox_annotations.py --method watershed --in_dir /path/to/input --out_dir /path/to/output

# 7. Custom GECO2 checkpoint and tuning parameters:
python bbox_annotations.py \
    --geco2_checkpoint /path/to/my_model.pth \
    --max_bbox_frac 0.4 \
    --dense_min_neighbors 5 \
    --in_dir /path/to/input \
    --out_dir /path/to/output
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# Add the GECO2 root so that local packages (models, configs, …) are importable
# and so that "import sam2" resolves to GECO2/sam2/ (the repo directory whose
# inner sam2/ sub-package is the actual library).  With GECO2/ on sys.path the
# import path used in GECO2's own model files  (from sam2.sam2.modeling…)
# resolves correctly without needing the package installed.
SAM2_ROOT = Path(__file__).resolve().parent.parent
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

# Convenience alias — used for config paths and checkpoint defaults below.
SAM2_REPO = SAM2_ROOT / "sam2"

# Default paths relative to the dataset folder.
# Input lives under point_annotations/, output under annotated_images/.
BASE_DIR       = Path(__file__).parent
IN_DIR         = BASE_DIR / "point_annotations"
IMG_DIR        = IN_DIR / "images"
ANN_DIR        = IN_DIR / "xml"
DEPTH_DIR      = IN_DIR / "color"
OUT_DIR        = BASE_DIR / "auto_annotated_images"
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
RGBD_CENTERING_GOOD  = 0.5   # point within half the bbox radius of centre counts as "well-centred"
RGBD_CENTERING_DIFF  = 0.20  # one modality must beat the other's centring by this much to win
RGBD_SIZE_DOMINANCE  = 2.0   # when both masks are well-centred, the one this much larger wins

# Post-processing: enforce centrality of the annotation point in the final bbox.
RECENTER_MAX_ASYMMETRY = 0.5  # shrink the long side of an axis when |l-r|/width > this

# Dense-cluster and bbox-size-cap settings.
DENSE_RADIUS_FACTOR = 1.5     # points within median_nn * this factor are "neighbors"
DENSE_MIN_NEIGHBORS = 3       # need at least this many neighbours to count as dense
DENSE_SCALE_SHRINK  = 0.45    # in a dense cluster, local scale = nn_dist * this
MAX_BBOX_FRAC       = 0.95    # bbox may not span (nearly) the full image width/height
OVERSIZE_FALLBACK   = 0.15    # when bbox spans the full image, replace with this fraction
SWARM_OVERSIZE_THR  = 3.0     # bbox area > median_cluster_area * this = "oversized"
MULTIPOINT_SIZE_THR = 3.0     # multipoint bbox replaced only if side > this * typical side
MULTIPOINT_MIN_PTS  = 3       # ... and only if it contains at least this many points
MULTIPOINT_OWN_CENTRED = 0.4  # ... and only if its own point is NOT this well-centred
MULTIPOINT_LARGE_OBJ_THR = 5.0  # own-centred exemption applies only above this * typical side

# Duplicate-bbox detection settings.
DUPLICATE_IOU_THR     = 0.70  # IoU above this means "same bbox" (SAM2 swarm failure)
DUPLICATE_MIN_GROUP   = 3     # min near-identical bboxes to trigger replacement

# GECO2 model settings.
GECO2_DEFAULT_CKPT = str(
    SAM2_ROOT / "CNTQG_multitrain_ca44.pth"
)  # default checkpoint; override with --geco2_checkpoint
GECO2_MAX_EXEMPLAR_FRAC = 0.125  # 1/8 -- exemplars larger than this are useless for GECO2
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
    diff = pts[:, None, :] - pts[None, :, :]        # (N, N, 2)
    dists = np.sqrt((diff * diff).sum(axis=2))       # (N, N)
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    return max(15, int(np.median(nn) * 0.7))


def _per_point_local_scale(
    points: list[tuple[int, int]], h: int, w: int,
) -> list[int]:
    """Return a per-point scale that is tighter for points in dense clusters.

    Uses a KD-tree (O(n log n)) to count neighbours within a radius derived
    from the median nearest-neighbour distance.  Points that have many close
    neighbours get a smaller scale so that bbox generation is nudged towards
    smaller boxes.  Isolated points keep the global fish-scale estimate.
    """
    from scipy.spatial import cKDTree

    global_scale = _fish_scale(points, h, w)
    n = len(points)
    if n <= 2:
        return [global_scale] * n

    pts = np.array(points, dtype=np.float64)
    tree = cKDTree(pts)

    # Nearest-neighbour distance per point (k=2 because first hit is itself).
    nn_dists, _ = tree.query(pts, k=2)
    nn1 = nn_dists[:, 1]                       # distance to closest other point
    median_nn = float(np.median(nn1))
    radius = median_nn * DENSE_RADIUS_FACTOR

    # Count neighbours within the radius for every point.
    neigh_counts = tree.query_ball_point(pts, radius, return_length=True)
    # query_ball_point counts the point itself, so subtract 1.
    neigh_counts = np.asarray(neigh_counts) - 1

    scales: list[int] = []
    for i in range(n):
        if neigh_counts[i] >= DENSE_MIN_NEIGHBORS:
            # Dense cluster: shrink scale based on local nearest-neighbour.
            local = max(10, int(nn1[i] * DENSE_SCALE_SHRINK))
            scales.append(local)
        else:
            scales.append(global_scale)
    return scales


def _cap_bbox(
    bx: int, by: int, bw: int, bh: int,
    cx: int, cy: int, h: int, w: int,
) -> tuple[int, int, int, int]:
    """Enforce the maximum bbox size (MAX_BBOX_FRAC of image dims).

    If either dimension exceeds the limit the bbox is replaced by a tiny
    fallback box (OVERSIZE_FALLBACK of image dim) centred on the point.
    """
    max_w = max(1, int(w * MAX_BBOX_FRAC))
    max_h = max(1, int(h * MAX_BBOX_FRAC))
    if bw > max_w or bh > max_h:
        fb_w = max(8, int(w * OVERSIZE_FALLBACK))
        fb_h = max(8, int(h * OVERSIZE_FALLBACK))
        bx = max(0, cx - fb_w // 2)
        by = max(0, cy - fb_h // 2)
        bw = min(fb_w, w - bx)
        bh = min(fb_h, h - by)
    return bx, by, bw, bh


def _iou_matrix_xywh(bboxes: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU for N bboxes in (x, y, w, h) format.

    Returns an (N, N) float64 matrix.  Fully numpy-vectorized, no loops.
    """
    x1 = bboxes[:, 0].astype(np.float64)
    y1 = bboxes[:, 1].astype(np.float64)
    x2 = x1 + bboxes[:, 2]
    y2 = y1 + bboxes[:, 3]
    areas = bboxes[:, 2].astype(np.float64) * bboxes[:, 3]

    # Pairwise intersection via broadcasting.
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    union = areas[:, None] + areas[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _fix_multipoint_bboxes(
    points: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int, int, int]]:
    """Collapse bboxes that swallow multiple annotation points into small per-point boxes.

    A bbox is considered "over-grouping" when it contains >=
    MULTIPOINT_MIN_PTS annotation points *and* is noticeably larger than the
    typical single-point bbox in the image (side > MULTIPOINT_SIZE_THR *
    median single-point side).

    The well-centred own-point exemption (centring ratio <=
    MULTIPOINT_OWN_CENTRED) only applies when the bbox is a *clear*
    large-object outlier (side >= MULTIPOINT_LARGE_OBJ_THR * typical side).
    In that regime a centred own-point is strong evidence of a legitimate
    large object (shark, manta) that happens to overlap some smaller fish
    annotations.  Below that threshold, a merely-moderately-oversized bbox
    with a centred own-point is more likely a SAM2 swarm-grouping failure
    (especially in dense-swarm images where many small fish occasionally
    get lumped into a single mask whose centroid happens to land near one
    of their annotation points), and is still collapsed.

    Each over-grouping bbox that fails the exemption is replaced by a
    square centred on its own point, sized by that point's nearest-
    neighbour distance (capped at the typical single-point side so
    replacements don't balloon).
    """
    n = len(bboxes)
    if n == 0:
        return list(bboxes)

    pts_arr = np.array(points, dtype=np.float64)
    bb_arr = np.array(bboxes, dtype=np.float64)

    x1 = bb_arr[:, 0]
    y1 = bb_arr[:, 1]
    x2 = x1 + bb_arr[:, 2]
    y2 = y1 + bb_arr[:, 3]

    # contains[i, j] = bbox i contains point j
    px = pts_arr[:, 0]
    py = pts_arr[:, 1]
    contains = (
        (px[None, :] >= x1[:, None]) & (px[None, :] <= x2[:, None])
        & (py[None, :] >= y1[:, None]) & (py[None, :] <= y2[:, None])
    )
    num_contained = contains.sum(axis=1)

    # Reference side: median of bboxes containing exactly one point.
    single_mask = num_contained <= 1
    if single_mask.sum() > 0:
        ref_side = float(np.median(
            (bb_arr[single_mask, 2] + bb_arr[single_mask, 3]) / 2.0
        ))
        ref_side = max(15.0, ref_side)
    else:
        ref_side = max(15.0, min(h, w) / 40.0)

    # Per-point NN distance for sizing the replacement boxes.
    if n > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_arr)
        nn_d, _ = tree.query(pts_arr, k=2)
        nn_dists = nn_d[:, 1]
    else:
        nn_dists = np.array([ref_side])

    out = list(bboxes)
    for i in range(n):
        if num_contained[i] < MULTIPOINT_MIN_PTS:
            continue
        bbox_side = (bb_arr[i, 2] + bb_arr[i, 3]) / 2.0
        if bbox_side <= ref_side * MULTIPOINT_SIZE_THR:
            continue  # near-typical size, likely genuine overlap — keep as-is
        # The "own point is well-centred → keep" exemption only fires when
        # the bbox is a *clear* large-object outlier (>= MULTIPOINT_LARGE_OBJ_THR
        # * ref_side).  A merely-oversized bbox (3–5× typical fish) whose own
        # point is centred is more likely a SAM2 swarm-grouping failure than a
        # legitimate large object — the grouped mask's centroid tends to land
        # near one of the enclosed annotation points and would otherwise
        # escape collapse.
        own_ratio = _bbox_centering(
            (int(bb_arr[i, 0]), int(bb_arr[i, 1]),
             int(bb_arr[i, 2]), int(bb_arr[i, 3])),
            int(points[i][0]), int(points[i][1]),
        )
        if (
            own_ratio <= MULTIPOINT_OWN_CENTRED
            and bbox_side > ref_side * MULTIPOINT_LARGE_OBJ_THR
        ):
            continue
        cx, cy = points[i]
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))
        side = max(15, int(min(nn_dists[i] * 0.7, ref_side)))
        half = side // 2
        bx = max(0, cx_c - half)
        by = max(0, cy_c - half)
        bw = min(side, w - bx)
        bh = min(side, h - by)
        out[i] = (bx, by, bw, bh)
    return out


def _recenter_off_center_bboxes(
    points: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    threshold: float = RECENTER_MAX_ASYMMETRY,
) -> list[tuple[int, int, int, int]]:
    """Crop bboxes whose annotation point is significantly off-centre.

    When SAM 2 over-segments (e.g. the mask extends beyond the target fish
    into a neighbour), the annotation point ends up far from one edge and
    close to the opposite edge.  For each axis, we compute the per-axis
    asymmetry ``|l - r| / bbox_width`` (0 = centred, 1 = point at the edge).
    When it exceeds *threshold*, the long side is cropped so the new half-
    size equals the short side's distance to the point — mirroring the
    short side across the point.  This trims the spurious extension that
    swallowed a neighbour while preserving the axis where the bbox was
    already centred.

    Each axis is handled independently so a bbox that is centred on one
    axis but not the other is only cropped on the bad axis.
    """
    out: list[tuple[int, int, int, int]] = []
    for (cx, cy), (bx, by, bw, bh) in zip(points, bboxes):
        if bw <= 0 or bh <= 0:
            out.append((bx, by, bw, bh))
            continue

        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))

        new_bx, new_bw = bx, bw
        new_by, new_bh = by, bh

        left = cx_c - bx
        right = bx + bw - cx_c
        if left > 0 and right > 0:
            dx_ratio = abs(left - right) / float(bw)
            if dx_ratio > threshold:
                half = max(1, min(left, right))
                new_bx = cx_c - half
                new_bw = 2 * half

        top = cy_c - by
        bottom = by + bh - cy_c
        if top > 0 and bottom > 0:
            dy_ratio = abs(top - bottom) / float(bh)
            if dy_ratio > threshold:
                half = max(1, min(top, bottom))
                new_by = cy_c - half
                new_bh = 2 * half

        # Clamp to image bounds.
        new_bx = max(0, new_bx)
        new_by = max(0, new_by)
        new_bw = max(1, min(new_bw, w - new_bx))
        new_bh = max(1, min(new_bh, h - new_by))
        out.append((new_bx, new_by, new_bw, new_bh))
    return out


def _enforce_point_inside_bbox(
    points: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int, int, int]]:
    """Final invariant: guarantee each bbox strictly contains its annotation point.

    Only expands the bbox (never shrinks) so that existing content is preserved
    while the point is brought inside. Runs last in the pipeline to catch cases
    where earlier stages (mask fusion, GECO2 matching, etc.) produced a bbox
    whose point drifted outside.
    """
    out: list[tuple[int, int, int, int]] = []
    for (cx, cy), (bx, by, bw, bh) in zip(points, bboxes):
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))

        # Extend left/up if point is before the bbox start (keep right/bottom edge).
        if cx_c < bx:
            bw += bx - cx_c
            bx = cx_c
        if cy_c < by:
            bh += by - cy_c
            by = cy_c
        # Extend right/down if point is at/after the bbox end.
        if cx_c >= bx + bw:
            bw = cx_c - bx + 1
        if cy_c >= by + bh:
            bh = cy_c - by + 1

        bx = max(0, bx)
        by = max(0, by)
        bw = max(1, min(bw, w - bx))
        bh = max(1, min(bh, h - by))
        out.append((bx, by, bw, bh))
    return out


def _postprocess_bboxes(
    points: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    scores: list[float],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Fix near-duplicate bboxes and cap any remaining oversized ones.

    **Duplicate detection** (vectorized via IoU matrix):
    SAM 2 often maps many swarm points onto the *same* large mask, producing
    near-identical bboxes.  Groups of >= DUPLICATE_MIN_GROUP bboxes with
    pairwise IoU >= DUPLICATE_IOU_THR are detected and replaced with small
    boxes whose side length equals the *median nearest-neighbour distance*
    among the points in the group.

    **Oversized cap**:
    Any remaining bbox larger than MAX_BBOX_FRAC of the image is replaced
    with a box whose side is the *median side length across all (already
    fixed) bboxes in the image*.
    """
    n = len(bboxes)
    if n == 0:
        return bboxes, scores

    bboxes = list(bboxes)
    scores = list(scores)
    arr = np.array(bboxes, dtype=np.float64)      # (N, 4)
    pts = np.array(points, dtype=np.float64)       # (N, 2)

    # ------- Step 1: detect and fix duplicate bboxes in swarms -------
    if n >= DUPLICATE_MIN_GROUP:
        iou = _iou_matrix_xywh(arr)
        np.fill_diagonal(iou, 0.0)
        adj = iou >= DUPLICATE_IOU_THR              # (N, N) bool

        # Connected components via BFS on the adjacency matrix.
        visited = np.zeros(n, dtype=bool)
        dup_groups: list[list[int]] = []
        for seed in range(n):
            if visited[seed]:
                continue
            # Quick skip: no chance of forming a big-enough group.
            if adj[seed].sum() < DUPLICATE_MIN_GROUP - 1:
                visited[seed] = True
                continue
            queue = [seed]
            visited[seed] = True
            component: list[int] = []
            while queue:
                node = queue.pop()
                component.append(node)
                nbs = np.flatnonzero(adj[node] & ~visited)
                visited[nbs] = True
                queue.extend(nbs.tolist())
            if len(component) >= DUPLICATE_MIN_GROUP:
                dup_groups.append(component)

        for group in dup_groups:
            g_pts = pts[group]                      # (G, 2)
            g = len(group)
            # Vectorized pairwise NN distance within the group.
            if g >= 2:
                diffs = g_pts[:, None, :] - g_pts[None, :, :]   # (G, G, 2)
                dists = np.sqrt((diffs * diffs).sum(axis=2))     # (G, G)
                np.fill_diagonal(dists, np.inf)
                nn_d = dists.min(axis=1)                         # (G,)
                side = max(8, int(np.median(nn_d)))
            else:
                side = max(8, int((arr[:, 2] + arr[:, 3]).mean() / 2))

            half = side // 2
            for idx in group:
                cx, cy = points[idx]
                cx_c = int(np.clip(cx, 0, w - 1))
                cy_c = int(np.clip(cy, 0, h - 1))
                bx = max(0, cx_c - half)
                by = max(0, cy_c - half)
                bw = min(side, w - bx)
                bh = min(side, h - by)
                bboxes[idx] = (bx, by, bw, bh)
                # Update the array for step 2.
                arr[idx] = [bx, by, bw, bh]

    # ------- Step 2: cap remaining oversized bboxes -------
    # Median side across all bboxes (now including the fixed swarm boxes).
    all_sides = (arr[:, 2] + arr[:, 3]) / 2.0
    median_side_global = max(8, int(np.median(all_sides)))
    max_bw = max(1, int(w * MAX_BBOX_FRAC))
    max_bh = max(1, int(h * MAX_BBOX_FRAC))

    for i in range(n):
        bx, by, bw, bh = bboxes[i]
        if bw > max_bw or bh > max_bh:
            fb = median_side_global
            half = fb // 2
            cx, cy = points[i]
            cx_c = int(np.clip(cx, 0, w - 1))
            cy_c = int(np.clip(cy, 0, h - 1))
            bx = max(0, cx_c - half)
            by = max(0, cy_c - half)
            bw = min(fb, w - bx)
            bh = min(fb, h - by)
            bboxes[i] = (bx, by, bw, bh)

    return bboxes, scores


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
    from sam2.sam2.build_sam import build_sam2_hf
    from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (will be slow).")
        device = "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    sam2_configs_dir = str(SAM2_ROOT / "sam2" / "sam2_configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=sam2_configs_dir, version_base="1.2"):
        model = build_sam2_hf(SAM2_HF_MODEL_ID, device=device)
    return SAM2ImagePredictor(model)


def _sam2_predict_per_point(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
    local_scales: list[int] | None = None,
    use_box_hint: bool = True,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[float],
    list[np.ndarray | None],
]:
    """
    Run SAM 2 on each center point and return bboxes, pIoU scores, and masks.
    Masks are None for points where SAM 2 returned a too-small or empty mask.

    If *local_scales* is provided (one int per point), points whose local
    scale is smaller than the global fish-scale estimate are additionally
    prompted with a tight box hint so that SAM 2 focuses on a small region
    instead of swallowing the whole swarm.  Set *use_box_hint* to False to
    skip the hint entirely — useful on depth colormaps where large objects
    are already easy to segment and the hint would artificially shrink the
    mask of any large object whose annotation point happens to sit near
    smaller neighbours.
    """
    h, w = img_bgr.shape[:2]
    if not points:
        return [], [], []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    global_scale = _fish_scale(points, h, w)
    if local_scales is None:
        local_scales = [global_scale] * len(points)

    bboxes: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []
    raw_masks: list[np.ndarray | None] = []

    for idx, (cx, cy) in enumerate(points):
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))
        lscale = local_scales[idx]

        # Build SAM 2 prompt.  For dense-cluster points, add a box hint
        # centred on the point so the model doesn't over-segment.
        predict_kwargs: dict = dict(
            point_coords=np.array([[cx_c, cy_c]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        if use_box_hint and lscale < global_scale:
            # Provide a tight box prompt around the point to constrain SAM 2.
            half = lscale
            bx0 = max(0, cx_c - half)
            by0 = max(0, cy_c - half)
            bx1 = min(w - 1, cx_c + half)
            by1 = min(h - 1, cy_c + half)
            predict_kwargs["box"] = np.array(
                [bx0, by0, bx1, by1], dtype=np.float32,
            )

        masks, scores, _ = predictor.predict(**predict_kwargs)

        best_idx = scores.argmax()
        confidence = float(scores[best_idx])
        mask = masks[best_idx]

        ys, xs = np.nonzero(mask)
        if len(ys) >= SAM2_MIN_MASK_PX:
            bx = int(xs.min())
            by = int(ys.min())
            bw = int(xs.max()) - bx + 1
            bh = int(ys.max()) - by + 1
            raw_masks.append(mask.astype(bool))
        else:
            confidence = 0.0
            half = lscale // 2
            bx = max(0, cx_c - half)
            by = max(0, cy_c - half)
            bw = min(lscale, w - bx)
            bh = min(lscale, h - by)
            raw_masks.append(None)

        bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
        bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx_c, cy_c, h, w)
        bx, by, bw, bh = _cap_bbox(bx, by, bw, bh, cx_c, cy_c, h, w)
        bboxes.append((bx, by, bw, bh))
        confidences.append(confidence)

    predictor.reset_predictor()
    return bboxes, confidences, raw_masks


def _find_dense_clusters(
    points: list[tuple[int, int]], h: int, w: int,
) -> list[list[int]]:
    """Return groups of point indices that form dense clusters.

    Each returned list contains the *indices* into *points* that belong to one
    cluster.  Only clusters with >= DENSE_MIN_NEIGHBORS+1 members are returned.
    Uses DBSCAN-style connected-component grouping via the KD-tree.
    """
    from scipy.spatial import cKDTree

    n = len(points)
    if n < DENSE_MIN_NEIGHBORS + 1:
        return []

    pts = np.array(points, dtype=np.float64)
    tree = cKDTree(pts)

    nn_dists, _ = tree.query(pts, k=2)
    median_nn = float(np.median(nn_dists[:, 1]))
    radius = median_nn * DENSE_RADIUS_FACTOR

    # Build adjacency via ball query and extract connected components.
    neighbours = tree.query_ball_point(pts, radius)
    visited = [False] * n
    clusters: list[list[int]] = []
    for seed in range(n):
        if visited[seed]:
            continue
        # BFS
        queue = [seed]
        visited[seed] = True
        component: list[int] = []
        while queue:
            node = queue.pop()
            component.append(node)
            for nb in neighbours[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        if len(component) >= DENSE_MIN_NEIGHBORS + 1:
            clusters.append(component)
    return clusters


def _pick_median_exemplars(
    indices: list[int],
    bboxes: list[tuple[int, int, int, int]],
    n: int = GECO2_NUM_OBJECTS,
) -> list[tuple[int, int, int, int]]:
    """Pick *n* exemplar bboxes whose area is closest to the cluster median."""
    areas = [(bboxes[i][2] * bboxes[i][3], i) for i in indices]
    areas.sort()
    median_area = areas[len(areas) // 2][0]
    # Sort by distance to median area, take the n closest.
    by_dist = sorted(areas, key=lambda t: abs(t[0] - median_area))
    chosen = [bboxes[by_dist[j][1]] for j in range(min(n, len(by_dist)))]
    while len(chosen) < n:
        chosen.append(chosen[-1])
    return chosen


def _refine_swarm_with_geco2(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    scores: list[float],
    geco2_model,
    device: str = "cuda",
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Refine bboxes in dense swarms using GECO2.

    For each detected dense cluster:
      1. Pick 3 median-area bboxes from the cluster as exemplars.
      2. Run GECO2 on the full image with those exemplars.
      3. For each cluster point whose SAM2 bbox is oversized relative to
         the cluster median, replace it with the GECO2 bbox if GECO2
         produced a valid (non-zero) detection.

    Points outside any dense cluster, or whose SAM2 bbox is already
    reasonable, are left untouched.
    """
    if geco2_model is None or not points:
        return bboxes, scores

    h, w = img_bgr.shape[:2]
    clusters = _find_dense_clusters(points, h, w)
    if not clusters:
        return bboxes, scores

    bboxes = list(bboxes)   # make mutable copies
    scores = list(scores)

    for cluster_idx in clusters:
        # Compute median bbox area in this cluster.
        areas = [bboxes[i][2] * bboxes[i][3] for i in cluster_idx]
        median_area = float(np.median(areas))

        # Identify cluster points with oversized bboxes.
        oversized = [
            i for i in cluster_idx
            if bboxes[i][2] * bboxes[i][3] > median_area * SWARM_OVERSIZE_THR
        ]
        if not oversized:
            continue  # all bboxes in this cluster are reasonable

        # Pick 3 median exemplars from the cluster.
        exemplars = _pick_median_exemplars(cluster_idx, bboxes)

        # Skip GECO2 if exemplars are too large (> 1/8 of image dim).
        max_ex_w = w * GECO2_MAX_EXEMPLAR_FRAC
        max_ex_h = h * GECO2_MAX_EXEMPLAR_FRAC
        if any(ex[2] > max_ex_w or ex[3] > max_ex_h for ex in exemplars):
            continue

        # Collect all cluster points and run GECO2 on them.
        cluster_points = [points[i] for i in cluster_idx]
        geco2_bboxes, geco2_scores = compute_bboxes_geco2(
            img_bgr, cluster_points, exemplars, geco2_model, device,
        )

        # Map cluster-local index back to global index.
        for local_j, global_i in enumerate(cluster_idx):
            gbx, gby, gbw, gbh = geco2_bboxes[local_j]
            g_area = gbw * gbh
            s_area = bboxes[global_i][2] * bboxes[global_i][3]

            # Replace if SAM2 bbox is oversized and GECO2 gave a valid box.
            if global_i in oversized and g_area > 0:
                cx, cy = points[global_i]
                cx_c = int(np.clip(cx, 0, w - 1))
                cy_c = int(np.clip(cy, 0, h - 1))
                gbx, gby, gbw, gbh = _cap_bbox(
                    gbx, gby, gbw, gbh, cx_c, cy_c, h, w,
                )
                bboxes[global_i] = (gbx, gby, gbw, gbh)
                scores[global_i] = geco2_scores[local_j]
            # Also replace non-oversized points if GECO2 found a tighter box.
            elif g_area > 0 and g_area < s_area * 0.8:
                cx, cy = points[global_i]
                cx_c = int(np.clip(cx, 0, w - 1))
                cy_c = int(np.clip(cy, 0, h - 1))
                gbx, gby, gbw, gbh = _cap_bbox(
                    gbx, gby, gbw, gbh, cx_c, cy_c, h, w,
                )
                bboxes[global_i] = (gbx, gby, gbw, gbh)
                scores[global_i] = geco2_scores[local_j]

    return bboxes, scores


def compute_bboxes_sam2(
    img_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
    geco2_model=None,
    device: str = "cuda",
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Run SAM 2 on the RGB image, then refine dense swarms with GECO2."""
    h, w = img_bgr.shape[:2]
    local_scales = _per_point_local_scale(points, h, w)
    bboxes, confidences, _ = _sam2_predict_per_point(
        img_bgr, points, predictor, local_scales=local_scales,
    )
    # Fix duplicate / oversized bboxes before GECO2 sees them.
    bboxes, confidences = _postprocess_bboxes(points, bboxes, confidences, h, w)
    bboxes, confidences = _refine_swarm_with_geco2(
        img_bgr, points, bboxes, confidences, geco2_model, device,
    )
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
        bx, by, bw, bh = _cap_bbox(bx, by, bw, bh, cx_c, cy_c, h, w)
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
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    return model


def _preprocess_for_geco2(
    img_bgr: np.ndarray,
    exemplar_bboxes_xyxy: list[tuple[int, int, int, int]],
    image_size: int = GECO2_IMAGE_SIZE,
    device: str = "cpu",
):
    """Resize and pad the image to GECO2 input size, scaling exemplar boxes accordingly.

    Returns (img_tensor, exemplar_tensor, scaling_factor, (pad_w, pad_h)).
    Tensors are placed on *device* so the interpolation and padding run on GPU.
    """
    import torch

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Move to target device before interpolation so the resize runs on GPU.
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div_(255.0).to(device)
    _, oh, ow = img_t.shape
    longer = max(oh, ow)
    sf = image_size / longer

    ex_t = torch.tensor(exemplar_bboxes_xyxy, dtype=torch.float32, device=device)
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

    no_boxes = len(boxes_xyxy) == 0
    centres = None if no_boxes else (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2

    for cx, cy in points:
        if no_boxes:
            result_bboxes.append((0, 0, 0, 0))
            result_scores.append(0.0)
            continue

        contained = (
            (boxes_xyxy[:, 0] <= cx) & (cx <= boxes_xyxy[:, 2])
            & (boxes_xyxy[:, 1] <= cy) & (cy <= boxes_xyxy[:, 3])
        )
        if contained.any():
            best_idx = int(np.where(contained, scores, -np.inf).argmax())
        else:
            dists = np.hypot(centres[:, 0] - cx, centres[:, 1] - cy)
            best_idx = int(dists.argmin())

        best_score = float(scores[best_idx])
        x1, y1, x2, y2 = boxes_xyxy[best_idx]
        bx = max(0, int(x1))
        by = max(0, int(y1))
        bw = min(max(1, int(x2 - x1)), w - bx)
        bh = min(max(1, int(y2 - y1)), h - by)
        bx, by, bw, bh = _cap_bbox(bx, by, bw, bh, cx, cy, h, w)
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
        img_bgr, ex_xyxy, device=device,
    )
    img_tensor = img_tensor.unsqueeze(0)
    ex_tensor = ex_tensor.unsqueeze(0)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ref, _cent, _coord, _masks = geco2_model(img_tensor, ex_tensor)
        else:
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

    sam2_bboxes, sam2_scores = compute_bboxes_sam2(
        img_bgr, points, sam2_predictor,
    )

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

    h, w = img_bgr.shape[:2]
    final_bboxes: list[tuple[int, int, int, int]] = []
    final_scores: list[float] = []
    for i in range(len(points)):
        cx, cy = points[i]
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))
        if geco2_scores[i] > sam2_scores[i]:
            bx, by, bw, bh = _cap_bbox(*geco2_bboxes[i], cx_c, cy_c, h, w)
            final_bboxes.append((bx, by, bw, bh))
            final_scores.append(geco2_scores[i])
        else:
            bx, by, bw, bh = _cap_bbox(*sam2_bboxes[i], cx_c, cy_c, h, w)
            final_bboxes.append((bx, by, bw, bh))
            final_scores.append(sam2_scores[i])

    return final_bboxes, final_scores


def _mask_to_bbox_xywh(
    mask: np.ndarray, cx: int, cy: int, h: int, w: int, scale: int,
) -> tuple[int, int, int, int]:
    """Convert a binary mask to (x, y, w, h). Falls back to a square box if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) >= SAM2_MIN_MASK_PX:
        bx = int(xs.min())
        by = int(ys.min())
        bw = int(xs.max()) - bx + 1
        bh = int(ys.max()) - by + 1
    else:
        half = scale // 2
        bx = max(0, cx - half)
        by = max(0, cy - half)
        bw = min(scale, w - bx)
        bh = min(scale, h - by)
    bx, by, bw, bh = _pad_box(bx, by, bw, bh, h, w)
    bx, by, bw, bh = _ensure_center_inside(bx, by, bw, bh, cx, cy, h, w)
    bx, by, bw, bh = _cap_bbox(bx, by, bw, bh, cx, cy, h, w)
    return bx, by, bw, bh


def _mask_bbox_centering(
    mask: np.ndarray | None, cx: int, cy: int,
) -> float:
    """Return how far the annotation point is from the mask-bbox centre.

    The value is normalised by the bbox half-size, so:
      * 0.0  = perfectly centred,
      * ~1.0 = at the bbox edge,
      * >1.0 = the point lies outside the bbox altogether,
      * inf  = mask is missing or too small to trust.
    """
    if mask is None:
        return float("inf")
    ys, xs = np.nonzero(mask)
    if len(ys) < SAM2_MIN_MASK_PX:
        return float("inf")
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    cx_box = (xmin + xmax) / 2.0
    cy_box = (ymin + ymax) / 2.0
    half_w = max(1.0, (xmax - xmin) / 2.0)
    half_h = max(1.0, (ymax - ymin) / 2.0)
    dx = (float(cx) - cx_box) / half_w
    dy = (float(cy) - cy_box) / half_h
    return float((dx * dx + dy * dy) ** 0.5)


def _bbox_centering(
    bbox: tuple[int, int, int, int], cx: int, cy: int,
) -> float:
    """Same as :func:`_mask_bbox_centering` but directly from an (x, y, w, h) box."""
    bx, by, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return float("inf")
    cx_box = bx + bw / 2.0
    cy_box = by + bh / 2.0
    half_w = max(1.0, bw / 2.0)
    half_h = max(1.0, bh / 2.0)
    dx = (float(cx) - cx_box) / half_w
    dy = (float(cy) - cy_box) / half_h
    return float((dx * dx + dy * dy) ** 0.5)


def _mask_bbox_contains_point(
    mask: np.ndarray | None, cx: int, cy: int,
) -> bool:
    """True iff the mask's bounding rect contains the annotation point."""
    if mask is None:
        return False
    ys, xs = np.nonzero(mask)
    if len(ys) < SAM2_MIN_MASK_PX:
        return False
    return int(xs.min()) <= cx <= int(xs.max()) and int(ys.min()) <= cy <= int(ys.max())


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
    rgb_was_hinted: bool = False,
) -> tuple[tuple[int, int, int, int], float]:
    """
    Fuse the RGB and depth SAM 2 masks for one point into a single bbox.

    Precedence rules (highest first):
      0. Containment — a mask whose bounding rect contains the annotation
         point beats one whose bounding rect does not.
      1. Hint override — when the RGB run was given a tight box-hint (dense
         cluster point) the RGB mask is an artificial lower bound.  If the
         un-hinted depth mask contains the point, is reasonably well-centred
         on it, and is larger than the RGB mask, trust depth — it likely
         found a large object that the RGB hint cut off (e.g. a turtle
         swimming amongst small fish).
      2. Size-when-both-centred — when both masks contain the point *and*
         both are well-centred on it (ratio <= RGBD_CENTERING_GOOD), prefer
         the substantially larger one (>= RGBD_SIZE_DOMINANCE× area).  This
         rescues large, un-hinted objects (shark, manta) where one modality
         only captured a fragment and the other captured the full body:
         without this rule the fragment mask would win on centring alone
         (a tiny bbox is trivially centred on its own point).
      3. Centering — of the masks that contain the point, prefer the one
         whose bbox is better centred on the annotation point.  A
         RGBD_CENTERING_DIFF margin is required to unseat the other
         modality; if both are well-centred within the tolerance, RGB wins.
      4. Intersection — if the two masks agree (IoU >= RGBD_IOU_AGREE_THR)
         and the intersection still contains the point, use the intersection
         for a tighter box.
      5. Score dominance / depth compactness / higher-score fall-through
         (as in previous versions).
    If only one mask is valid we use it directly.
    If neither is valid we fall back to a square box sized by fish scale.
    """
    rgb_ok = rgb_mask is not None and rgb_mask.sum() >= SAM2_MIN_MASK_PX
    depth_ok = depth_mask is not None and depth_mask.sum() >= SAM2_MIN_MASK_PX

    if rgb_ok and depth_ok:
        rgb_center = _mask_bbox_centering(rgb_mask, cx, cy)
        depth_center = _mask_bbox_centering(depth_mask, cx, cy)
        rgb_has_pt = rgb_center < 1.0
        depth_has_pt = depth_center < 1.0

        # 0. Containment wins outright when only one modality contains the point.
        if rgb_has_pt and not depth_has_pt:
            return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score
        if depth_has_pt and not rgb_has_pt:
            return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

        if rgb_has_pt and depth_has_pt:
            rgb_area = float(rgb_mask.sum())
            depth_area = float(depth_mask.sum())

            # 1. Hint override: an un-hinted depth mask that is clearly
            # larger and reasonably centred on the point overrules a
            # hint-tightened RGB bbox.  This rescues large objects
            # (turtle, big fish) whose point happens to sit in a dense
            # cluster and would otherwise get a matchbox-sized bbox.
            if rgb_was_hinted:
                if depth_area > rgb_area * 1.5 and depth_center <= RGBD_CENTERING_GOOD:
                    return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

            # 2. Size-when-both-centred: pure centring is insufficient
            # because any small mask is trivially centred on its own point.
            # When both modalities are well-centred, the substantially
            # larger one more likely reflects the true object extent
            # (catches un-hinted large objects like sharks and mantas
            # whose RGB-only mask may only cover a fragment of the body).
            if (
                rgb_center <= RGBD_CENTERING_GOOD
                and depth_center <= RGBD_CENTERING_GOOD
            ):
                if depth_area > rgb_area * RGBD_SIZE_DOMINANCE:
                    return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score
                if rgb_area > depth_area * RGBD_SIZE_DOMINANCE:
                    return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score

            # 3. Centering-first comparison.
            if depth_center + RGBD_CENTERING_DIFF < rgb_center:
                return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score
            if rgb_center + RGBD_CENTERING_DIFF < depth_center:
                return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score
            if rgb_center <= RGBD_CENTERING_GOOD and depth_center <= RGBD_CENTERING_GOOD:
                # Both well-centred within the tolerance and similar size — prefer RGB.
                return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score

        # 3. Intersection for agreeing masks.
        intersection = rgb_mask & depth_mask
        union = rgb_mask | depth_mask
        union_px = union.sum()
        iou = float(intersection.sum()) / max(float(union_px), 1.0)
        if (
            iou >= RGBD_IOU_AGREE_THR
            and intersection.sum() >= SAM2_MIN_MASK_PX
            and _mask_bbox_contains_point(intersection, cx, cy)
        ):
            conf = max(rgb_score, depth_score) * (0.5 + 0.5 * iou)
            return _mask_to_bbox_xywh(intersection, cx, cy, h, w, scale), conf

        # 4a. One modality is much more confident, trust it.
        if rgb_score > depth_score * RGBD_SCORE_DOMINANCE:
            return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score
        if depth_score > rgb_score * RGBD_SCORE_DOMINANCE:
            return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

        # 4b. Depth mask is more compact and reasonably confident, prefer it.
        rgb_area = float(rgb_mask.sum())
        depth_area = float(depth_mask.sum())
        if depth_area < rgb_area and depth_score >= rgb_score * RGBD_DEPTH_COMPACT_W:
            return _mask_to_bbox_xywh(depth_mask, cx, cy, h, w, scale), depth_score

        # 4c. No strong preference — higher score, ties to RGB.
        if rgb_score >= depth_score:
            return _mask_to_bbox_xywh(rgb_mask, cx, cy, h, w, scale), rgb_score
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
        bx, by, bw, bh = _cap_bbox(bx, by, bw, bh, cx, cy, h, w)
        return (bx, by, bw, bh), 0.0


def compute_bboxes_rgbd(
    img_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    points: list[tuple[int, int]],
    predictor,
    geco2_model=None,
    device: str = "cuda",
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """
    Run SAM 2 on the RGB image and separately on the depth colormap,
    then fuse the resulting masks per annotation point.
    Dense swarms are further refined with GECO2 when *geco2_model* is provided.
    depth_bgr must be the colorized depth image at the same (or similar) resolution.
    """
    h, w = img_bgr.shape[:2]
    if not points:
        return [], []

    dh, dw = depth_bgr.shape[:2]
    if (dh, dw) != (h, w):
        depth_bgr = cv2.resize(depth_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    local_scales = _per_point_local_scale(points, h, w)
    global_scale = _fish_scale(points, h, w)
    # A point receives the tight box hint on the RGB run iff its local scale
    # was shrunk below the global fish-scale estimate.  The fusion step uses
    # this flag to know when RGB was "artificially constrained" so it can
    # defer to the un-hinted depth mask for genuinely large objects.
    rgb_hinted = [ls < global_scale for ls in local_scales]

    rgb_bboxes, rgb_scores, rgb_masks = _sam2_predict_per_point(
        img_bgr, points, predictor,
        local_scales=local_scales, use_box_hint=True,
    )
    # Depth runs WITHOUT the hint so large objects whose annotation point
    # happens to sit in a dense cluster still get segmented correctly.
    # Depth boundaries are usually clean enough that over-segmentation of
    # swarms is rare; when it does happen, the fusion step filters it out
    # via the point-centring check.
    depth_bboxes, depth_scores, depth_masks = _sam2_predict_per_point(
        depth_bgr, points, predictor,
        local_scales=local_scales, use_box_hint=False,
    )

    scale = global_scale
    fused_bboxes: list[tuple[int, int, int, int]] = []
    fused_scores: list[float] = []

    for i, (cx, cy) in enumerate(points):
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))

        bbox, conf = _fuse_single_point(
            rgb_masks[i], rgb_scores[i],
            depth_masks[i], depth_scores[i],
            cx_c, cy_c, h, w, min(scale, local_scales[i]),
            rgb_was_hinted=rgb_hinted[i],
        )
        fused_bboxes.append(bbox)
        fused_scores.append(conf)

    # Fix duplicate / oversized bboxes, then refine with GECO2.
    fused_bboxes, fused_scores = _postprocess_bboxes(
        points, fused_bboxes, fused_scores, h, w,
    )
    fused_bboxes, fused_scores = _refine_swarm_with_geco2(
        img_bgr, points, fused_bboxes, fused_scores, geco2_model, device,
    )
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
        _d1 = depth_dir / (img_path.stem + "_depth.jpg")
        _d2 = depth_dir / (img_path.stem + ".jpg")
        depth_path = _d1 if _d1.exists() else _d2
        if depth_path.exists():
            depth_img = cv2.imread(str(depth_path))
        else:
            depth_img = None

        if depth_img is not None:
            bboxes, scores = compute_bboxes_rgbd(
                img, depth_img, points, sam2_predictor,
                geco2_model=geco2_model, device=device,
            )
        else:
            # No depth image found for this frame, fall back to RGB-only SAM 2.
            bboxes, scores = compute_bboxes_sam2(
                img, points, sam2_predictor,
                geco2_model=geco2_model, device=device,
            )

    elif method == "combined" and sam2_predictor is not None and geco2_model is not None:
        bboxes, scores = compute_bboxes_combined(
            img, points, sam2_predictor, geco2_model, device,
        )
    elif method == "sam2" and sam2_predictor is not None:
        bboxes, scores = compute_bboxes_sam2(
            img, points, sam2_predictor,
            geco2_model=geco2_model, device=device,
        )
    else:
        bboxes, scores = compute_bboxes_watershed(img, points)

    # Final fixups (apply to every method):
    #   1) replace bboxes that swallow multiple annotation points,
    #   2) crop off-centre bboxes so the point is roughly centred (this
    #      removes spurious extensions into neighbouring fish),
    #   3) guarantee the annotation point lies inside its bbox.
    h_img, w_img = img.shape[:2]
    bboxes = _fix_multipoint_bboxes(points, bboxes, h_img, w_img)
    bboxes = _recenter_off_center_bboxes(points, bboxes, h_img, w_img)
    bboxes = _enforce_point_inside_bbox(points, bboxes, h_img, w_img)

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


def _run_rank(
    rank: int,
    world_size: int,
    img_files: list,
    cfg: dict,
) -> None:
    """Worker: load models on cuda:{rank} and process img_files[rank::world_size].

    Each spawned process gets a clean Python state, so models are loaded
    independently on each GPU.  Output directories are shared but each
    process writes different files, so no locking is needed.
    """
    # Apply global overrides forwarded from the main process.
    global MAX_BBOX_FRAC, DENSE_MIN_NEIGHBORS
    if cfg["max_bbox_frac"] is not None:
        MAX_BBOX_FRAC = cfg["max_bbox_frac"]
    if cfg["dense_min_neighbors"] is not None:
        DENSE_MIN_NEIGHBORS = cfg["dense_min_neighbors"]

    # Pin this process to its GPU *before* any CUDA operation.
    # Without this, every spawned process defaults to cuda:0 for internal
    # allocations (e.g. model buffers), even when tensors are explicitly
    # moved to cuda:1 — causing illegal memory access errors on rank ≥ 1.
    if cfg["use_cuda"]:
        import torch
        torch.cuda.set_device(rank)

    device = f"cuda:{rank}" if cfg["use_cuda"] else "cpu"
    img_subset = img_files[rank::world_size]

    prefix = f"[rank {rank}/{world_size}]" if world_size > 1 else ""

    sam2_predictor = None
    geco2_model = None

    if cfg["method"] in ("combined", "sam2", "rgbd"):
        geco2_ckpt: str | None = cfg["geco2_ckpt"]
        if geco2_ckpt is not None:
            print(f"{prefix} Loading GECO2 on {device} ...", flush=True)
            geco2_model = build_geco2_model(geco2_ckpt, device=device)
            print(f"{prefix} GECO2 ready.", flush=True)
        print(f"{prefix} Loading SAM 2 on {device} ...", flush=True)
        sam2_predictor = build_sam2_predictor(device=device)
        print(f"{prefix} SAM 2 ready.", flush=True)

        if cfg["method"] == "rgbd" and cfg["depth_dir"] is None:
            print(
                f"{prefix} Warning: no depth folder found. "
                "Falling back to RGB-only SAM 2.",
                flush=True,
            )

    ann_dir: "Path" = cfg["ann_dir"]
    xml_out_dir: "Path" = cfg["xml_out_dir"]
    out_img_dir: "Path" = cfg["out_img_dir"]
    out_depth_dir: "Path" = cfg["out_depth_dir"]
    depth_dir: "Path | None" = cfg["depth_dir"]

    skipped = processed = 0
    for img_path in img_subset:
        ann_xml = ann_dir / (img_path.stem + ".xml")
        if not ann_xml.exists():
            skipped += 1
            continue

        count = generate_bbox_xml(
            img_path, ann_xml,
            method=cfg["method"],
            sam2_predictor=sam2_predictor,
            geco2_model=geco2_model,
            device=device,
            depth_dir=depth_dir,
            xml_out_dir=xml_out_dir,
        )
        if count == 0:
            continue

        _copy_if_exists(img_path, out_img_dir / img_path.name)
        if depth_dir is not None:
            _dn1 = img_path.stem + "_depth.jpg"
            _dn2 = img_path.stem + ".jpg"
            depth_name = _dn1 if (depth_dir / _dn1).exists() else _dn2
            _copy_if_exists(depth_dir / depth_name, out_depth_dir / depth_name)

        processed += 1
        print(f"  {prefix} {img_path.name}: {count} bboxes", flush=True)

    if skipped:
        print(f"  {prefix} Skipped {skipped} image(s) with no annotation file.", flush=True)
    print(f"  {prefix} Done: {processed} image(s).", flush=True)


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
        default=GECO2_DEFAULT_CKPT,
        help=(
            "Path to GECO2 checkpoint (.pth). "
            f"Default: {GECO2_DEFAULT_CKPT}"
        ),
    )
    parser.add_argument(
        "--no_geco2",
        action="store_true",
        default=False,
        help="Disable GECO2 swarm refinement (SAM2-only / RGBD-only).",
    )
    parser.add_argument(
        "--in_dir",
        type=Path,
        default=None,
        help=(
            "Root input folder containing images/, xml/, and color/ sub-folders. "
            "Overrides --img_dir, --ann_dir, and --depth_dir when specified. "
            "Default: point_annotations/"
        ),
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=None,
        help="Input RGB images folder. Default: <in_dir>/images/",
    )
    parser.add_argument(
        "--ann_dir",
        type=Path,
        default=None,
        help="Input point-annotation XMLs folder. Default: <in_dir>/xml/",
    )
    parser.add_argument(
        "--depth_dir",
        type=Path,
        default=None,
        help=(
            "Input depth colormaps folder (<stem>_depth.jpg). "
            "Default: <in_dir>/color/"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Root output folder. Default: auto_annotated_images/. "
            "Sub-folders images/, color/, xml/ are created automatically."
        ),
    )
    parser.add_argument(
        "--vis_out_dir",
        type=Path,
        default=None,
        help="Optional folder for visualization images with drawn bboxes.",
    )
    parser.add_argument(
        "--max_bbox_frac",
        type=float,
        default=None,
        help=(
            "Maximum bbox dimension as a fraction of image size (default: 0.5). "
            "Bboxes exceeding this are replaced by a tiny fallback box."
        ),
    )
    parser.add_argument(
        "--dense_min_neighbors",
        type=int,
        default=None,
        help=(
            "Minimum number of nearby points for a cluster to be considered dense "
            "(default: 3). Lower values make the dense-cluster logic more aggressive."
        ),
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help=(
            "Number of GPUs to use for parallel processing (default: 0 = all available). "
            "Each GPU runs its own SAM 2 / GECO2 instance and processes a disjoint "
            "subset of images.  Set to 1 to disable multi-GPU even when more are present."
        ),
    )
    args = parser.parse_args()

    # Override global tuning constants if the user supplied CLI flags.
    global MAX_BBOX_FRAC, DENSE_MIN_NEIGHBORS
    if args.max_bbox_frac is not None:
        MAX_BBOX_FRAC = args.max_bbox_frac
    if args.dense_min_neighbors is not None:
        DENSE_MIN_NEIGHBORS = args.dense_min_neighbors

    # Resolve the input root when --in_dir is given.
    if args.in_dir is not None:
        in_root = args.in_dir.resolve()
    else:
        in_root = IN_DIR  # default: point_annotations/

    # Resolve input directories.  Explicit --img_dir / --ann_dir win over --in_dir.
    img_dir = args.img_dir.resolve() if args.img_dir else (in_root / "images")
    ann_dir = args.ann_dir.resolve() if args.ann_dir else (in_root / "xml")

    # Resolve depth directory:
    #   explicit --depth_dir > <in_root>/color/ > color/ sibling of img_dir > default.
    if args.depth_dir is not None:
        depth_dir: Path | None = args.depth_dir.resolve()
    elif (in_root / "color").is_dir():
        depth_dir = in_root / "color"
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

    # Resolve GECO2 checkpoint once so all workers use the same path.
    geco2_ckpt: str | None = None
    if args.method in ("combined", "sam2", "rgbd") and not args.no_geco2:
        if args.geco2_checkpoint is not None:
            ckpt = Path(args.geco2_checkpoint)
            if ckpt.is_file():
                geco2_ckpt = str(ckpt)
            else:
                print(
                    f"Warning: GECO2 checkpoint not found at {ckpt}. "
                    "Swarm refinement with GECO2 will be skipped. "
                    "Pass --geco2_checkpoint <path> or --no_geco2."
                )
    elif args.no_geco2:
        print("GECO2 swarm refinement disabled (--no_geco2).")

    n_depth = 0
    if depth_dir is not None:
        n_depth = len(list(depth_dir.glob("*.jpg")))

    # Determine how many GPUs to use.
    import torch
    n_cuda = torch.cuda.device_count() if args.device != "cpu" else 0
    use_cuda = n_cuda > 0 and args.device != "cpu"

    if args.num_gpus > 0:
        world_size = min(args.num_gpus, max(1, n_cuda))
        if world_size < args.num_gpus:
            print(
                f"Warning: requested {args.num_gpus} GPUs but only {n_cuda} "
                f"available — using {world_size}."
            )
    else:
        world_size = max(1, n_cuda)

    # Force single-process when CUDA is unavailable or method needs no GPU.
    if not use_cuda:
        world_size = 1

    print(
        f"Found {len(img_files)} images, {n_depth} depth maps. "
        f"Method: {args.method}. "
        f"GPUs: {world_size if use_cuda else 0} "
        f"({'multi-GPU' if world_size > 1 else 'single-process'})."
    )

    # Ensure output sub-folders exist before workers start writing.
    for d in (xml_out_dir, out_img_dir, out_depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Writing bbox XMLs to {xml_out_dir}")

    cfg = dict(
        method=args.method,
        ann_dir=ann_dir,
        depth_dir=depth_dir,
        xml_out_dir=xml_out_dir,
        out_img_dir=out_img_dir,
        out_depth_dir=out_depth_dir,
        geco2_ckpt=geco2_ckpt,
        use_cuda=use_cuda,
        max_bbox_frac=args.max_bbox_frac,
        dense_min_neighbors=args.dense_min_neighbors,
    )

    if world_size > 1:
        import torch.multiprocessing as mp
        # "spawn" gives each worker a clean Python state so Hydra and CUDA
        # initialise independently without conflicts.
        mp.spawn(
            _run_rank,
            args=(world_size, img_files, cfg),
            nprocs=world_size,
            join=True,
        )
    else:
        _run_rank(0, 1, img_files, cfg)

    print(f"All workers finished. Output in {out_root}")

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
