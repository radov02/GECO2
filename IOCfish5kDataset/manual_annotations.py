"""
Manual bounding-box annotation tool with zoom, pan, and precise editing.

Shows two images side by side:
  Left  – image from IOCfish5kDataset/images2 with interactive bounding boxes
  Right – corresponding depth-colour image from IOCfish5k-DDataset/color

Controls
--------
  D / Right / Next button    next image
  A / Left  / Prev button    previous image
  Ctrl + Scroll              zoom in/out (both panels)
  Right-click drag           pan when zoomed in
  R                          reset zoom to fit
  S                          save annotations
  Q / Escape / Window X      quit (auto-saves)
"""

import cv2
import re
import numpy as np
import xml.etree.ElementTree as ET
import ctypes
import time
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure physical pixel coordinates on high-DPI displays
ctypes.windll.user32.SetProcessDPIAware()

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
IMG_DIR   = BASE_DIR / "images2"
DEPTH_DIR = BASE_DIR.parent / "IOCfish5k-DDataset" / "color"
XML_DIR   = IMG_DIR / "xml"
DONE_DIR  = BASE_DIR / "done"

# ── visual settings ────────────────────────────────────────────────────────────
BBOX_COLOR     = (0, 255, 0)       # green – untouched boxes
BBOX_SEL_COLOR = (0, 255, 255)     # yellow for selected
BBOX_MOVED_CLR = (255, 255, 255)   # white – already adjusted boxes
BBOX_THICKNESS = 1                 # thin for precision
DOT_COLOR      = (0, 0, 255)       # red centre dot
DOT_RADIUS     = 3                 # fixed display pixels
HANDLE_RADIUS  = 4                 # small square handles
HANDLE_COLOR   = (255, 255, 0)     # cyan handles
GRAB_RADIUS    = 10                # display-px grab threshold
BUTTON_BAR_H   = 52

# ── Arrow key codes (OpenCV waitKeyEx on Windows) ──────────────────────────────
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904

# ── Windows API for modifier key detection ─────────────────────────────────────
VK_CONTROL = 0x11
VK_SHIFT   = 0x10

def _is_ctrl_pressed():
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)

def _is_shift_pressed():
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_SHIFT) & 0x8000)


# ── XML helpers ────────────────────────────────────────────────────────────────

def parse_bbox_xml(xml_path: Path) -> list[dict]:
    """Parse bbox XML → list of dicts with centre, bbox, confidence."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []
    for obj in root.findall("object"):
        pt = obj.find("point")
        if pt is None:
            continue
        cx = int(pt.findtext("x", "0"))
        cy = int(pt.findtext("y", "0"))
        bb = obj.find("bndbox")
        if bb is not None:
            xmin = int(bb.findtext("xmin", "0"))
            ymin = int(bb.findtext("ymin", "0"))
            xmax = int(bb.findtext("xmax", "0"))
            ymax = int(bb.findtext("ymax", "0"))
            bbox = [xmin, ymin, xmax, ymax]
        else:
            bbox = None
        conf_el = obj.find("confidence")
        conf = float(conf_el.text) if conf_el is not None else None
        results.append({
            "center": [cx, cy],
            "bbox": bbox,
            "confidence": conf,
        })
    return results


def save_bbox_xml(
    xml_path: Path,
    img_path: Path,
    img_shape: tuple,
    annotations: list[dict],
) -> None:
    """Write annotations back to Pascal-VOC-style XML."""
    h, w = img_shape[:2]
    d = img_shape[2] if len(img_shape) > 2 else 3

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

    for ann in annotations:
        cx, cy = ann["center"]

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "fish"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        pt = ET.SubElement(obj, "point")
        ET.SubElement(pt, "x").text = str(cx)
        ET.SubElement(pt, "y").text = str(cy)

        if ann["bbox"] is not None:
            xmin, ymin, xmax, ymax = ann["bbox"]
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        if ann["confidence"] is not None:
            ET.SubElement(obj, "confidence").text = f"{ann['confidence']:.4f}"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(xml_path, encoding="unicode", xml_declaration=True)


# ── Done-state helpers ────────────────────────────────────────────────────────

_RE_DONE  = re.compile(r"<done>1</done>")
_RE_XMIN  = re.compile(r"<xmin>(\d+)</xmin>")
_RE_YMIN  = re.compile(r"<ymin>(\d+)</ymin>")
_RE_XMAX  = re.compile(r"<xmax>(\d+)</xmax>")
_RE_YMAX  = re.compile(r"<ymax>(\d+)</ymax>")
_RE_CX    = re.compile(r"<x>(\d+)</x>")
_RE_CY    = re.compile(r"<y>(\d+)</y>")


def _scan_xml(xml_path: Path) -> tuple[int, bool, float, float]:
    """Single text-read: return (bbox_count, is_done, avg_pairwise_iou, center_inside_ratio)."""
    try:
        text = xml_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0, False, 0.0, 1.0

    count = text.count("<object>")
    is_done = bool(_RE_DONE.search(text))

    if is_done:
        return count, is_done, 0.0, 1.0

    xmins = [int(v) for v in _RE_XMIN.findall(text)]
    ymins = [int(v) for v in _RE_YMIN.findall(text)]
    xmaxs = [int(v) for v in _RE_XMAX.findall(text)]
    ymaxs = [int(v) for v in _RE_YMAX.findall(text)]
    cxs = [int(v) for v in _RE_CX.findall(text)]
    cys = [int(v) for v in _RE_CY.findall(text)]
    n = min(len(xmins), len(ymins), len(xmaxs), len(ymaxs))
    nc = min(len(cxs), len(cys), n)

    # Center-inside ratio
    if nc > 0:
        inside = sum(1 for i in range(nc)
                     if xmins[i] <= cxs[i] <= xmaxs[i] and ymins[i] <= cys[i] <= ymaxs[i])
        center_ratio = inside / nc
    else:
        center_ratio = 1.0

    if n < 2:
        return count, is_done, 0.0, center_ratio

    # Average pairwise IoU
    total_iou = 0.0
    num_pairs = 0
    for i in range(n):
        a = (xmins[i], ymins[i], xmaxs[i], ymaxs[i])
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        if a_area <= 0:
            continue
        for j in range(i + 1, n):
            b = (xmins[j], ymins[j], xmaxs[j], ymaxs[j])
            b_area = (b[2] - b[0]) * (b[3] - b[1])
            if b_area <= 0:
                continue
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            iou = inter / (a_area + b_area - inter)
            total_iou += iou
            num_pairs += 1
    avg_iou = total_iou / num_pairs if num_pairs > 0 else 0.0
    return count, is_done, avg_iou, center_ratio


def _scan_xml_fast(xml_path: Path) -> tuple[int, bool]:
    """Thin wrapper kept for callers that only need (count, is_done)."""
    count, is_done, _, _ = _scan_xml(xml_path)
    return count, is_done


def _compute_iou(box1: list[int], box2: list[int]) -> float:
    """IoU between two [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _avg_pairwise_iou(annotations: list[dict]) -> float:
    """Return the average pairwise IoU among all annotated bboxes."""
    bboxes = [a["bbox"] for a in annotations if a["bbox"] is not None]
    if len(bboxes) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            total += _compute_iou(bboxes[i], bboxes[j])
            count += 1
    return total / count if count > 0 else 0.0


def _center_inside_ratio(annotations: list[dict]) -> float:
    """Return the fraction of annotations whose center is inside their bbox."""
    valid = [a for a in annotations if a["bbox"] is not None]
    if not valid:
        return 1.0
    inside = sum(1 for a in valid
                 if a["bbox"][0] <= a["center"][0] <= a["bbox"][2]
                 and a["bbox"][1] <= a["center"][1] <= a["bbox"][3])
    return inside / len(valid)


def _avg_iou_from_xml(xml_path: Path) -> float:
    """Fast regex-based avg pairwise IoU (no XML parser)."""
    _, _, iou, _ = _scan_xml(xml_path)
    return iou


def set_done_in_xml(xml_path: Path, done: bool) -> None:
    """Set or clear the <done> flag inside an existing XML annotation file."""
    if not xml_path or not xml_path.exists():
        return
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        el = root.find("done")
        if done:
            if el is None:
                ET.SubElement(root, "done").text = "1"
            else:
                el.text = "1"
        else:
            if el is not None:
                root.remove(el)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="unicode", xml_declaration=True)
    except Exception:
        pass


# ── Handle constants ───────────────────────────────────────────────────────────
HANDLE_TL = 0;  HANDLE_TR = 1;  HANDLE_BL = 2;  HANDLE_BR = 3
HANDLE_T  = 4;  HANDLE_B  = 5;  HANDLE_L  = 6;  HANDLE_R  = 7
HANDLE_MOVE = 8;  HANDLE_CREATE = 9;  HANDLE_NONE = -1
DEFAULT_BBOX_HALF = 25   # half-size of auto-generated bbox (pixels)


# ── Annotation state ──────────────────────────────────────────────────────────

class AnnotationState:
    """Holds all mutable state for the annotation session."""

    def __init__(self):
        self.annotations: list[dict] = []
        self.img_left = None
        self.img_right = None
        self.img_path: Path | None = None
        self.xml_path: Path | None = None
        self.img_shape = (0, 0, 3)

        # Zoom / pan  (pan = image-space centre of visible viewport)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Window layout
        self.win_w = 1600
        self.win_h = 900
        self.panel_w = 0
        self.panel_h = 0
        self.base_scale = 1.0
        self.gap = 4

        # Bbox interaction
        self.dragging = False
        self.drag_idx = -1
        self.drag_handle = HANDLE_NONE
        self.drag_start = (0, 0)
        self.drag_bbox_start: list[int] | None = None
        self.drag_center_start: list[int] | None = None
        self.selected_idx = -1
        self.dirty = False
        self.moved: set[int] = set()   # indices of bboxes that were adjusted
        self.hidden: set[int] = set()  # indices of bboxes hidden from display
        self.hidden_stack: list[int] = []  # undo stack for hide operations
        self.hover_idx = -1  # index of bbox under cursor
        self.load_time = time.time()  # timestamp when image was loaded
        self.session_elapsed: float = 0.0  # accumulated time across all images

        # Pan interaction
        self.panning = False
        self.pan_start = (0, 0)
        self.pan_origin = (0.0, 0.0)



    # ── derived values ─────────────────────────────────────────────────────

    def eff_scale(self) -> float:
        return self.base_scale * self.zoom

    def viewport_origin(self) -> tuple[float, float]:
        """Top-left of visible region in image coords."""
        es = self.eff_scale()
        if es <= 0:
            return 0.0, 0.0
        return (self.pan_x - self.panel_w / (2.0 * es),
                self.pan_y - self.panel_h / (2.0 * es))

    def img_to_disp(self, ix: float, iy: float) -> tuple[int, int]:
        """Image coords → display coords (left panel)."""
        vl, vt = self.viewport_origin()
        es = self.eff_scale()
        return int((ix - vl) * es), int((iy - vt) * es)

    def disp_to_img(self, dx: int, dy: int) -> tuple[float, float]:
        """Display coords (left panel) → image coords."""
        vl, vt = self.viewport_origin()
        es = self.eff_scale()
        if es <= 0:
            return 0.0, 0.0
        return dx / es + vl, dy / es + vt

    # ── handle positions ───────────────────────────────────────────────────

    def _handles_for(self, idx: int) -> list[tuple[int, int]]:
        """8 handle positions in *image* coordinates."""
        if self.annotations[idx]["bbox"] is None:
            return []
        xmin, ymin, xmax, ymax = self.annotations[idx]["bbox"]
        mx = (xmin + xmax) // 2
        my = (ymin + ymax) // 2
        return [
            (xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax),
            (mx, ymin), (mx, ymax), (xmin, my), (xmax, my),
        ]

    # ── hit testing ────────────────────────────────────────────────────────

    def hit_test(self, dx: int, dy: int, right_panel: bool = False,
                 for_hide: bool = False) -> tuple[int, int]:
        """Return (annotation_idx, handle_type) or (-1, HANDLE_NONE).

        If *right_panel* is True, only moved/selected boxes are interactive
        (unless *for_hide* is True, which allows any visible box).
        """
        if dy >= self.panel_h:
            return -1, HANDLE_NONE

        ix, iy = self.disp_to_img(dx, dy)
        es = self.eff_scale()
        grab = GRAB_RADIUS / es if es > 0 else GRAB_RADIUS

        def _eligible(i: int) -> bool:
            if i in self.hidden:
                return False
            if not right_panel or for_hide:
                return True
            return i in self.moved or i == self.selected_idx

        # Selected box handles first (priority)
        if 0 <= self.selected_idx < len(self.annotations) and _eligible(self.selected_idx):
            for h_idx, (hx, hy) in enumerate(self._handles_for(self.selected_idx)):
                if abs(ix - hx) < grab and abs(iy - hy) < grab:
                    return self.selected_idx, h_idx

        # All eligible handles
        for i in range(len(self.annotations)):
            if not _eligible(i):
                continue
            for h_idx, (hx, hy) in enumerate(self._handles_for(i)):
                if abs(ix - hx) < grab and abs(iy - hy) < grab:
                    return i, h_idx

        # Selected box interior first (priority – it's rendered on top)
        if 0 <= self.selected_idx < len(self.annotations) and _eligible(self.selected_idx):
            s_ann = self.annotations[self.selected_idx]
            if s_ann["bbox"] is not None:
                sxmin, symin, sxmax, symax = s_ann["bbox"]
                if sxmin <= ix <= sxmax and symin <= iy <= symax:
                    return self.selected_idx, HANDLE_MOVE

        # Hovered box interior (priority – it's rendered above normal boxes)
        if (0 <= self.hover_idx < len(self.annotations)
                and self.hover_idx != self.selected_idx
                and _eligible(self.hover_idx)):
            h_ann = self.annotations[self.hover_idx]
            if h_ann["bbox"] is not None:
                hxmin, hymin, hxmax, hymax = h_ann["bbox"]
                if hxmin <= ix <= hxmax and hymin <= iy <= hymax:
                    return self.hover_idx, HANDLE_MOVE

        # Other box interiors
        for i, ann in enumerate(self.annotations):
            if not _eligible(i) or ann["bbox"] is None:
                continue
            xmin, ymin, xmax, ymax = ann["bbox"]
            if xmin <= ix <= xmax and ymin <= iy <= ymax:
                return i, HANDLE_MOVE

        # Center points without bboxes (click to generate)
        for i, ann in enumerate(self.annotations):
            if not _eligible(i) or ann["bbox"] is not None:
                continue
            cx, cy = ann["center"]
            if abs(ix - cx) < grab and abs(iy - cy) < grab:
                return i, HANDLE_CREATE

        return -1, HANDLE_NONE

    # ── persistence ────────────────────────────────────────────────────────

    def save(self) -> None:
        if self.xml_path and self.img_path:
            save_bbox_xml(self.xml_path, self.img_path,
                          self.img_shape, self.annotations)
            self.dirty = False

    # ── layout ─────────────────────────────────────────────────────────────

    def compute_layout(self) -> None:
        self.panel_w = max(1, (self.win_w - self.gap) // 2)
        self.panel_h = max(1, self.win_h - BUTTON_BAR_H)
        ih, iw = self.img_shape[:2]
        if ih > 0 and iw > 0:
            self.base_scale = min(self.panel_w / iw, self.panel_h / ih)
        else:
            self.base_scale = 1.0

    def reset_view(self) -> None:
        ih, iw = self.img_shape[:2]
        self.zoom = 1.0
        self.pan_x = iw / 2.0
        self.pan_y = ih / 2.0


# ── Rendering ──────────────────────────────────────────────────────────────────

def _warp_matrix(state: AnnotationState) -> np.ndarray:
    es = state.eff_scale()
    vl, vt = state.viewport_origin()
    return np.float32([[es, 0, -vl * es],
                       [0, es, -vt * es]])


def render(state: AnnotationState, cur_idx: int, total: int, is_done: bool = False, app: dict | None = None) -> np.ndarray:
    w, h = state.win_w, state.win_h
    if w <= 0 or h <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    M = _warp_matrix(state)
    pw, ph = state.panel_w, state.panel_h

    # ── left panel (annotated image) ───────────────────────────────────────
    if state.img_left is not None:
        left = cv2.warpAffine(state.img_left, M, (pw, ph),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(30, 30, 30))
        canvas[0:ph, 0:pw] = left

    # ── right panel (depth colour) ─────────────────────────────────────────
    rx = pw + state.gap
    rw = w - rx
    if state.img_right is not None and rw > 0:
        right = cv2.warpAffine(state.img_right, M, (rw, ph),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(30, 30, 30))
        canvas[0:ph, rx:rx + rw] = right

    # ── separator ──────────────────────────────────────────────────────────
    sx = pw + state.gap // 2
    cv2.line(canvas, (sx, 0), (sx, ph), (70, 70, 70), 1)

    # ── bounding boxes (selected drawn last to appear on top) ──────────
    def _draw_bbox(i, ann):
        # Point-only annotation: draw just the center dot
        if ann["bbox"] is None:
            cx, cy = ann["center"]
            dc = state.img_to_disp(cx, cy)
            selected = (i == state.selected_idx)
            dot_col = (0, 255, 0) if selected else (255, 0, 255)  # magenta for bbox-less
            if 0 <= dc[0] <= pw and 0 <= dc[1] <= ph:
                cv2.circle(canvas, dc, DOT_RADIUS + 2, dot_col, -1, cv2.LINE_AA)
            rdx = dc[0] + rx
            if rx <= rdx <= w and 0 <= dc[1] <= ph:
                cv2.circle(canvas, (rdx, dc[1]), DOT_RADIUS + 2, dot_col, -1, cv2.LINE_AA)
            return
        xmin, ymin, xmax, ymax = ann["bbox"]
        selected = (i == state.selected_idx)
        hovered = (i == state.hover_idx and not selected)
        moved = i in state.moved
        if selected:
            color = BBOX_SEL_COLOR
            thick = BBOX_THICKNESS + 1
        elif hovered:
            color = BBOX_MOVED_CLR
            thick = BBOX_THICKNESS + 1
        elif moved:
            color = BBOX_MOVED_CLR
            thick = BBOX_THICKNESS
        else:
            color = BBOX_COLOR
            thick = BBOX_THICKNESS

        p1 = state.img_to_disp(xmin, ymin)
        p2 = state.img_to_disp(xmax, ymax)
        bbox_visible = not (p2[0] < 0 or p1[0] > pw or p2[1] < 0 or p1[1] > ph)

        if bbox_visible:
            # Clamp to left panel bounds
            lp1 = (max(0, p1[0]), max(0, p1[1]))
            lp2 = (min(pw, p2[0]), min(ph, p2[1]))
            cv2.rectangle(canvas, lp1, lp2, color, thick, cv2.LINE_AA)

        # Center dot + line to bottom-right corner
        cx, cy = ann["center"]
        dc = state.img_to_disp(cx, cy)
        br = state.img_to_disp(xmax, ymax)
        dot_col = (0, 255, 0) if selected else DOT_COLOR
        # Left panel: line from center to bottom-right
        cv2.line(canvas,
                 (max(0, min(pw, dc[0])), max(0, min(ph, dc[1]))),
                 (max(0, min(pw, br[0])), max(0, min(ph, br[1]))),
                 (0, 255, 0), 1, cv2.LINE_AA)
        if 0 <= dc[0] <= pw and 0 <= dc[1] <= ph:
            cv2.circle(canvas, dc, DOT_RADIUS, dot_col, -1, cv2.LINE_AA)

        # Right panel – only moved / selected bboxes
        if moved or selected:
            rp1 = (max(rx, p1[0] + rx), max(0, p1[1]))
            rp2 = (min(w, p2[0] + rx), min(ph, p2[1]))
            if bbox_visible and rp2[0] > rx and rp1[0] < w:
                cv2.rectangle(canvas, rp1, rp2, color, thick, cv2.LINE_AA)
            rdx = dc[0] + rx
            rbr_x = br[0] + rx
            # Right panel: line from center to bottom-right
            cv2.line(canvas,
                     (max(rx, min(w, rdx)), max(0, min(ph, dc[1]))),
                     (max(rx, min(w, rbr_x)), max(0, min(ph, br[1]))),
                     (0, 255, 0), 1, cv2.LINE_AA)
            if rx <= rdx <= w and 0 <= dc[1] <= ph:
                cv2.circle(canvas, (rdx, dc[1]), DOT_RADIUS, dot_col, -1, cv2.LINE_AA)

        # Square handles for selected box (both panels)
        if selected:
            for hx, hy in state._handles_for(i):
                dh = state.img_to_disp(hx, hy)
                r = HANDLE_RADIUS
                # Left panel handle – clamp to panel
                if -r <= dh[0] <= pw + r and -r <= dh[1] <= ph + r:
                    hl1 = (max(0, dh[0] - r), max(0, dh[1] - r))
                    hl2 = (min(pw, dh[0] + r), min(ph, dh[1] + r))
                    cv2.rectangle(canvas, hl1, hl2, HANDLE_COLOR, -1)
                    cv2.rectangle(canvas, hl1, hl2, (0, 0, 0), 1)
                # Right panel handle – clamp to panel
                rdh_x = dh[0] + rx
                if rx - r <= rdh_x <= w + r and -r <= dh[1] <= ph + r:
                    hr1 = (max(rx, rdh_x - r), max(0, dh[1] - r))
                    hr2 = (min(w, rdh_x + r), min(ph, dh[1] + r))
                    cv2.rectangle(canvas, hr1, hr2, HANDLE_COLOR, -1)
                    cv2.rectangle(canvas, hr1, hr2, (0, 0, 0), 1)

    # Save clean canvas (images only, no bbox overlays) for erasing under hovered/selected
    sel_idx = state.selected_idx
    hover_idx = state.hover_idx
    has_sel = (0 <= sel_idx < len(state.annotations)
               and sel_idx not in state.hidden
               and state.annotations[sel_idx]["bbox"] is not None)
    has_hover = (0 <= hover_idx < len(state.annotations)
                 and hover_idx not in state.hidden
                 and state.annotations[hover_idx]["bbox"] is not None
                 and hover_idx != sel_idx)

    if has_sel or has_hover:
        clean = canvas.copy()

    # First pass: non-selected, non-hovered bboxes
    for i, ann in enumerate(state.annotations):
        if i in state.hidden or i == sel_idx or (has_hover and i == hover_idx):
            continue
        _draw_bbox(i, ann)

    # Erase + draw hovered bbox (above normal, below selected)
    if has_hover:
        h_ann = state.annotations[hover_idx]
        hp1 = state.img_to_disp(h_ann["bbox"][0], h_ann["bbox"][1])
        hp2 = state.img_to_disp(h_ann["bbox"][2], h_ann["bbox"][3])
        y1 = max(0, hp1[1]); y2 = min(ph, hp2[1])
        x1 = max(0, hp1[0]); x2 = min(pw, hp2[0])
        if y2 > y1 and x2 > x1:
            canvas[y1:y2, x1:x2] = clean[y1:y2, x1:x2]
        rx1 = max(rx, hp1[0] + rx); rx2 = min(w, hp2[0] + rx)
        if y2 > y1 and rx2 > rx1:
            canvas[y1:y2, rx1:rx2] = clean[y1:y2, rx1:rx2]
        _draw_bbox(hover_idx, h_ann)

    # Erase + draw selected bbox (topmost)
    if has_sel:
        s_ann = state.annotations[sel_idx]
        sp1 = state.img_to_disp(s_ann["bbox"][0], s_ann["bbox"][1])
        sp2 = state.img_to_disp(s_ann["bbox"][2], s_ann["bbox"][3])
        y1 = max(0, sp1[1]); y2 = min(ph, sp2[1])
        x1 = max(0, sp1[0]); x2 = min(pw, sp2[0])
        if y2 > y1 and x2 > x1:
            canvas[y1:y2, x1:x2] = clean[y1:y2, x1:x2]
        rx1 = max(rx, sp1[0] + rx); rx2 = min(w, sp2[0] + rx)
        if y2 > y1 and rx2 > rx1:
            canvas[y1:y2, rx1:rx2] = clean[y1:y2, rx1:rx2]
        _draw_bbox(sel_idx, s_ann)

    # ── button bar ─────────────────────────────────────────────────────────
    bar_y = ph
    canvas[bar_y:, :] = 45

    # Prev / Next buttons
    bw, bh = 96, 38
    by = bar_y + (BUTTON_BAR_H - bh) // 2
    mid = w // 2
    for label, bx in [("< Prev", mid - bw - 8), ("Next >", mid + 8)]:
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (90, 90, 90), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        tx = bx + (bw - ts[0]) // 2
        ty = by + (bh + ts[1]) // 2
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

    # Done! toggle button
    bw_done = 90
    bx_done = mid + 112
    done_col = (0, 140, 0) if is_done else (90, 90, 90)
    done_label = "Undone" if is_done else "Done!"
    cv2.rectangle(canvas, (bx_done, by), (bx_done + bw_done, by + bh), done_col, -1)
    cv2.rectangle(canvas, (bx_done, by), (bx_done + bw_done, by + bh), (160, 160, 160), 1)
    ts_d = cv2.getTextSize(done_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    tx_d = bx_done + (bw_done - ts_d[0]) // 2
    ty_d = by + (bh + ts_d[1]) // 2
    cv2.putText(canvas, done_label, (tx_d, ty_d),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

    # Jump-to input field
    if app is not None:
        jmp_label = "Go#"
        jmp_lbl_ts = cv2.getTextSize(jmp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        bx_jmp = bx_done + bw_done + 16
        jmp_field_w = 60
        jmp_active = app.get("jump_active", False)
        jmp_text = app.get("jump_text", "")
        # Label
        cv2.putText(canvas, jmp_label, (bx_jmp, by + (bh + jmp_lbl_ts[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        # Input box
        fx = bx_jmp + jmp_lbl_ts[0] + 4
        border_col = (100, 200, 255) if jmp_active else (120, 120, 120)
        bg_col = (60, 60, 60) if jmp_active else (50, 50, 50)
        cv2.rectangle(canvas, (fx, by), (fx + jmp_field_w, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (fx, by), (fx + jmp_field_w, by + bh), border_col, 1)
        # Text inside field
        display_txt = jmp_text if jmp_text else ""
        if jmp_active:
            display_txt += "|"
        txt_ts = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        cv2.putText(canvas, display_txt, (fx + 4, by + (bh + txt_ts[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
        # Store field geometry for mouse click detection
        app["_jmp_field_rect"] = (fx, by, fx + jmp_field_w, by + bh)

    # Status text
    max_iou = _avg_pairwise_iou(state.annotations)
    cir = _center_inside_ratio(state.annotations)
    name = state.img_path.stem if state.img_path else "?"
    modified = " *" if state.dirty else ""
    done_mark = "  [DONE]" if is_done else ""
    zoom_pct = int(state.zoom * 100)
    iou_str = f"  |  AvgIoU: {max_iou:.2f}" if max_iou > 0 else ""
    cir_str = f"  |  CIR: {cir:.0%}" if cir < 1.0 else ""
    info = f"{name}{modified}{done_mark}  |  {cur_idx + 1}/{total}  |  Boxes: {len(state.annotations)}{iou_str}{cir_str}  |  {zoom_pct}%"
    info_col = (80, 210, 80) if is_done else (200, 200, 200)
    cv2.putText(canvas, info, (8, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, info_col, 1, cv2.LINE_AA)

    # Green DONE overlay on the left panel
    if is_done:
        cv2.putText(canvas, "DONE", (pw - 72, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)

    # Avg-IoU overlay (top-right of left panel, coloured by severity)
    overlay_y = 30
    if max_iou > 0 and not is_done:
        if max_iou >= 0.5:
            iou_col = (0, 0, 255)     # red – heavy overlap
        elif max_iou >= 0.2:
            iou_col = (0, 165, 255)   # orange – moderate
        else:
            iou_col = (0, 200, 200)   # yellow – mild
        iou_overlay = f"AvgIoU {max_iou:.2f}"
        ots = cv2.getTextSize(iou_overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(canvas, iou_overlay, (pw - ots[0] - 8, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, iou_col, 2, cv2.LINE_AA)
        overlay_y += 28

    # Center-inside ratio overlay
    if cir < 1.0 and not is_done:
        if cir <= 0.5:
            cir_col = (0, 0, 255)     # red – many outside
        elif cir <= 0.8:
            cir_col = (0, 165, 255)   # orange – some outside
        else:
            cir_col = (0, 200, 200)   # yellow – few outside
        cir_overlay = f"CIR {cir:.0%}"
        cots = cv2.getTextSize(cir_overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(canvas, cir_overlay, (pw - cots[0] - 8, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cir_col, 2, cv2.LINE_AA)

    nhid = len(state.hidden)
    if nhid:
        info += f"  |  Hidden: {nhid}"

    hint = "A/D prev/next  S save  R reset  Ctrl+Scroll zoom  RightDrag pan  Ctrl+RightClick hide  Q quit"
    ts = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(canvas, hint, (w - ts[0] - 8, bar_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    # Global fish-done counter (bottom-left, second row)
    if app is not None:
        gfd = app.get("global_fish_done")
        if gfd is not None:
            gfd_str = f"Fish done: {gfd:,}"
            cv2.putText(canvas, gfd_str, (8, bar_y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 210, 80), 1, cv2.LINE_AA)

    # ── elapsed time overlay (top-left corner) ─────────────────────────────
    elapsed = time.time() - state.load_time
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"IMG  {mins}:{secs:02d}"
    total_elapsed = state.session_elapsed + elapsed
    tmins, tsecs = divmod(int(total_elapsed), 60)
    thours, tmins = divmod(tmins, 60)
    tot_str = (f"TOT  {thours}:{tmins:02d}:{tsecs:02d}" if thours
               else f"TOT  {tmins}:{tsecs:02d}")
    cv2.putText(canvas, time_str, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, tot_str, (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 180, 220), 1, cv2.LINE_AA)

    # ── remaining center-points countdown (top-right of right panel) ──────
    if app is not None:
        rem = app.get("remaining_points")
        if rem is not None:
            rem_str = f"Remaining: {rem:,} pts"
            rem_col = (80, 255, 80) if rem == 0 else (200, 200, 200)
            rts = cv2.getTextSize(rem_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(canvas, rem_str, (w - rts[0] - 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, rem_col, 1, cv2.LINE_AA)

    return canvas


# ── Zoom helper ────────────────────────────────────────────────────────────────

def _handle_zoom(state: AnnotationState, x: int, y: int, delta: int) -> None:
    """Zoom centred on cursor so the point under the cursor stays stationary."""
    if x < state.panel_w:
        dx_p = x
    elif x >= state.panel_w + state.gap:
        dx_p = x - state.panel_w - state.gap
    else:
        return
    dy_p = y
    if dy_p >= state.panel_h:
        return

    old_eff = state.eff_scale()
    factor = 1.15 if delta > 0 else 1.0 / 1.15
    new_zoom = max(0.25, min(40.0, state.zoom * factor))
    new_eff = state.base_scale * new_zoom

    if old_eff > 0 and new_eff > 0:
        state.pan_x += (dx_p - state.panel_w / 2.0) * (1.0 / old_eff - 1.0 / new_eff)
        state.pan_y += (dy_p - state.panel_h / 2.0) * (1.0 / old_eff - 1.0 / new_eff)
    state.zoom = new_zoom


# ── Mouse callback ─────────────────────────────────────────────────────────────

def on_mouse(event: int, x: int, y: int, flags: int, param):
    state, app = param

    # ── Scroll zoom (Ctrl + wheel) ─────────────────────────────────────────
    if event == cv2.EVENT_MOUSEWHEEL:
        if _is_ctrl_pressed():
            delta = 1 if flags > 0 else -1
            _handle_zoom(state, x, y, delta)
        return

    # ── Ctrl+Shift + Middle-click: delete annotation ───────────────────────
    if event == cv2.EVENT_MBUTTONDOWN and _is_ctrl_pressed() and _is_shift_pressed():
        pw = state.panel_w
        rx = pw + state.gap
        if y < state.panel_h:
            if x < pw:
                idx, _ = state.hit_test(x, y, right_panel=False, for_hide=True)
            elif x >= rx:
                idx, _ = state.hit_test(x - rx, y, right_panel=True, for_hide=True)
            else:
                idx = -1
            if idx >= 0:
                state.annotations.pop(idx)
                # Fix up index-based sets after removal
                def _shift(s, removed):
                    return {(j - 1 if j > removed else j) for j in s if j != removed}
                state.moved = _shift(state.moved, idx)
                state.hidden = _shift(state.hidden, idx)
                state.hidden_stack = [j - 1 if j > idx else j for j in state.hidden_stack if j != idx]
                if state.selected_idx == idx:
                    state.selected_idx = -1
                elif state.selected_idx > idx:
                    state.selected_idx -= 1
                if state.hover_idx == idx:
                    state.hover_idx = -1
                elif state.hover_idx > idx:
                    state.hover_idx -= 1
                state.dirty = True
                state.save()
        return

    # ── Button clicks in bottom bar ────────────────────────────────────────
    if event == cv2.EVENT_LBUTTONDOWN and y >= state.panel_h:
        bw, bh = 80, 26
        by = state.panel_h + (BUTTON_BAR_H - bh) // 2
        mid = state.win_w // 2
        # Check jump field click first
        jrect = app.get("_jmp_field_rect")
        if jrect and jrect[0] <= x <= jrect[2] and jrect[1] <= y <= jrect[3]:
            app["jump_active"] = True
            app["jump_text"] = ""  # clear so user can type fresh
            return
        # Click elsewhere in bar deactivates field
        app["jump_active"] = False
        if mid - bw - 8 <= x <= mid - 8 and by <= y <= by + bh:
            app["nav"] = -1
        elif mid + 8 <= x <= mid + 88 and by <= y <= by + bh:
            app["nav"] = 1
        elif mid + 100 <= x <= mid + 172 and by <= y <= by + bh:
            app["toggle_done"] = True
        return

    # ── Ctrl + Right-click: toggle bbox visibility ─────────────────────────
    if event == cv2.EVENT_RBUTTONDOWN and _is_ctrl_pressed():
        pw = state.panel_w
        rx = pw + state.gap
        if y < state.panel_h:
            if x < pw:
                hit_x = x
                idx, _ = state.hit_test(hit_x, y, right_panel=False, for_hide=True)
            elif x >= rx:
                hit_x = x - rx
                idx, _ = state.hit_test(hit_x, y, right_panel=True, for_hide=True)
            else:
                idx = -1
            if idx >= 0:
                state.hidden.add(idx)
                state.hidden_stack.append(idx)
                state.selected_idx = -1
        return

    # ── Right-click pan ────────────────────────────────────────────────────
    if event == cv2.EVENT_RBUTTONDOWN:
        state.panning = True
        state.pan_start = (x, y)
        state.pan_origin = (state.pan_x, state.pan_y)
        return

    if event == cv2.EVENT_MOUSEMOVE and state.panning:
        es = state.eff_scale()
        if es > 0:
            state.pan_x = state.pan_origin[0] - (x - state.pan_start[0]) / es
            state.pan_y = state.pan_origin[1] - (y - state.pan_start[1]) / es
        return

    if event == cv2.EVENT_RBUTTONUP:
        state.panning = False
        return

    # ── Left-click: bbox interaction (both panels) ─────────────────────────
    if event == cv2.EVENT_LBUTTONDOWN:
        pw = state.panel_w
        rx = pw + state.gap
        if y < state.panel_h:
            if x < pw:
                # Left panel – all bboxes
                hit_x = x
                idx, handle = state.hit_test(hit_x, y, right_panel=False)
            elif x >= rx:
                # Right panel – only moved/selected bboxes
                hit_x = x - rx
                idx, handle = state.hit_test(hit_x, y, right_panel=True)
            else:
                idx, handle = -1, HANDLE_NONE

            if idx >= 0:
                ann = state.annotations[idx]
                if handle == HANDLE_CREATE:
                    # Generate default bbox around the center point
                    cx, cy = ann["center"]
                    ih, iw = state.img_shape[:2]
                    h = DEFAULT_BBOX_HALF
                    ann["bbox"] = [
                        max(0, cx - h), max(0, cy - h),
                        min(iw, cx + h), min(ih, cy + h),
                    ]
                    state.selected_idx = idx
                    state.moved.add(idx)
                    state.dirty = True
                    state.save()
                else:
                    state.dragging = True
                    state.drag_idx = idx
                    state.drag_handle = handle
                    state.drag_start = (x, y)
                    state.drag_bbox_start = list(ann["bbox"])
                    state.drag_center_start = list(ann["center"])
                    state.selected_idx = idx
            else:
                state.selected_idx = -1

    elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
        es = state.eff_scale()
        if es <= 0:
            return
        dix = (x - state.drag_start[0]) / es
        diy = (y - state.drag_start[1]) / es

        ann = state.annotations[state.drag_idx]
        o = state.drag_bbox_start
        ih, iw = state.img_shape[:2]

        if state.drag_handle == HANDLE_MOVE:
            bw = o[2] - o[0]
            bh = o[3] - o[1]
            nx = int(max(0, min(iw - bw, o[0] + dix)))
            ny = int(max(0, min(ih - bh, o[1] + diy)))
            ann["bbox"] = [nx, ny, nx + bw, ny + bh]
            # center point stays fixed – only bbox position changes
        else:
            nb = list(o)
            ht = state.drag_handle
            if ht in (HANDLE_TL, HANDLE_L, HANDLE_BL):
                nb[0] = int(max(0, min(o[2] - 5, o[0] + dix)))
            if ht in (HANDLE_TR, HANDLE_R, HANDLE_BR):
                nb[2] = int(max(o[0] + 5, min(iw, o[2] + dix)))
            if ht in (HANDLE_TL, HANDLE_T, HANDLE_TR):
                nb[1] = int(max(0, min(o[3] - 5, o[1] + diy)))
            if ht in (HANDLE_BL, HANDLE_B, HANDLE_BR):
                nb[3] = int(max(o[1] + 5, min(ih, o[3] + diy)))
            ann["bbox"] = nb

        state.dirty = True

    elif event == cv2.EVENT_LBUTTONUP:
        if state.dragging:
            state.moved.add(state.drag_idx)
            state.dragging = False
            if state.dirty:
                state.save()

    # ── Hover detection (passive MOUSEMOVE) ────────────────────────────────────────
    if event == cv2.EVENT_MOUSEMOVE and not state.dragging and not state.panning:
        pw = state.panel_w
        rx = pw + state.gap
        if y < state.panel_h:
            if x < pw:
                idx, _ = state.hit_test(x, y, right_panel=False)
            elif x >= rx:
                idx, _ = state.hit_test(x - rx, y, right_panel=True)
            else:
                idx = -1
            if idx >= 0 and state.annotations[idx]["bbox"] is not None:
                state.hover_idx = idx
            else:
                state.hover_idx = -1
        else:
            state.hover_idx = -1


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(state: AnnotationState, img_id: str,
               img_source_dir: Path | None = None,
               xml_source_dir: Path | None = None) -> None:
    """Load an image pair + its annotations into *state*."""
    if state.dirty:
        state.save()

    # Accumulate time spent on the previous image
    if state.img_path is not None:
        state.session_elapsed += time.time() - state.load_time

    _img_dir = img_source_dir if img_source_dir is not None else (BASE_DIR / "images" / "unannotated")
    _xml_dir = xml_source_dir if xml_source_dir is not None else XML_DIR
    img_path   = _img_dir / f"{img_id}.jpg"
    depth_path = DEPTH_DIR / f"{img_id}.jpg"
    xml_path   = _xml_dir / f"{img_id}.xml"

    state.img_path = img_path
    state.xml_path = xml_path
    state.dirty = False
    state.dragging = False
    state.panning = False
    state.selected_idx = -1
    state.moved = set()
    state.hidden = set()
    state.hidden_stack = []
    state.hover_idx = -1
    state.load_time = time.time()

    # Left image
    if img_path.exists():
        state.img_left = cv2.imread(str(img_path))
    else:
        state.img_left = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(state.img_left, f"Missing: {img_path.name}",
                    (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    state.img_shape = state.img_left.shape

    # Right image (resize to match left so both share the same viewport)
    if depth_path.exists():
        raw = cv2.imread(str(depth_path))
        lh, lw = state.img_shape[:2]
        rh, rw = raw.shape[:2]
        if (rh, rw) != (lh, lw):
            raw = cv2.resize(raw, (lw, lh), interpolation=cv2.INTER_LINEAR)
        state.img_right = raw
    else:
        state.img_right = np.zeros_like(state.img_left)
        cv2.putText(state.img_right, f"Missing: {depth_path.name}",
                    (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Annotations
    state.annotations = parse_bbox_xml(xml_path) if xml_path.exists() else []

    state.compute_layout()
    state.reset_view()


# ── Fix bboxes to centers ──────────────────────────────────────────────────────

def _show_splash(win: str, win_w: int, win_h: int, msg: str, elapsed: float) -> None:
    """Draw a splash screen with message and elapsed timer."""
    mins, secs = divmod(int(elapsed), 60)
    _splash = np.full((win_h, win_w, 3), 30, dtype=np.uint8)
    _ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0]
    cv2.putText(_splash, msg,
                (win_w // 2 - _ts[0] // 2, win_h // 2 + _ts[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2, cv2.LINE_AA)
    time_str = f"{mins}:{secs:02d}"
    tts = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(_splash, time_str,
                (win_w // 2 - tts[0] // 2, win_h // 2 + _ts[1] // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.imshow(win, _splash)
    cv2.waitKey(1)


def _fix_bboxes_to_centers(xml_files: list[Path], win: str, win_w: int, win_h: int) -> int:
    """Fix bboxes: re-center if center outside, shrink if too many centers inside."""
    start = time.time()
    fixed_total = 0
    total = len(xml_files)
    CENTER_THRESHOLD = 10  # shrink bbox if it contains >= this many centers

    for file_idx, xml_path in enumerate(xml_files):
        _show_splash(win, win_w, win_h, f"Fixing... {file_idx + 1}/{total}", time.time() - start)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue

        size_el = root.find("size")
        img_w = int(size_el.findtext("width", "0")) if size_el is not None else 0
        img_h = int(size_el.findtext("height", "0")) if size_el is not None else 0

        # Collect all center points and bbox data in one pass
        objs = root.findall("object")
        centers = []
        bboxes = []
        for obj in objs:
            pt = obj.find("point")
            bb = obj.find("bndbox")
            if pt is not None:
                centers.append((int(pt.findtext("x", "0")), int(pt.findtext("y", "0"))))
            else:
                centers.append(None)
            if bb is not None:
                bboxes.append((int(bb.findtext("xmin", "0")), int(bb.findtext("ymin", "0")),
                               int(bb.findtext("xmax", "0")), int(bb.findtext("ymax", "0"))))
            else:
                bboxes.append(None)

        modified = False
        for i, obj in enumerate(objs):
            if centers[i] is None or bboxes[i] is None:
                continue
            cx, cy = centers[i]
            xmin, ymin, xmax, ymax = bboxes[i]
            bb = obj.find("bndbox")

            need_fix = False
            # Fix 1: center point outside its bbox → shrink to 1/2 size, re-center
            if cx < xmin or cx > xmax or cy < ymin or cy > ymax:
                bw = (xmax - xmin) / 2
                bh = (ymax - ymin) / 2
                half_w = bw / 2
                half_h = bh / 2
                xmin = int(max(0, cx - half_w))
                ymin = int(max(0, cy - half_h))
                xmax = int(min(img_w, cx + half_w)) if img_w > 0 else int(cx + half_w)
                ymax = int(min(img_h, cy + half_h)) if img_h > 0 else int(cy + half_h)
                need_fix = True

            # Fix 2: bbox contains >= CENTER_THRESHOLD center points → shrink to 1/4 size
            if not need_fix:
                count = sum(1 for c in centers
                            if c is not None and xmin <= c[0] <= xmax and ymin <= c[1] <= ymax)
                if count >= CENTER_THRESHOLD:
                    bw = (xmax - xmin) / 4
                    bh = (ymax - ymin) / 4
                    half_w = bw / 2
                    half_h = bh / 2
                    xmin = int(max(0, cx - half_w))
                    ymin = int(max(0, cy - half_h))
                    xmax = int(min(img_w, cx + half_w)) if img_w > 0 else int(cx + half_w)
                    ymax = int(min(img_h, cy + half_h)) if img_h > 0 else int(cy + half_h)
                    need_fix = True

            if need_fix:
                bb.find("xmin").text = str(xmin)
                bb.find("ymin").text = str(ymin)
                bb.find("xmax").text = str(xmax)
                bb.find("ymax").text = str(ymax)
                # Update cached bbox so subsequent checks use new coords
                bboxes[i] = (xmin, ymin, xmax, ymax)
                modified = True
                fixed_total += 1

        if modified:
            ET.indent(tree, space="  ")
            tree.write(xml_path, encoding="unicode", xml_declaration=True)

    elapsed = time.time() - start
    print(f"Fixed {fixed_total} bbox(es) across {total} not-done image(s) in {elapsed:.1f}s")
    return fixed_total


# ── Main ───────────────────────────────────────────────────────────────────────

def main(xml_source_dir: Path | None = None, img_source_dir: Path | None = None,
         fix_bboxes: bool = False, count_points: bool = False) -> None:
    _xml_dir = xml_source_dir if xml_source_dir is not None else XML_DIR
    xml_files_raw = sorted(_xml_dir.glob("*.xml"))
    if not xml_files_raw:
        print(f"No XML files found in {_xml_dir}")
        return

    state = AnnotationState()
    cur = 0
    app = {"nav": 0, "toggle_done": False, "jump_text": "", "jump_active": False,
           "global_fish_done": 0}

    win = "Manual Bbox Annotation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, state.win_w, state.win_h)

    # ── Fix bboxes to centers (if requested) ───────────────────────────────
    if fix_bboxes:
        # Quick parallel check for done status
        with ThreadPoolExecutor() as pool:
            done_flags = list(pool.map(lambda f: _RE_DONE.search(
                f.read_text(encoding="utf-8", errors="ignore")), xml_files_raw))
        notdone_xml = [f for f, d in zip(xml_files_raw, done_flags) if not d]
        if notdone_xml:
            _fix_bboxes_to_centers(notdone_xml, win, state.win_w, state.win_h)

    # ── Sorting with timer splash ──────────────────────────────────────────
    sort_start = time.time()
    _show_splash(win, state.win_w, state.win_h, "Sorting...", 0)

    # Parallel scan: one file read per XML – (count, is_done, avg_iou, center_ratio) in one pass
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_scan_xml, f): i for i, f in enumerate(xml_files_raw)}
        scan_results = [None] * len(xml_files_raw)
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            scan_results[idx] = future.result()
            completed += 1
            if completed % 20 == 0 or completed == len(xml_files_raw):
                _show_splash(win, state.win_w, state.win_h,
                             f"Sorting... {completed}/{len(xml_files_raw)}",
                             time.time() - sort_start)
    done_ids: set[str] = {
        f.stem for f, (_, is_done, _iou, _cir) in zip(xml_files_raw, scan_results) if is_done
    }

    # Separate done vs not-done; metrics already computed in scan
    done_files = []
    notdone_files = []
    for f, (count, is_done, avg_iou, center_ratio) in zip(xml_files_raw, scan_results):
        if is_done:
            done_files.append((f, count))
        else:
            notdone_files.append((f, count, avg_iou, center_ratio))

    # Sort done by bbox count (ascending)
    # Sort not-done by avg IoU descending, then center-inside ratio ascending
    done_files.sort(key=lambda x: x[1])
    notdone_files.sort(key=lambda x: (-x[2], x[3]))

    xml_files = [f for f, _ in done_files] + [f for f, _, _, _ in notdone_files]

    image_ids = [f.stem for f in xml_files]
    total = len(image_ids)
    print(f"Found {total} annotated images ({len(done_files)} done, {len(notdone_files)} not done)")

    # ── global fish-done counter (always, uses scan_results already available) ──
    app["global_fish_done"] = sum(count for _, count in done_files)

    # ── count total center points in not-done images (--countpoints) ───────
    if count_points and notdone_files:
        count_start = time.time()
        _show_splash(win, state.win_w, state.win_h, "Counting points...", 0)
        total_notdone_points = 0
        for i, (_, cnt, _, _) in enumerate(notdone_files):
            total_notdone_points += cnt
            if i % 100 == 0 or i == len(notdone_files) - 1:
                _show_splash(win, state.win_w, state.win_h,
                             f"Counting points... {i + 1}/{len(notdone_files)}",
                             time.time() - count_start)
        app["remaining_points"] = total_notdone_points
        print(f"Total center points in not-done images: {total_notdone_points:,}")

    if notdone_files:
        top = notdone_files[:min(5, len(notdone_files))]
        print("Top not-done by AvgIoU (desc):")
        for f, cnt, iou, cir in top:
            print(f"  {f.stem}: IoU={iou:.3f}  CIR={cir:.0%}  boxes={cnt}")

    # Start on the last Done image (if any exist)
    last_done = next(
        (i for i in range(len(image_ids) - 1, -1, -1) if image_ids[i] in done_ids),
        0,
    )
    cur = last_done
    app["jump_text"] = str(last_done + 1)  # 1-based default

    cv2.setMouseCallback(win, on_mouse, (state, app))

    load_image(state, image_ids[cur], img_source_dir=img_source_dir, xml_source_dir=xml_source_dir)

    while True:
        # ── detect window close (X button) ─────────────────────────────────
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                if state.dirty:
                    state.save()
                break
        except cv2.error:
            break

        # ── track window resize → re-render at correct size ────────────────
        try:
            rect = cv2.getWindowImageRect(win)
            nw, nh = rect[2], rect[3]
            if nw > 0 and nh > 0 and (nw != state.win_w or nh != state.win_h):
                state.win_w = nw
                state.win_h = nh
                state.compute_layout()
        except cv2.error:
            pass

        # ── handle button navigation ───────────────────────────────────────
        if app["nav"] != 0:
            new_cur = cur + app["nav"]
            app["nav"] = 0
            if 0 <= new_cur < total:
                cur = new_cur
                load_image(state, image_ids[cur], img_source_dir=img_source_dir, xml_source_dir=xml_source_dir)

        # ── handle Done toggle ─────────────────────────────────────────────
        if app["toggle_done"]:
            app["toggle_done"] = False
            cur_id = image_ids[cur]
            cur_count = len(state.annotations)
            if cur_id in done_ids:
                done_ids.discard(cur_id)
                set_done_in_xml(state.xml_path, False)
                if "remaining_points" in app:
                    app["remaining_points"] += cur_count
                app["global_fish_done"] -= cur_count
            else:
                done_ids.add(cur_id)
                set_done_in_xml(state.xml_path, True)
                if "remaining_points" in app:
                    app["remaining_points"] -= cur_count
                app["global_fish_done"] += cur_count
            # Update jump_text default to latest last-done position
            ld = next(
                (i for i in range(len(image_ids) - 1, -1, -1) if image_ids[i] in done_ids),
                0,
            )
            if not app["jump_active"]:
                app["jump_text"] = str(ld + 1)

        # ── render & display ───────────────────────────────────────────────
        canvas = render(state, cur, total, image_ids[cur] in done_ids, app=app)
        cv2.imshow(win, canvas)
        key = cv2.waitKeyEx(16)

        # ── jump field keyboard handling ───────────────────────────────────
        if app["jump_active"] and key != -1:
            if key == 27:                             # Escape – deactivate
                app["jump_active"] = False
                ld = next(
                    (i for i in range(total - 1, -1, -1) if image_ids[i] in done_ids),
                    0,
                )
                app["jump_text"] = str(ld + 1)
            elif key == 13:                           # Enter – jump
                try:
                    target = int(app["jump_text"]) - 1  # 1-based → 0-based
                    if 0 <= target < total:
                        cur = target
                        load_image(state, image_ids[cur], img_source_dir=img_source_dir, xml_source_dir=xml_source_dir)
                except ValueError:
                    pass
                app["jump_active"] = False
                ld = next(
                    (i for i in range(total - 1, -1, -1) if image_ids[i] in done_ids),
                    0,
                )
                app["jump_text"] = str(ld + 1)
            elif key == 8:                            # Backspace
                app["jump_text"] = app["jump_text"][:-1]
            elif ord("0") <= key <= ord("9"):
                app["jump_text"] += chr(key)
            continue

        if key in (ord("q"), ord("Q"), 27):           # Q / Escape
            if state.dirty:
                state.save()
            break

        elif key in (ord("d"), ord("D"), KEY_RIGHT):  # next
            if cur < total - 1:
                cur += 1
                load_image(state, image_ids[cur], img_source_dir=img_source_dir, xml_source_dir=xml_source_dir)

        elif key in (ord("a"), ord("A"), KEY_LEFT):   # prev
            if cur > 0:
                cur -= 1
                load_image(state, image_ids[cur], img_source_dir=img_source_dir, xml_source_dir=xml_source_dir)

        elif key in (ord("s"), ord("S")):             # save
            state.save()
            print(f"Saved {state.xml_path.name}")

        elif key == 26:                               # Ctrl+Z – unhide last
            if state.hidden_stack:
                idx = state.hidden_stack.pop()
                state.hidden.discard(idx)

        elif key in (ord("r"), ord("R")):             # reset zoom
            state.reset_view()

    cv2.destroyAllWindows()
    print("Done.")


# ── Done-search CLI mode ──────────────────────────────────────────────────────

def donesearch() -> None:
    """Find all Done-flagged XMLs and copy them + their images to done/."""
    done_dir = BASE_DIR / "done"
    done_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(XML_DIR.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {XML_DIR}")
        return

    found = 0
    for xml_path in xml_files:
        _, is_done = _scan_xml_fast(xml_path)
        if not is_done:
            continue
        img_id = xml_path.stem
        img_path = BASE_DIR / "images" / "unannotated" / f"{img_id}.jpg"

        dst_xml = done_dir / xml_path.name
        shutil.copy2(xml_path, dst_xml)

        if img_path.exists():
            dst_img = done_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            print(f"  {img_id}: image + xml copied")
        else:
            print(f"  {img_id}: xml copied (image not found at {img_path})")
        found += 1

    print(f"\nDone-search complete: {found} annotated image(s) copied to {done_dir}")


if __name__ == "__main__":
    _fix_flag = "--fixbboxestocenters" in sys.argv
    _count_flag = "--countpoints" in sys.argv
    if "--donesearch" in sys.argv:
        donesearch()
    elif "--showdone" in sys.argv:
        main(xml_source_dir=DONE_DIR, img_source_dir=DONE_DIR, fix_bboxes=_fix_flag,
             count_points=_count_flag)
    else:
        main(fix_bboxes=_fix_flag, count_points=_count_flag)
