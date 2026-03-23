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
import numpy as np
import xml.etree.ElementTree as ET
import ctypes
import time
from pathlib import Path

# Ensure physical pixel coordinates on high-DPI displays
ctypes.windll.user32.SetProcessDPIAware()

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
IMG_DIR   = BASE_DIR / "images2"
DEPTH_DIR = BASE_DIR.parent / "IOCfish5k-DDataset" / "color"
XML_DIR   = IMG_DIR / "xml"

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
BUTTON_BAR_H   = 36

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

def _scan_xml_fast(xml_path: Path) -> tuple[int, bool]:
    """Return (bbox_count, is_done) by fast text scan — no XML parse overhead."""
    try:
        text = xml_path.read_text(encoding="utf-8", errors="ignore")
        count = text.count("<object>")
        done  = "<done>1</done>" in text
        return count, done
    except OSError:
        return 0, False


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


def render(state: AnnotationState, cur_idx: int, total: int, is_done: bool = False) -> np.ndarray:
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
    bw, bh = 80, 26
    by = bar_y + (BUTTON_BAR_H - bh) // 2
    mid = w // 2
    for label, bx in [("< Prev", mid - bw - 8), ("Next >", mid + 8)]:
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (90, 90, 90), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = bx + (bw - ts[0]) // 2
        ty = by + (bh + ts[1]) // 2
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    # Done! toggle button
    bw_done = 72
    bx_done = mid + 100
    done_col = (0, 140, 0) if is_done else (90, 90, 90)
    done_label = "Undone" if is_done else "Done!"
    cv2.rectangle(canvas, (bx_done, by), (bx_done + bw_done, by + bh), done_col, -1)
    cv2.rectangle(canvas, (bx_done, by), (bx_done + bw_done, by + bh), (160, 160, 160), 1)
    ts_d = cv2.getTextSize(done_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    tx_d = bx_done + (bw_done - ts_d[0]) // 2
    ty_d = by + (bh + ts_d[1]) // 2
    cv2.putText(canvas, done_label, (tx_d, ty_d),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    # Status text
    name = state.img_path.stem if state.img_path else "?"
    modified = " *" if state.dirty else ""
    done_mark = "  [DONE]" if is_done else ""
    zoom_pct = int(state.zoom * 100)
    info = f"{name}{modified}{done_mark}  |  {cur_idx + 1}/{total}  |  Boxes: {len(state.annotations)}  |  {zoom_pct}%"
    info_col = (80, 210, 80) if is_done else (200, 200, 200)
    cv2.putText(canvas, info, (8, bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, info_col, 1, cv2.LINE_AA)

    # Green DONE overlay on the left panel
    if is_done:
        cv2.putText(canvas, "DONE", (pw - 72, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)

    nhid = len(state.hidden)
    if nhid:
        info += f"  |  Hidden: {nhid}"

    hint = "A/D prev/next  S save  R reset  Ctrl+Scroll zoom  RightDrag pan  Ctrl+RightClick hide  Q quit"
    ts = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(canvas, hint, (w - ts[0] - 8, bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    # ── elapsed time overlay (top-left corner) ─────────────────────────────
    elapsed = time.time() - state.load_time
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins}:{secs:02d}"
    cv2.putText(canvas, time_str, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

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

def load_image(state: AnnotationState, img_id: str) -> None:
    """Load an image pair + its annotations into *state*."""
    if state.dirty:
        state.save()

    img_path   = BASE_DIR / "images" / "unannotated" / f"{img_id}.jpg"
    depth_path = DEPTH_DIR / f"{img_id}.jpg"
    xml_path   = XML_DIR  / f"{img_id}.xml"

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    xml_files_raw = sorted(XML_DIR.glob("*.xml"))
    if not xml_files_raw:
        print(f"No XML files found in {XML_DIR}")
        return

    state = AnnotationState()
    cur = 0
    app = {"nav": 0, "toggle_done": False}

    win = "Manual Bbox Annotation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, state.win_w, state.win_h)

    # Show "Sorting..." splash while scanning + sorting
    _splash = np.full((state.win_h, state.win_w, 3), 30, dtype=np.uint8)
    _msg = "Sorting..."
    _ts = cv2.getTextSize(_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0]
    cv2.putText(_splash, _msg,
                (state.win_w // 2 - _ts[0] // 2, state.win_h // 2 + _ts[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.imshow(win, _splash)
    cv2.waitKey(1)

    # Fast scan: bbox count + done flag in one text read per file
    scan_results = [_scan_xml_fast(f) for f in xml_files_raw]
    xml_files = [f for f, _ in sorted(zip(xml_files_raw, scan_results),
                                      key=lambda pair: pair[1][0])]
    done_ids: set[str] = {
        f.stem for f, (_, is_done) in zip(xml_files_raw, scan_results) if is_done
    }

    image_ids = [f.stem for f in xml_files]
    total = len(image_ids)
    print(f"Found {total} annotated images")

    cv2.setMouseCallback(win, on_mouse, (state, app))

    load_image(state, image_ids[cur])

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
                load_image(state, image_ids[cur])

        # ── handle Done toggle ─────────────────────────────────────────────
        if app["toggle_done"]:
            app["toggle_done"] = False
            cur_id = image_ids[cur]
            if cur_id in done_ids:
                done_ids.discard(cur_id)
                set_done_in_xml(state.xml_path, False)
            else:
                done_ids.add(cur_id)
                set_done_in_xml(state.xml_path, True)

        # ── render & display ───────────────────────────────────────────────
        canvas = render(state, cur, total, image_ids[cur] in done_ids)
        cv2.imshow(win, canvas)
        key = cv2.waitKeyEx(16)

        if key in (ord("q"), ord("Q"), 27):           # Q / Escape
            if state.dirty:
                state.save()
            break

        elif key in (ord("d"), ord("D"), KEY_RIGHT):  # next
            if cur < total - 1:
                cur += 1
                load_image(state, image_ids[cur])

        elif key in (ord("a"), ord("A"), KEY_LEFT):   # prev
            if cur > 0:
                cur -= 1
                load_image(state, image_ids[cur])

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


if __name__ == "__main__":
    main()
