"""
Draw point annotations from XML files onto every image and save the results
to images/annotated/.  Run from inside the dataset folder or adjust BASE_DIR.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
IMG_DIR    = BASE_DIR / "images"
ANN_DIR    = BASE_DIR / "point_annotations"
OUT_DIR    = IMG_DIR  / "annotated"

# ── visual settings ────────────────────────────────────────────────────────────
DOT_RADIUS  = 4          # px, radius of each annotation dot
DOT_COLOR   = (255, 0, 0, 220)   # red, semi-transparent
OUTLINE     = (255, 255, 255, 180)
FONT_SIZE   = 20
TEXT_COLOR  = (255, 255, 0)      # yellow count label


def parse_points(xml_path: Path) -> list[tuple[int, int]]:
    """Return list of (x, y) tuples from a Pascal-VOC-style point annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    points = []
    for obj in root.findall("object"):
        pt = obj.find("point")
        if pt is not None:
            x = int(pt.findtext("x", default="0"))
            y = int(pt.findtext("y", default="0"))
            points.append((x, y))
    return points


def annotate_image(img_path: Path, xml_path: Path, out_path: Path) -> int:
    """Draw dots on the image and save; returns point count."""
    img    = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw   = ImageDraw.Draw(overlay)

    points = parse_points(xml_path)

    for x, y in points:
        draw.ellipse(
            (x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS),
            fill=DOT_COLOR,
            outline=OUTLINE,
        )

    # Count label (top-left)
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    label = f"count: {len(points)}"
    draw.text((8, 6), label, fill=TEXT_COLOR, font=font)

    composite = Image.alpha_composite(img, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, quality=95)
    return len(points)


def main() -> None:
    img_files = sorted(IMG_DIR.glob("*.jpg"))
    if not img_files:
        print(f"No .jpg files found in {IMG_DIR}")
        return

    print(f"Found {len(img_files)} images. Saving annotated copies to {OUT_DIR} …")
    skipped = 0
    for img_path in img_files:
        xml_path = ANN_DIR / (img_path.stem + ".xml")
        if not xml_path.exists():
            skipped += 1
            continue
        out_path = OUT_DIR / img_path.name
        count = annotate_image(img_path, xml_path, out_path)
        print(f"  {img_path.name}  →  {count} points", flush=True)

    if skipped:
        print(f"\nSkipped {skipped} image(s) with no matching annotation file.")
    print("Done.")


if __name__ == "__main__":
    main()
