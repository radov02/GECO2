"""
prepare_done_images.py
----------------------
Collect all "Done" annotated images from a divided sub-folder and copy them
(XML, image, depth/colour) into a new *_done sibling folder.

Usage
-----
    python prepare_done_images.py --folder path/to/divided/2300

The script expects the input folder to have this structure::

    <folder>/
        color/      # depth colour images  (<id>.jpg)
        images/     # raw RGB images        (<id>.jpg)
        xml/        # Pascal-VOC XML files  (<id>.xml)

It creates::

    <DIVIDED_DIR>/<folder_name>_done/
        color/
        images/
        xml/

containing only the files whose XML contains <done>1</done>.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# ── Repo-relative paths ────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent          # IOCfish5kDataset/
DIVIDED_DIR = BASE_DIR / "divided"

# Regex that matches the done flag written by manual_annotations.py
_RE_DONE = re.compile(r"<done>1</done>")


def is_done(xml_path: Path) -> bool:
    """Return True when the XML file contains the <done>1</done> marker."""
    try:
        return bool(_RE_DONE.search(xml_path.read_text(encoding="utf-8")))
    except OSError:
        return False


def prepare_done_images(folder: Path) -> None:
    xml_dir   = folder / "xml"
    img_dir   = folder / "images"
    color_dir = folder / "color"

    for d in (xml_dir, img_dir, color_dir):
        if not d.exists():
            print(f"[ERROR] Expected sub-folder not found: {d}", file=sys.stderr)
            sys.exit(1)

    # ── Output folders: *_done and *_undone ────────────────────────────────
    def _make_out_dirs(suffix: str):
        root  = DIVIDED_DIR / f"{folder.name}_{suffix}"
        dirs  = (root / "xml", root / "images", root / "color")
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return root, *dirs

    done_root,   done_xml,   done_img,   done_color   = _make_out_dirs("done")
    undone_root, undone_xml, undone_img, undone_color = _make_out_dirs("undone")

    # ── Scan XMLs ──────────────────────────────────────────────────────────
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"[WARN] No XML files found in {xml_dir}")
        return

    copied_done   = 0
    copied_undone = 0
    missing = []

    for xml_path in xml_files:
        done = is_done(xml_path)
        img_id = xml_path.stem  # filename without extension

        img_src   = img_dir   / f"{img_id}.jpg"
        color_src = color_dir / f"{img_id}.jpg"

        out_xml_dir   = done_xml   if done else undone_xml
        out_img_dir   = done_img   if done else undone_img
        out_color_dir = done_color if done else undone_color

        # Copy XML
        shutil.copy2(xml_path, out_xml_dir / xml_path.name)

        # Copy image
        if img_src.exists():
            shutil.copy2(img_src, out_img_dir / img_src.name)
        else:
            missing.append(str(img_src))

        # Copy depth/colour image
        if color_src.exists():
            shutil.copy2(color_src, out_color_dir / color_src.name)
        else:
            missing.append(str(color_src))

        if done:
            copied_done += 1
            print(f"  [DONE]   {img_id}")
        else:
            copied_undone += 1
            print(f"  [UNDONE] {img_id}")

    print(f"\nDone.   Copied {copied_done} image(s)   to: {done_root}")
    print(f"Undone. Copied {copied_undone} image(s) to: {undone_root}")

    if missing:
        print(f"\n[WARN] {len(missing)} source file(s) were missing:")
        for m in missing:
            print(f"  - {m}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy Done-annotated images into a <folder>_done output directory."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Path to the input folder containing color/, images/, and xml/ sub-folders.",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {folder}")
    prepare_done_images(folder)


if __name__ == "__main__":
    main()
