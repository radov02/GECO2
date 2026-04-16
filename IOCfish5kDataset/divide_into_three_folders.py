import argparse
import shutil
from pathlib import Path


def divide_into_subfolders(folder_path: str) -> None:
    root = Path(folder_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Provided path is not a directory: {root}")

    images_dir = root / "images"
    xml_dir = root / "xml"
    color_dir = root / "color"

    for d in (images_dir, xml_dir, color_dir):
        d.mkdir(exist_ok=True)

    moved = {"images": 0, "xml": 0, "color": 0}

    for file in root.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() == ".xml":
            shutil.move(str(file), xml_dir / file.name)
            moved["xml"] += 1
        elif file.suffix.lower() == ".jpg":
            if file.stem.endswith("_depth"):
                shutil.move(str(file), color_dir / file.name)
                moved["color"] += 1
            else:
                shutil.move(str(file), images_dir / file.name)
                moved["images"] += 1

    print(
        f"Done. Moved {moved['images']} image(s) → images/, "
        f"{moved['xml']} XML file(s) → xml/, "
        f"{moved['color']} depth image(s) → color/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organise a flat folder of images, XML annotations and "
                    "colour-depth images into images/, xml/ and color/ subfolders."
    )
    parser.add_argument("folder", help="Path to the folder to organise.")
    args = parser.parse_args()
    divide_into_subfolders(args.folder)
