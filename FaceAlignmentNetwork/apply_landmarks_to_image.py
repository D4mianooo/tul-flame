import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_landmarks_overlay(
        image_path: str | Path,
        landmarks_2d: np.ndarray,
        output_path: str | Path,
        point_radius: int = 2,
        point_color: tuple[int, int, int] = (255, 0, 0),
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)

    if landmarks_2d.ndim != 2 or landmarks_2d.shape[1] != 2:
        raise ValueError(f"Expected landmarks shape (N, 2), got {landmarks_2d.shape}")

    img = Image.open(image_path).convert("RGB")
    out_img = img.copy()
    draw = ImageDraw.Draw(out_img)

    for x, y in landmarks_2d:
        x, y = float(x), float(y)
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=point_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)
    return output_path


def extract_landmark_set(loaded: np.ndarray) -> np.ndarray:
    """
    Normalize loaded .npy landmarks into shape (N, 2).
    Supports common outputs:
    - (N, 2)
    - (F, N, 2) -> first face
    - object arrays from np.save(list_of_arrays) -> first item
    """
    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        if len(loaded) == 0:
            raise ValueError("Empty landmarks object array.")
        first = loaded[0]
        arr = np.asarray(first)
    else:
        arr = np.asarray(loaded)

    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[0]  # first face
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Unsupported landmarks shape: {arr.shape}")

    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply landmarks from .npy files to matching images.")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="data/datasets",
        help="Directory with original images (default: data/datasets)",
    )
    parser.add_argument(
        "-l",
        "--landmarks-dir",
        type=str,
        default="data/landmarks",
        help="Directory with .npy landmarks (default: data/landmarks)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/output",
        help="Directory for output images (default: data/output)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.input_dir)
    landmarks_dir = Path(args.landmarks_dir)
    output_dir = Path(args.output_dir)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    output_dir.mkdir(parents=True, exist_ok=True)

    landmark_files = sorted(landmarks_dir.rglob("*.npy"))

    if not landmark_files:
        print(f"No .npy files found in {landmarks_dir.resolve()}")

    for lmk_file in landmark_files:
        stem = lmk_file.stem

        image_path = None
        for ext in image_extensions:
            candidate = dataset_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"Skipped {lmk_file.name}: original image not found in {dataset_dir}")
            continue

        try:
            loaded = np.load(lmk_file, allow_pickle=True)
            landmarks_2d = extract_landmark_set(loaded)

            out_path = output_dir / f"{image_path.stem}_l{image_path.suffix}"
            saved = save_landmarks_overlay(
                image_path=image_path,
                landmarks_2d=landmarks_2d,
                output_path=out_path,
                point_radius=3,
            )
            print(f"Saved: {saved}")
        except Exception as exc:
            print(f"Failed for {lmk_file.name}: {exc}")