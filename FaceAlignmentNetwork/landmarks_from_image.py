import argparse
from pathlib import Path

import face_alignment
import numpy as np
from skimage import io


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 2D face landmarks for all images in a folder.")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="data/datasets",
        help="Directory with input images (default: data/datasets)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/landmarks",
        help="Directory to save .npy landmark files (default: data/landmarks)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    saved_count = 0
    failed_count = 0

    for image_path in image_paths:
        try:
            image = io.imread(str(image_path))
            landmarks = fa.get_landmarks_from_image(image)

            out_path = output_dir / f"{image_path.stem}.npy"
            np.save(out_path, landmarks)
            print(f"Saved landmarks: {out_path}")
            saved_count += 1
        except Exception as exc:
            print(f"Failed for {image_path}: {exc}")
            failed_count += 1

    print(f"Done. Saved: {saved_count}, Failed: {failed_count}")


if __name__ == "__main__":
    main()