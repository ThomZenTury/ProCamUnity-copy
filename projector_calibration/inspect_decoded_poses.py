import json
import cv2
import numpy as np

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAPTURE_ROOT = (
    PROJECT_ROOT /
    "calibration_data" /
    "projector"
)

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080


def main():

    folders = sorted(CAPTURE_ROOT.glob("pose_*"))

    print("\nPROJECTOR DECODING QUALITY INSPECTION")
    print("=" * 65)

    results = []

    for folder in folders:

        metadata_path = folder / "metadata.json"
        npz_path = folder / "decoded_projector.npz"

        if not metadata_path.exists() or not npz_path.exists():
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        if metadata.get("accepted") is not True:
            continue

        print(f"\n{folder.name}")
        print("-" * 45)

        with np.load(npz_path) as data:

            x = data["projector_x"]
            y = data["projector_y"]
            valid = data["valid"]

        if not (x.shape == y.shape == valid.shape):
            print("ERROR: Array dimensions differ")
            continue

        valid = (
            valid &
            np.isfinite(x) &
            np.isfinite(y)
        )

        in_bounds = (
            (x >= 0) &
            (x < PROJECTOR_WIDTH) &
            (y >= 0) &
            (y < PROJECTOR_HEIGHT)
        )

        valid &= in_bounds

        percentage = 100 * valid.mean()

        print("Image dimensions:", x.shape)
        print(f"Valid pixels: {percentage:.2f}%")

        if valid.any():

            print(
                "X range:",
                f"{x[valid].min():.0f}",
                "-",
                f"{x[valid].max():.0f}"
            )

            print(
                "Y range:",
                f"{y[valid].min():.0f}",
                "-",
                f"{y[valid].max():.0f}"
            )

            print(
                "Unique X coordinates:",
                len(np.unique(x[valid]))
            )

            print(
                "Unique Y coordinates:",
                len(np.unique(y[valid]))
            )

        corners_start = np.asarray(
            metadata["corners_start"]
        )

        corners_end = np.asarray(
            metadata["corners_end"]
        )

        print(
            "Checkerboard corners:",
            len(corners_start)
        )

        print(
            "Recorded movement:",
            metadata["board_movement_rms_px"],
            "px"
        )

        # Generate visualizations.

        preview = np.zeros(
            (*valid.shape, 3),
            dtype=np.uint8
        )

        preview[:, :, 2] = np.clip(
            x / (PROJECTOR_WIDTH - 1) * 255,
            0, 255
        ).astype(np.uint8)

        preview[:, :, 1] = np.clip(
            y / (PROJECTOR_HEIGHT - 1) * 255,
            0, 255
        ).astype(np.uint8)

        preview[~valid] = 0

        cv2.imwrite(
            str(folder / "decoded_preview.png"),
            preview
        )

        cv2.imwrite(
            str(folder / "decoded_valid.png"),
            valid.astype(np.uint8) * 255
        )

        results.append(percentage)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    if results:

        print("Decoded captures:", len(results))

        print(
            f"Average valid pixels: {np.mean(results):.2f}%"
        )

        print(
            f"Minimum valid pixels: {np.min(results):.2f}%"
        )

        print(
            f"Maximum valid pixels: {np.max(results):.2f}%"
        )

    print("=" * 65)


if __name__ == "__main__":
    main()
