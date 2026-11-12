import cv2
import numpy as np
from pathlib import Path

ROOT = Path("data/procam")

BOARD_SIZE = (8, 5)

folders = sorted(ROOT.glob("capture_*"))

print("=" * 60)
print("PROJECTOR CALIBRATION DATASET INSPECTION")
print("=" * 60)

successful = 0

for folder in folders:

    board_path = folder / "board.png"
    npz_path = folder / "decoded_projector.npz"

    print(f"\n{folder.name}")
    print("-" * 40)

    # Inspect checkerboard
    image = cv2.imread(str(board_path))

    if image is None:
        print("Board image: MISSING")
    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        found, corners = (
            cv2.findChessboardCornersSB(
                gray,
                BOARD_SIZE
            )
        )

        if found:
            successful += 1
            print("Checkerboard: FOUND")
            print("Corners:", len(corners))
        else:
            print("Checkerboard: NOT FOUND")

    # Inspect decoded coordinates
    if not npz_path.exists():
        print("NPZ: MISSING")
        continue

    data = np.load(npz_path)

    print("NPZ arrays:")

    for name in data.files:

        arr = data[name]

        print(
            f"  {name}: "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("Total captures:", len(folders))
print("Detected checkerboards:", successful)
