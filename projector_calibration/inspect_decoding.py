import cv2
import numpy as np

from pathlib import Path

folder = Path("data/procam/capture_001")

data = np.load(
    folder / "decoded_projector.npz"
)

px = data["projector_x"]
py = data["projector_y"]
valid = data["valid"]

print("Camera dimensions:", valid.shape)
print("Valid pixels:", valid.sum())

# --------------------------------------------------
# Projector coordinate visualization
# --------------------------------------------------

output = np.zeros(
    (*valid.shape, 3),
    dtype=np.uint8
)

output[:, :, 2] = np.clip(
    px / 1919 * 255,
    0,
    255
).astype(np.uint8)

output[:, :, 1] = np.clip(
    py / 1079 * 255,
    0,
    255
).astype(np.uint8)

output[~valid] = 0

cv2.imshow(
    "Projector Coordinates",
    output
)

# --------------------------------------------------
# Validity mask
# --------------------------------------------------

mask = valid.astype(np.uint8) * 255

cv2.imshow(
    "Valid Pixels",
    mask
)

# --------------------------------------------------
# Camera image
# --------------------------------------------------

board = cv2.imread(
    str(folder / "board.png")
)

if board is not None:

    cv2.imshow(
        "Original Camera Image",
        board
    )

cv2.imwrite(
    str(folder / "inspection_coordinates.png"),
    output
)

cv2.imwrite(
    str(folder / "inspection_valid_mask.png"),
    mask
)

if board is not None:
    cv2.imwrite(
        str(folder / "inspection_camera.png"),
        board
    )

print("Inspection images saved.")


cv2.waitKey(0)
cv2.destroyAllWindows()
