import cv2
import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

DEFAULT_THRESHOLD = 15


# ============================================================
# GRAY CODE CONVERSION
# ============================================================

def gray_to_binary(gray):

    binary = gray.copy()
    shift = 1

    while shift < 32:
        binary ^= binary >> shift
        shift *= 2

    return binary


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: {path}"
        )

    return image


# ============================================================
# FIND PATTERN
# ============================================================

def find_pattern(folder, filename):

    matches = list(folder.glob(filename))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one file matching {filename}, "
            f"found {len(matches)}"
        )

    return matches[0]


# ============================================================
# DECODE ONE AXIS
# ============================================================

def decode_axis(folder, axis, number_of_bits, threshold):

    gray = None
    valid = None

    height = None
    width = None

    for index in range(number_of_bits):

        normal_path = folder / f"{axis}_{index:02d}.png"
        inverse_path = folder / f"{axis}_{index:02d}_inv.png"

        normal = load_image(normal_path)
        inverse = load_image(inverse_path)

        if gray is None:

            height, width = normal.shape

            gray = np.zeros(
                (height, width),
                dtype=np.uint32
            )

            valid = np.ones(
                (height, width),
                dtype=bool
            )

        difference = (
            normal.astype(np.int16)
            - inverse.astype(np.int16)
        )

        valid &= np.abs(difference) > threshold

        bit_value = (
            difference > 0
        ).astype(np.uint32)

        # index 0 = most significant bit
        # index 10 = least significant bit

        bit_position = number_of_bits - 1 - index

        gray |= bit_value << bit_position

    binary = gray_to_binary(gray)

    return binary, valid


# ============================================================
# DECODE PROJECTOR COORDINATES
# ============================================================

def decode_projector(folder, threshold=DEFAULT_THRESHOLD):

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(folder)

    print("\nGRAY-CODE DECODING")
    print("==================")
    print("Folder:", folder)

    # --------------------------------------------------------
    # LOAD BLACK AND WHITE REFERENCES
    # --------------------------------------------------------

    black = load_image(
        folder / "black.png"
    )

    white = load_image(
        folder / "white.png"
    )

    if black.shape != white.shape:
        raise RuntimeError(
            "Black and white reference dimensions differ."
        )

    height, width = black.shape

    print(f"Camera image: {width} x {height}")

    # --------------------------------------------------------
    # REFERENCE CONTRAST
    # --------------------------------------------------------

    reference_contrast = (
        white.astype(np.int16)
        - black.astype(np.int16)
    )

    valid_reference = (
        reference_contrast > threshold
    )

    # --------------------------------------------------------
    # DECODE HORIZONTAL COORDINATES
    # --------------------------------------------------------

    horizontal_bits = (
        PROJECTOR_WIDTH - 1
    ).bit_length()

    print("\nDecoding horizontal patterns...")

    x_gray, valid_x = decode_axis(
        folder,
        "x",
        horizontal_bits,
        threshold
    )

    # --------------------------------------------------------
    # DECODE VERTICAL COORDINATES
    # --------------------------------------------------------

    vertical_bits = (
        PROJECTOR_HEIGHT - 1
    ).bit_length()

    print("Decoding vertical patterns...")

    y_gray, valid_y = decode_axis(
        folder,
        "y",
        vertical_bits,
        threshold
    )

    # --------------------------------------------------------
    # COMBINE VALIDITY MASKS
    # --------------------------------------------------------

    valid = (
        valid_reference
        & valid_x
        & valid_y
    )

    valid &= x_gray < PROJECTOR_WIDTH
    valid &= y_gray < PROJECTOR_HEIGHT

    px = x_gray.astype(np.float32)
    py = y_gray.astype(np.float32)

    print("\nDecoding complete.")

    print("Valid pixels:", int(valid.sum()))

    print(
        "Valid percentage:",
        f"{100.0 * valid.mean():.2f}%"
    )

    return px, py, valid


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_decoding(px, py, valid):

    height, width = valid.shape

    output = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    output[:, :, 2] = np.clip(
        px / (PROJECTOR_WIDTH - 1) * 255,
        0,
        255
    ).astype(np.uint8)

    output[:, :, 1] = np.clip(
        py / (PROJECTOR_HEIGHT - 1) * 255,
        0,
        255
    ).astype(np.uint8)

    output[~valid] = 0

    cv2.imshow(
        "Decoded Projector Coordinates",
        output
    )

    cv2.imshow(
        "Valid Pixel Mask",
        valid.astype(np.uint8) * 255
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_decoding(folder, px, py, valid):

    folder = Path(folder)

    output_file = (
        folder / "decoded_projector.npz"
    )

    np.savez_compressed(
        output_file,
        projector_x=px,
        projector_y=py,
        valid=valid
    )

    print("\nSaved:", output_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "folder",
        help="Captured pose directory"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD
    )

    parser.add_argument(
        "--visualize",
        action="store_true"
    )

    args = parser.parse_args()

    px, py, valid = decode_projector(
        args.folder,
        args.threshold
    )

    if valid.any():

        print(
            "\nProjector X range:",
            px[valid].min(),
            px[valid].max()
        )

        print(
            "Projector Y range:",
            py[valid].min(),
            py[valid].max()
        )

    else:
        print("WARNING: No valid pixels decoded!")

    save_decoding(
        args.folder,
        px,
        py,
        valid
    )

    if args.visualize:

        visualize_decoding(
            px,
            py,
            valid
        )
