import cv2
import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 1920
HEIGHT = 1080

OUTPUT_DIR = Path("patterns")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GRAY CODE
# ============================================================

def binary_to_gray(value):

    return value ^ (value >> 1)


def generate_gray_pattern(
    width,
    height,
    axis,
    bit
):

    if axis == "horizontal":

        coordinates = np.arange(
            width,
            dtype=np.uint32
        )

        gray = binary_to_gray(coordinates)

        values = (
            (gray >> bit) & 1
        ).astype(np.uint8)

        pattern = np.tile(
            values,
            (height, 1)
        )

    elif axis == "vertical":

        coordinates = np.arange(
            height,
            dtype=np.uint32
        )

        gray = binary_to_gray(coordinates)

        values = (
            (gray >> bit) & 1
        ).astype(np.uint8)

        pattern = np.tile(
            values[:, None],
            (1, width)
        )

    else:

        raise ValueError(
            "Invalid axis"
        )

    return pattern * 255


# ============================================================
# SAVE
# ============================================================

def save_pattern(name, image):

    filename = OUTPUT_DIR / f"{name}.png"

    if not cv2.imwrite(
        str(filename),
        image
    ):

        raise RuntimeError(
            f"Could not save {filename}"
        )

    print("Saved:", filename)


# ============================================================
# GENERATE
# ============================================================

def generate_patterns():

    print("\nGRAY CODE GENERATOR")
    print("===================")

    print(
        f"Resolution: {WIDTH} x {HEIGHT}"
    )

    # Black reference.

    black = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint8
    )

    save_pattern(
        "000_black",
        black
    )

    # White reference.

    white = np.full(
        (HEIGHT, WIDTH),
        255,
        dtype=np.uint8
    )

    save_pattern(
        "001_white",
        white
    )

    index = 2

    # Horizontal patterns.

    horizontal_bits = (
        WIDTH - 1
    ).bit_length()

    for bit in reversed(
        range(horizontal_bits)
    ):

        pattern = generate_gray_pattern(
            WIDTH,
            HEIGHT,
            "horizontal",
            bit
        )

        save_pattern(
            f"{index:03d}_horizontal_{bit}",
            pattern
        )

        index += 1

        save_pattern(
            f"{index:03d}_horizontal_{bit}_inverse",
            255 - pattern
        )

        index += 1

    # Vertical patterns.

    vertical_bits = (
        HEIGHT - 1
    ).bit_length()

    for bit in reversed(
        range(vertical_bits)
    ):

        pattern = generate_gray_pattern(
            WIDTH,
            HEIGHT,
            "vertical",
            bit
        )

        save_pattern(
            f"{index:03d}_vertical_{bit}",
            pattern
        )

        index += 1

        save_pattern(
            f"{index:03d}_vertical_{bit}_inverse",
            255 - pattern
        )

        index += 1

    print("\nGeneration complete.")

    print(
        "Total patterns:",
        index
    )


if __name__ == "__main__":

    generate_patterns()
