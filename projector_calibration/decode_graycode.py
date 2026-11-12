import cv2
import numpy as np
import json
import re
import argparse

from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAPTURE_ROOT = (
    PROJECT_ROOT /
    "calibration_data" /
    "projector"
)

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
# IMAGE LOADING
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
# LOAD AND VALIDATE METADATA
# ============================================================

def load_metadata(folder):

    metadata_path = folder / "metadata.json"

    if not metadata_path.exists():

        raise RuntimeError(
            "metadata.json missing"
        )

    with open(metadata_path, "r") as f:

        metadata = json.load(f)

    if metadata.get("accepted") is not True:

        raise RuntimeError(
            "Capture is not accepted"
        )

    pattern_files = metadata.get("pattern_files")

    if not pattern_files or len(pattern_files) != 46:

        raise RuntimeError(
            "Expected exactly 46 pattern filenames"
        )

    missing = [
        name for name in pattern_files
        if not (folder / name).is_file()
    ]

    if missing:

        raise RuntimeError(
            f"Missing pattern images: {missing}"
        )

    return metadata


# ============================================================
# IDENTIFY PATTERN FILENAMES
# ============================================================

def get_pattern_map(metadata):

    patterns = {}

    for filename in metadata["pattern_files"]:

        match = re.match(
            r"^\d+_(horizontal|vertical)_(\d+)(_inverse)?\.png$",
            filename
        )

        if match is None:
            continue

        axis = match.group(1)

        bit = int(match.group(2))

        inverse = match.group(3) is not None

        key = (axis, bit, inverse)

        if key in patterns:

            raise RuntimeError(
                f"Duplicate pattern: {filename}"
            )

        patterns[key] = filename

    return patterns


# ============================================================
# DECODE ONE AXIS
# ============================================================

def decode_axis(
    folder,
    patterns,
    axis,
    number_of_bits,
    threshold
):

    gray = None

    valid = None

    for bit in range(number_of_bits - 1, -1, -1):

        normal_name = patterns.get(
            (axis, bit, False)
        )

        inverse_name = patterns.get(
            (axis, bit, True)
        )

        if normal_name is None or inverse_name is None:

            raise RuntimeError(
                f"Missing {axis} bit {bit} patterns"
            )

        normal = load_image(
            folder / normal_name
        )

        inverse = load_image(
            folder / inverse_name
        )

        if normal.shape != inverse.shape:

            raise RuntimeError(
                f"Dimension mismatch for {axis} bit {bit}"
            )

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

        elif normal.shape != gray.shape:

            raise RuntimeError(
                "Inconsistent camera image dimensions"
            )

        difference = (
            normal.astype(np.int16)
            -
            inverse.astype(np.int16)
        )

        valid &= (
            np.abs(difference) > threshold
        )

        bit_value = (
            difference > 0
        ).astype(np.uint32)

        # Bit position comes directly from filename.
        # Bit 10 = MSB.
        # Bit 0 = LSB.

        gray |= bit_value << bit

    binary = gray_to_binary(gray)

    return binary, valid


# ============================================================
# DECODE PROJECTOR COORDINATES
# ============================================================

def decode_projector(
    folder,
    threshold=DEFAULT_THRESHOLD
):

    folder = Path(folder)

    print("\nDECODING:", folder.name)

    metadata = load_metadata(folder)

    patterns = get_pattern_map(metadata)

    # --------------------------------------------------------
    # VERIFY PATTERN COUNT
    # --------------------------------------------------------

    for axis in ("horizontal", "vertical"):

        for bit in range(11):

            for inverse in (False, True):

                if (axis, bit, inverse) not in patterns:

                    raise RuntimeError(
                        f"Missing pattern: "
                        f"{axis}, bit {bit}, inverse={inverse}"
                    )

    # --------------------------------------------------------
    # LOAD BLACK AND WHITE REFERENCES
    # --------------------------------------------------------

    black_files = [
        name for name in metadata["pattern_files"]
        if re.match(r"^\d+_black\.png$", name)
    ]

    white_files = [
        name for name in metadata["pattern_files"]
        if re.match(r"^\d+_white\.png$", name)
    ]

    if len(black_files) != 1 or len(white_files) != 1:

        raise RuntimeError(
            "Could not identify unique black/white references"
        )

    black = load_image(
        folder / black_files[0]
    )

    white = load_image(
        folder / white_files[0]
    )

    if black.shape != white.shape:

        raise RuntimeError(
            "Black/white image dimensions differ"
        )

    height, width = black.shape

    print(
        f"Camera resolution: {width} x {height}"
    )

    # --------------------------------------------------------
    # REFERENCE VALIDITY
    # --------------------------------------------------------

    reference_contrast = (
        white.astype(np.int16)
        -
        black.astype(np.int16)
    )

    valid_reference = (
        reference_contrast > threshold
    )

    # --------------------------------------------------------
    # HORIZONTAL DECODING
    # --------------------------------------------------------

    print("Decoding horizontal coordinates...")

    x_gray, valid_x = decode_axis(
        folder,
        patterns,
        "horizontal",
        11,
        threshold
    )

    # --------------------------------------------------------
    # VERTICAL DECODING
    # --------------------------------------------------------

    print("Decoding vertical coordinates...")

    y_gray, valid_y = decode_axis(
        folder,
        patterns,
        "vertical",
        11,
        threshold
    )

    # --------------------------------------------------------
    # COMBINE VALIDITY
    # --------------------------------------------------------

    valid = (
        valid_reference
        &
        valid_x
        &
        valid_y
    )

    valid &= (
        x_gray < PROJECTOR_WIDTH
    )

    valid &= (
        y_gray < PROJECTOR_HEIGHT
    )

    px = x_gray.astype(np.float32)

    py = y_gray.astype(np.float32)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    valid_count = int(valid.sum())

    percentage = (
        100.0 * valid_count / valid.size
    )

    print("\nDECODING RESULTS")

    print("Valid pixels:", valid_count)

    print(
        f"Valid percentage: {percentage:.2f}%"
    )

    if valid.any():

        print(
            "Projector X range:",
            px[valid].min(),
            "-",
            px[valid].max()
        )

        print(
            "Projector Y range:",
            py[valid].min(),
            "-",
            py[valid].max()
        )

    else:

        raise RuntimeError(
            "No valid pixels decoded"
        )

    return px, py, valid


# ============================================================
# SAVE RESULTS
# ============================================================

def save_decoding(folder, px, py, valid):

    output_file = (
        folder /
        "decoded_projector.npz"
    )

    np.savez_compressed(
        output_file,
        projector_x=px,
        projector_y=py,
        valid=valid
    )

    print("Saved:", output_file)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_decoding(px, py, valid):

    output = np.zeros(
        (*valid.shape, 3),
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
# AUTOMATIC BATCH PROCESSING
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Decode all accepted projector captures"
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

    parser.add_argument(
        "--skip-existing",
        action="store_true"
    )

    args = parser.parse_args()

    print("\nPROJECTOR GRAY-CODE DECODER")

    print("=" * 60)

    print("Project root:", PROJECT_ROOT)

    print("Dataset:", CAPTURE_ROOT)

    if not CAPTURE_ROOT.exists():

        raise FileNotFoundError(
            CAPTURE_ROOT
        )

    folders = sorted(
        folder for folder in CAPTURE_ROOT.glob("pose_*")
        if folder.is_dir()
        and re.fullmatch(r"pose_\d+", folder.name)
    )

    accepted = []
    rejected = []
    invalid = []

    # --------------------------------------------------------
    # DISCOVER ACCEPTED CAPTURES
    # --------------------------------------------------------

    for folder in folders:

        try:

            metadata = load_metadata(folder)

            accepted.append(folder)

        except RuntimeError as error:

            # Distinguish rejected from invalid captures.
            metadata_path = folder / "metadata.json"

            if metadata_path.exists():

                try:

                    with open(metadata_path) as f:
                        data = json.load(f)

                    if data.get("accepted") is False:

                        rejected.append(folder)
                        continue

                except Exception:
                    pass

            invalid.append(
                (folder.name, str(error))
            )

    print("\nCAPTURE DISCOVERY")

    print("Total:", len(folders))

    print("Accepted:", len(accepted))

    print("Rejected:", len(rejected))

    print("Invalid:", len(invalid))

    # --------------------------------------------------------
    # PROCESS ACCEPTED CAPTURES
    # --------------------------------------------------------

    successful = []

    failed = []

    skipped = []

    for index, folder in enumerate(accepted, 1):

        print("\n" + "=" * 60)

        print(
            f"PROCESSING {index}/{len(accepted)}: "
            f"{folder.name}"
        )

        output_file = (
            folder / "decoded_projector.npz"
        )

        if args.skip_existing and output_file.exists():

            print("Already decoded. Skipping.")

            skipped.append(folder.name)

            continue

        try:

            px, py, valid = decode_projector(
                folder,
                args.threshold
            )

            save_decoding(
                folder,
                px,
                py,
                valid
            )

            successful.append(folder.name)

            if args.visualize:

                visualize_decoding(
                    px,
                    py,
                    valid
                )

        except Exception as error:

            print(
                f"FAILED: {folder.name}"
            )

            print("Reason:", error)

            failed.append(folder.name)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)

    print("BATCH DECODING COMPLETE")

    print("=" * 60)

    print("Capture folders:", len(folders))

    print("Accepted:", len(accepted))

    print("Rejected:", len(rejected))

    print("Invalid:", len(invalid))

    print("Successfully decoded:", len(successful))

    print("Skipped:", len(skipped))

    print("Failed:", len(failed))

    if successful:

        print("\nSUCCESSFUL:")

        for name in successful:
            print(" ", name)

    if failed:

        print("\nFAILED:")

        for name in failed:
            print(" ", name)

    if invalid:

        print("\nINVALID:")

        for name, reason in invalid:
            print(f"  {name}: {reason}")

    print("\nDataset:", CAPTURE_ROOT)

    print("=" * 60)


if __name__ == "__main__":

    main()
