import json
import shutil

from pathlib import Path


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

SOURCE_DIR = Path("calibration_data/projector")
DEST_DIR = Path("data/procam")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def find_pattern(folder, filename):

    matches = list(folder.glob(filename))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename} in {folder}, "
            f"found {len(matches)}"
        )

    return matches[0]


def copy_file(source, destination):

    if not source.exists():
        raise FileNotFoundError(source)

    shutil.copy2(
        source,
        destination
    )


# ------------------------------------------------------------
# CONVERSION
# ------------------------------------------------------------

def main():

    DEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if list(DEST_DIR.glob("capture_*")):

        raise RuntimeError(
            "Destination already contains capture_* folders. "
            "Please inspect them before importing."
        )

    poses = sorted(
        SOURCE_DIR.glob("pose_*")
    )

    accepted = []

    for pose in poses:

        metadata_file = pose / "metadata.json"

        if not metadata_file.exists():
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        if metadata.get("accepted") is True:
            accepted.append(pose)

    print("Accepted source poses:", len(accepted))

    if len(accepted) < 8:
        raise RuntimeError(
            "At least 8 accepted poses are required."
        )

    # --------------------------------------------------------
    # CONVERT EACH ACCEPTED POSE
    # --------------------------------------------------------

    for session_number, pose in enumerate(
        accepted,
        start=1
    ):

        destination = (
            DEST_DIR /
            f"capture_{session_number:03d}"
        )

        destination.mkdir()

        print(
            f"\nConverting {pose.name} "
            f"-> {destination.name}"
        )

        # Use the black-projection image for
        # checkerboard detection.

        copy_file(
            pose / "rgb_end.png",
            destination / "board.png"
        )

        # Reference images.

        copy_file(
            pose / "000_black.png",
            destination / "black.png"
        )

        copy_file(
            pose / "001_white.png",
            destination / "white.png"
        )

        # ----------------------------------------------------
        # X COORDINATE PATTERNS
        # ----------------------------------------------------

        # Our horizontal_10 ... horizontal_0 patterns
        # correspond to repository x_00 ... x_10.

        for index, bit in enumerate(
            range(10, -1, -1)
        ):

            normal = find_pattern(
                pose,
                f"*horizontal_{bit}.png"
            )

            inverse = find_pattern(
                pose,
                f"*horizontal_{bit}_inverse.png"
            )

            copy_file(
                normal,
                destination / f"x_{index:02d}.png"
            )

            copy_file(
                inverse,
                destination / f"x_{index:02d}_inv.png"
            )

        # ----------------------------------------------------
        # Y COORDINATE PATTERNS
        # ----------------------------------------------------

        for index, bit in enumerate(
            range(10, -1, -1)
        ):

            normal = find_pattern(
                pose,
                f"*vertical_{bit}.png"
            )

            inverse = find_pattern(
                pose,
                f"*vertical_{bit}_inverse.png"
            )

            copy_file(
                normal,
                destination / f"y_{index:02d}.png"
            )

            copy_file(
                inverse,
                destination / f"y_{index:02d}_inv.png"
            )

        print("Converted successfully.")

    print("\n================================")
    print("CONVERSION COMPLETE")
    print("================================")

    print("Sessions:", len(accepted))

    print("Destination:", DEST_DIR.resolve())


if __name__ == "__main__":

    main()
