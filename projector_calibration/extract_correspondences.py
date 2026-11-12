import csv
import json

from pathlib import Path

import cv2
import numpy as np

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAPTURE_ROOT = (
    PROJECT_ROOT /
    "calibration_data" /
    "projector"
)

OUTPUT_NPZ = (
    CAPTURE_ROOT /
    "projector_correspondences.npz"
)

OUTPUT_CSV = (
    CAPTURE_ROOT /
    "projector_correspondences.csv"
)

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

BOARD_COLUMNS = 8
BOARD_ROWS = 5
SQUARE_SIZE_MM = 29.0

# Local neighbourhood around each camera corner.
PATCH_RADIUS = 8

# Minimum number of valid decoded pixels.
MIN_SAMPLES = 25

# RANSAC parameters.
RANSAC_THRESHOLD = 2.5
MIN_INLIER_RATIO = 0.50
MAX_RMS = 2.5

# ============================================================
# BOARD OBJECT POINTS
# ============================================================


def create_object_points():

    points = np.zeros(
        (BOARD_COLUMNS * BOARD_ROWS, 3),
        dtype=np.float32
    )

    points[:, :2] = np.mgrid[
        0:BOARD_COLUMNS,
        0:BOARD_ROWS
    ].T.reshape(-1, 2)

    points *= SQUARE_SIZE_MM

    return points


# ============================================================
# LOCAL PROJECTOR COORDINATE ESTIMATION
# ============================================================

def estimate_projector_corner(
    corner,
    projector_x,
    projector_y,
    valid
):

    u, v = corner

    height, width = valid.shape

    cx = int(round(u))
    cy = int(round(v))

    x0 = max(0, cx - PATCH_RADIUS)
    x1 = min(width, cx + PATCH_RADIUS + 1)

    y0 = max(0, cy - PATCH_RADIUS)
    y1 = min(height, cy + PATCH_RADIUS + 1)

    local_valid = valid[y0:y1, x0:x1]

    yy, xx = np.where(local_valid)

    if len(xx) < MIN_SAMPLES:
        return None

    # Camera image coordinates.
    camera_points = np.column_stack([
        xx + x0,
        yy + y0
    ]).astype(np.float32)

    # Corresponding projector coordinates.
    projector_points = np.column_stack([
        projector_x[y0:y1, x0:x1][local_valid],
        projector_y[y0:y1, x0:x1][local_valid]
    ]).astype(np.float32)

    # Robust local affine fitting.
    affine, inlier_mask = cv2.estimateAffine2D(
        camera_points,
        projector_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )

    if affine is None or inlier_mask is None:
        return None

    inliers = inlier_mask.ravel().astype(bool)

    inlier_count = int(inliers.sum())

    inlier_ratio = inlier_count / len(inliers)

    if inlier_count < MIN_SAMPLES:
        return None

    if inlier_ratio < MIN_INLIER_RATIO:
        return None

    # Calculate residuals using RANSAC inliers.
    predicted = (
        camera_points[inliers] @ affine[:, :2].T
        + affine[:, 2]
    )

    residuals = np.linalg.norm(
        predicted - projector_points[inliers],
        axis=1
    )

    rms = float(
        np.sqrt(np.mean(residuals ** 2))
    )

    if rms > MAX_RMS:
        return None

    # Evaluate the affine model at the exact
    # subpixel checkerboard corner.
    corner_h = np.array(
        [u, v, 1.0],
        dtype=np.float64
    )

    projector_corner = (
        affine @ corner_h
    )

    px, py = projector_corner

    if not (
        0 <= px < PROJECTOR_WIDTH
        and
        0 <= py < PROJECTOR_HEIGHT
    ):
        return None

    return {
        "projector": np.array([px, py]),
        "rms": rms,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio
    }


# ============================================================
# PROCESS ONE POSE
# ============================================================

def process_pose(folder, object_points):

    print("\n" + "=" * 65)
    print("PROCESSING:", folder.name)
    print("=" * 65)

    metadata_path = folder / "metadata.json"

    npz_path = folder / "decoded_projector.npz"

    image_path = folder / "rgb_end.png"

    for path in (
        metadata_path,
        npz_path,
        image_path
    ):

        if not path.exists():
            raise RuntimeError(
                f"Missing file: {path.name}"
            )

    with open(metadata_path) as f:
        metadata = json.load(f)

    if metadata.get("accepted") is not True:
        raise RuntimeError("Capture not accepted")

    corners_start = np.asarray(
        metadata["corners_start"],
        dtype=np.float64
    ).reshape(-1, 2)

    corners_end = np.asarray(
        metadata["corners_end"],
        dtype=np.float64
    ).reshape(-1, 2)

    if (
        len(corners_start) != len(object_points)
        or
        len(corners_end) != len(object_points)
    ):
        raise RuntimeError(
            "Unexpected checkerboard corner count"
        )

    # Use the midpoint of start/end observations.
    camera_corners = (
        corners_start + corners_end
    ) / 2.0

    # Load decoded coordinates.
    with np.load(npz_path) as data:

        projector_x = data["projector_x"].copy()
        projector_y = data["projector_y"].copy()
        valid = data["valid"].copy()

    if not (
        projector_x.shape ==
        projector_y.shape ==
        valid.shape
    ):
        raise RuntimeError(
            "Decoded array dimensions differ"
        )

    height, width = valid.shape

    # Exclude non-finite and out-of-range coordinates.
    valid &= (
        np.isfinite(projector_x)
        &
        np.isfinite(projector_y)
        &
        (projector_x >= 0)
        &
        (projector_x < PROJECTOR_WIDTH)
        &
        (projector_y >= 0)
        &
        (projector_y < PROJECTOR_HEIGHT)
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            "Cannot load rgb_end.png"
        )

    if image.shape[:2] != valid.shape:
        raise RuntimeError(
            "Camera image and decoded map dimensions differ"
        )

    if np.any(
        (camera_corners[:, 0] < 0)
        |
        (camera_corners[:, 0] >= width)
        |
        (camera_corners[:, 1] < 0)
        |
        (camera_corners[:, 1] >= height)
    ):
        raise RuntimeError(
            "Checkerboard corners outside camera image"
        )

    projector_corners = np.full(
        (len(camera_corners), 2),
        np.nan,
        dtype=np.float64
    )

    corner_rms = np.full(
        len(camera_corners),
        np.nan,
        dtype=np.float64
    )

    accepted = np.zeros(
        len(camera_corners),
        dtype=bool
    )

    preview = image.copy()

    for index, corner in enumerate(camera_corners):

        result = estimate_projector_corner(
            corner,
            projector_x,
            projector_y,
            valid
        )

        color = (0, 0, 255)

        if result is not None:

            projector_corners[index] = (
                result["projector"]
            )

            corner_rms[index] = result["rms"]

            accepted[index] = True

            color = (0, 255, 0)

        u, v = np.round(corner).astype(int)

        cv2.circle(
            preview,
            (u, v),
            5,
            color,
            -1
        )

        cv2.putText(
            preview,
            str(index),
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    output_image = (
        folder / "correspondence_preview.png"
    )

    cv2.imwrite(
        str(output_image),
        preview
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    count = int(accepted.sum())

    print("Checkerboard corners:", len(camera_corners))

    print("Valid correspondences:", count)

    print(
        "Rejected correspondences:",
        len(camera_corners) - count
    )

    if count:

        print(
            "Mean local RMS:",
            f"{np.mean(corner_rms[accepted]):.3f} px"
        )

        print(
            "Maximum local RMS:",
            f"{np.max(corner_rms[accepted]):.3f} px"
        )

    print(
        "Correspondence preview:",
        output_image
    )

    return {
        "object_points": object_points.copy(),
        "camera_points": camera_corners,
        "projector_points": projector_corners,
        "valid": accepted,
        "rms": corner_rms
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPROJECTOR CORRESPONDENCE EXTRACTION")

    print("=" * 65)

    print("Dataset:", CAPTURE_ROOT)

    folders = sorted(
        folder for folder in CAPTURE_ROOT.glob("pose_*")
        if folder.is_dir()
    )

    object_points = create_object_points()

    all_object = []
    all_camera = []
    all_projector = []
    all_valid = []
    all_rms = []
    pose_names = []

    csv_rows = []

    successful = 0
    failed = 0

    for folder in folders:

        metadata_path = folder / "metadata.json"

        if not metadata_path.exists():
            continue

        try:

            with open(metadata_path) as f:
                metadata = json.load(f)

        except Exception:
            continue

        if metadata.get("accepted") is not True:
            continue

        try:

            result = process_pose(
                folder,
                object_points
            )

            all_object.append(
                result["object_points"]
            )

            all_camera.append(
                result["camera_points"]
            )

            all_projector.append(
                result["projector_points"]
            )

            all_valid.append(
                result["valid"]
            )

            all_rms.append(
                result["rms"]
            )

            pose_names.append(folder.name)

            successful += 1

            for i in range(len(object_points)):

                valid_corner = result["valid"][i]

                projector_point = (
                    result["projector_points"][i]
                )

                csv_rows.append([
                    folder.name,
                    i,
                    *result["object_points"][i],
                    *result["camera_points"][i],
                    *projector_point,
                    result["rms"][i],
                    valid_corner
                ])

        except Exception as error:

            print(
                f"\nFAILED {folder.name}: {error}"
            )

            failed += 1

    # --------------------------------------------------------
    # SAVE AGGREGATED RESULTS
    # --------------------------------------------------------

    if not all_object:

        raise RuntimeError(
            "No accepted poses could be processed"
        )

    np.savez_compressed(
        OUTPUT_NPZ,

        object_points=np.stack(all_object),

        camera_points=np.stack(all_camera),

        projector_points=np.stack(all_projector),

        valid=np.stack(all_valid),

        local_rms=np.stack(all_rms),

        pose_names=np.asarray(pose_names)
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "pose",
            "corner_index",
            "object_x_mm",
            "object_y_mm",
            "object_z_mm",
            "camera_x_px",
            "camera_y_px",
            "projector_x_px",
            "projector_y_px",
            "local_rms_px",
            "valid"
        ])

        writer.writerows(csv_rows)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("CORRESPONDENCE EXTRACTION COMPLETE")
    print("=" * 65)

    print("Processed poses:", successful)
    print("Failed poses:", failed)

    valid_total = sum(
        int(mask.sum())
        for mask in all_valid
    )

    possible_total = (
        len(all_valid) * len(object_points)
    )

    print("Valid correspondences:", valid_total)

    print("Possible correspondences:", possible_total)

    print(
        "Overall success:",
        f"{100 * valid_total / possible_total:.2f}%"
    )

    print("\nSaved files:")

    print(OUTPUT_NPZ)

    print(OUTPUT_CSV)

    print("=" * 65)


if __name__ == "__main__":

    main()
