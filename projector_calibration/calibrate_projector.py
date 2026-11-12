import sys
import json
import csv

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# MUST BE PLACED after if str(PROJECT_ROOT) not in sys.path:
    # sys.path.insert(0, str(PROJECT_ROOT)
# OTHERWISE NO FUNZZE
from calib_common import create_object_points


CAPTURE_ROOT = (
    PROJECT_ROOT /
    "calibration_data" /
    "projector"
)

INPUT_NPZ = (
    CAPTURE_ROOT /
    "projector_correspondences.npz"
)

OUTPUT_NPZ = (
    CAPTURE_ROOT /
    "projector_intrinsics.npz"
)

OUTPUT_JSON = (
    CAPTURE_ROOT /
    "projector_intrinsics.json"
)

OUTPUT_CSV = (
    CAPTURE_ROOT /
    "projector_calibration_report.csv"
)

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

MINIMUM_CORNERS_PER_POSE = 12

# ============================================================
# POSE DIVERSITY
# ============================================================


def calculate_pose_diversity(pose_names):

    normals = []

    for name in pose_names:

        metadata_path = (
            CAPTURE_ROOT /
            str(name) /
            "metadata.json"
        )

        with open(metadata_path) as f:
            metadata = json.load(f)

        rvec = np.asarray(
            metadata["rvec"],
            dtype=np.float64
        ).reshape(3, 1)

        rotation, _ = cv2.Rodrigues(rvec)

        normal = rotation[:, 2]

        normals.append(normal)

    normals = np.asarray(normals)

    angles = []

    for i in range(len(normals)):

        for j in range(i + 1, len(normals)):

            cosine = np.clip(
                np.dot(normals[i], normals[j]),
                -1.0,
                1.0
            )

            angle = np.degrees(
                np.arccos(cosine)
            )

            angles.append(angle)

    angles = np.asarray(angles)

    print("\nPOSE DIVERSITY")

    print("-" * 50)

    print("Number of poses:", len(normals))

    print(
        "Maximum normal angle difference:",
        f"{angles.max():.2f} degrees"
    )

    print(
        "Median normal angle difference:",
        f"{np.median(angles):.2f} degrees"
    )

    if angles.max() < 10:

        print(
            "WARNING: Checkerboard orientations "
            "are very similar."
        )

    elif angles.max() < 20:

        print(
            "WARNING: Consider capturing additional "
            "tilted checkerboard poses."
        )

    else:

        print("Pose orientation diversity looks promising.")

    return angles


# ============================================================
# LOAD CORRESPONDENCES
# ============================================================

def load_correspondences():

    if not INPUT_NPZ.exists():

        raise FileNotFoundError(INPUT_NPZ)

    with np.load(INPUT_NPZ) as data:

        object_points = data["object_points"]
        projector_points = data["projector_points"]
        valid = data["valid"]
        pose_names = data["pose_names"]

    print("\nCORRESPONDENCE DATA")

    print("-" * 50)

    print("Object points:", object_points.shape)

    print("Projector points:", projector_points.shape)

    print("Validity mask:", valid.shape)

    print("Poses:", len(pose_names))

    return (
        object_points,
        projector_points,
        valid,
        pose_names
    )


# ============================================================
# PREPARE CALIBRATION DATA
# ============================================================

def prepare_calibration_data(
    object_points,
    projector_points,
    valid,
    pose_names
):

    calibration_object_points = []

    calibration_image_points = []

    used_pose_names = []

    print("\nPREPARING CALIBRATION")

    print("-" * 50)

    for i, pose_name in enumerate(pose_names):

        mask = valid[i].astype(bool)

        obj = object_points[i][mask]

        img = projector_points[i][mask]

        finite = (
            np.isfinite(img).all(axis=1)
            &
            np.isfinite(obj).all(axis=1)
        )

        obj = obj[finite]

        img = img[finite]

        count = len(obj)

        print(
            f"{str(pose_name):12s}: "
            f"{count:2d} valid corners"
        )

        if count < MINIMUM_CORNERS_PER_POSE:

            print("  Skipping pose: insufficient corners")

            continue

        calibration_object_points.append(
            obj.astype(np.float32).reshape(-1, 1, 3)
        )

        calibration_image_points.append(
            img.astype(np.float32).reshape(-1, 1, 2)
        )

        used_pose_names.append(str(pose_name))

    if len(used_pose_names) < 5:

        raise RuntimeError(
            "Insufficient usable calibration poses."
        )

    print("\nUsable poses:", len(used_pose_names))

    print(
        "Total calibration points:",
        sum(len(p) for p in calibration_object_points)
    )

    return (
        calibration_object_points,
        calibration_image_points,
        used_pose_names
    )


# ============================================================
# CALIBRATE PROJECTOR
# ============================================================

def calibrate_projector(
    object_points,
    image_points
):

    image_size = (
        PROJECTOR_WIDTH,
        PROJECTOR_HEIGHT
    )

    initial_matrix = cv2.initCameraMatrix2D(
        object_points,
        image_points,
        image_size,
        0
    )

    print("\nINITIAL PROJECTOR MATRIX")

    print(initial_matrix)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-10
    )

    (
        rms,
        camera_matrix,
        distortion,
        rvecs,
        tvecs,
        std_intrinsics,
        std_extrinsics,
        per_view_errors
    ) = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        initial_matrix,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=criteria
    )

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "std_intrinsics": std_intrinsics,
        "std_extrinsics": std_extrinsics,
        "per_view_errors": per_view_errors
    }


# ============================================================
# CALCULATE FIELD OF VIEW
# ============================================================

def calculate_fov(camera_matrix):

    fx = camera_matrix[0, 0]

    fy = camera_matrix[1, 1]

    fov_x = np.degrees(
        2 * np.arctan(
            PROJECTOR_WIDTH / (2 * fx)
        )
    )

    fov_y = np.degrees(
        2 * np.arctan(
            PROJECTOR_HEIGHT / (2 * fy)
        )
    )

    return float(fov_x), float(fov_y)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    result,
    pose_names,
    used_pose_names
):

    np.savez_compressed(

        OUTPUT_NPZ,

        camera_matrix=result["camera_matrix"],

        distortion=result["distortion"],

        image_width=PROJECTOR_WIDTH,

        image_height=PROJECTOR_HEIGHT,

        rms=result["rms"],

        per_view_errors=result["per_view_errors"],

        std_intrinsics=result["std_intrinsics"],

        pose_names=np.asarray(used_pose_names),

        rvecs=np.asarray(result["rvecs"]).reshape(-1, 3),

        tvecs=np.asarray(result["tvecs"]).reshape(-1, 3)
    )

    fx = float(result["camera_matrix"][0, 0])
    fy = float(result["camera_matrix"][1, 1])

    cx = float(result["camera_matrix"][0, 2])
    cy = float(result["camera_matrix"][1, 2])

    fov_x, fov_y = calculate_fov(
        result["camera_matrix"]
    )

    output = {

        "device": "projector",

        "resolution": [
            PROJECTOR_WIDTH,
            PROJECTOR_HEIGHT
        ],

        "camera_matrix": (
            result["camera_matrix"].tolist()
        ),

        "distortion_coefficients": (
            result["distortion"].reshape(-1).tolist()
        ),

        "fx": fx,
        "fy": fy,

        "cx": cx,
        "cy": cy,

        "horizontal_fov_degrees": fov_x,

        "vertical_fov_degrees": fov_y,

        "calibration_rms_px": result["rms"],

        "poses_used": used_pose_names,

        "coordinate_system": {

            "x": "right",

            "y": "down",

            "z": "forward"

        },

        "units": {

            "translation": "mm",

            "focal_length": "pixels",

            "image_coordinates": "pixels"

        }
    }

    with open(OUTPUT_JSON, "w") as f:

        json.dump(
            output,
            f,
            indent=4
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "pose",
            "reprojection_rms_px"
        ])

        for name, error in zip(
            used_pose_names,
            result["per_view_errors"].flatten()
        ):

            writer.writerow([
                name,
                float(error)
            ])

    print("\nSaved calibration files:")

    print(OUTPUT_NPZ)

    print(OUTPUT_JSON)

    print(OUTPUT_CSV)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPROJECTOR INTRINSIC CALIBRATION")

    print("=" * 65)

    print("Projector resolution:")
    print(
        f"{PROJECTOR_WIDTH} x {PROJECTOR_HEIGHT}"
    )

    (
        object_points,
        projector_points,
        valid,
        pose_names
    ) = load_correspondences()

    # Check orientation diversity.
    calculate_pose_diversity(
        pose_names
    )

    # Prepare valid observations.
    (
        calibration_object_points,
        calibration_image_points,
        used_pose_names
    ) = prepare_calibration_data(
        object_points,
        projector_points,
        valid,
        pose_names
    )

    # Perform intrinsic calibration.
    result = calibrate_projector(
        calibration_object_points,
        calibration_image_points
    )

    # Calculate field of view.
    fov_x, fov_y = calculate_fov(
        result["camera_matrix"]
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)

    print("PROJECTOR CALIBRATION RESULTS")

    print("=" * 65)

    print("\nCamera matrix:")

    print(result["camera_matrix"])

    print("\nDistortion coefficients:")

    print(result["distortion"].flatten())

    print("\nFocal lengths:")

    print("fx:", result["camera_matrix"][0, 0])

    print("fy:", result["camera_matrix"][1, 1])

    print("\nPrincipal point:")

    print("cx:", result["camera_matrix"][0, 2])

    print("cy:", result["camera_matrix"][1, 2])

    print("\nField of view:")

    print(f"Horizontal: {fov_x:.3f} degrees")

    print(f"Vertical:   {fov_y:.3f} degrees")

    print("\nCalibration reprojection RMS:")

    print(f"{result['rms']:.4f} px")

    print("\nPer-pose reprojection errors:")

    for name, error in zip(
        used_pose_names,
        result["per_view_errors"].flatten()
    ):

        print(
            f"{name:12s}: {error:.4f} px"
        )

    errors = result["per_view_errors"].flatten()

    print("\nError statistics:")

    print(
        "Median:",
        f"{np.median(errors):.4f} px"
    )

    print(
        "Maximum:",
        f"{np.max(errors):.4f} px"
    )

    print("=" * 65)

    # Save all results.
    save_results(
        result,
        pose_names,
        used_pose_names
    )


if __name__ == "__main__":

    main()
