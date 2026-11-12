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

INPUT_NPZ = (
    CAPTURE_ROOT /
    "projector_correspondences.npz"
)

OUTPUT_JSON = (
    CAPTURE_ROOT /
    "projector_intrinsics_diagnostics.json"
)

OUTPUT_CSV = (
    CAPTURE_ROOT /
    "projector_intrinsics_diagnostics.csv"
)

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

IMAGE_SIZE = (
    PROJECTOR_WIDTH,
    PROJECTOR_HEIGHT
)

MINIMUM_CORNERS_PER_POSE = 12


# ============================================================
# DISTORTION MODELS
# ============================================================

MODELS = {

    "A_full": {
        "description": "k1, k2, p1, p2, k3",
        "flags": 0
    },

    "B_no_k3": {
        "description": "k1, k2, p1, p2",
        "flags": cv2.CALIB_FIX_K3
    },

    "C_radial_only": {
        "description": "k1, k2, k3",
        "flags": cv2.CALIB_ZERO_TANGENT_DIST
    },

    "D_simple": {
        "description": "k1 only",
        "flags": (
            cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
        )
    }

}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_NPZ.exists():
        raise FileNotFoundError(INPUT_NPZ)

    with np.load(INPUT_NPZ) as data:

        object_points = data["object_points"].copy()

        projector_points = data["projector_points"].copy()

        valid = data["valid"].copy()

        pose_names = data["pose_names"].copy()

    if not (
        object_points.shape[:2]
        ==
        projector_points.shape[:2]
        ==
        valid.shape
    ):
        raise RuntimeError(
            "Correspondence array dimensions differ"
        )

    return (
        object_points,
        projector_points,
        valid,
        pose_names
    )


# ============================================================
# PREPARE ONE POSE
# ============================================================

def prepare_pose(
    object_points,
    projector_points,
    valid,
    index
):

    mask = valid[index].astype(bool)

    obj = object_points[index][mask]

    img = projector_points[index][mask]

    finite = (
        np.isfinite(obj).all(axis=1)
        &
        np.isfinite(img).all(axis=1)
    )

    obj = obj[finite]

    img = img[finite]

    if len(obj) < MINIMUM_CORNERS_PER_POSE:
        return None

    return (
        obj.astype(np.float32).reshape(-1, 1, 3),
        img.astype(np.float32).reshape(-1, 1, 2)
    )


# ============================================================
# CALIBRATION
# ============================================================

def calibrate(
    object_points,
    image_points,
    flags
):

    initial_matrix = cv2.initCameraMatrix2D(
        object_points,
        image_points,
        IMAGE_SIZE,
        0
    )

    initial_distortion = np.zeros(
        (5, 1),
        dtype=np.float64
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_MAX_ITER,
        300,
        1e-10
    )

    (
        rms,
        camera_matrix,
        distortion,
        rvecs,
        tvecs
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        IMAGE_SIZE,
        initial_matrix,
        initial_distortion,
        flags=(
            flags
            |
            cv2.CALIB_USE_INTRINSIC_GUESS
        ),
        criteria=criteria
    )

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "rvecs": rvecs,
        "tvecs": tvecs
    }


# ============================================================
# CALCULATE REPROJECTION ERRORS
# ============================================================

def calculate_reprojection_error(
    obj,
    img,
    camera_matrix,
    distortion,
    rvec,
    tvec
):

    projected, _ = cv2.projectPoints(
        obj,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    projected = projected.reshape(-1, 2)

    observed = img.reshape(-1, 2)

    residuals = np.linalg.norm(
        projected - observed,
        axis=1
    )

    rms = float(
        np.sqrt(np.mean(residuals ** 2))
    )

    return rms


# ============================================================
# LEAVE-ONE-POSE-OUT VALIDATION
# ============================================================

def validate_leave_one_out(
    poses,
    flags,
    model_name
):

    fold_results = []

    print(
        f"\nLeave-one-pose-out validation: {model_name}"
    )

    print("-" * 60)

    for held_out_index in range(len(poses)):

        training_poses = [
            pose for i, pose in enumerate(poses)
            if i != held_out_index
        ]

        training_object = [
            pose[0] for pose in training_poses
        ]

        training_image = [
            pose[1] for pose in training_poses
        ]

        held_object, held_image, held_name = poses[
            held_out_index
        ]

        try:

            result = calibrate(
                training_object,
                training_image,
                flags
            )

            # Estimate the held-out checkerboard pose
            # using only its projector observations.
            success, rvec, tvec = cv2.solvePnP(
                held_object,
                held_image,
                result["camera_matrix"],
                result["distortion"],
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                raise RuntimeError(
                    "solvePnP failed"
                )

            error = calculate_reprojection_error(
                held_object,
                held_image,
                result["camera_matrix"],
                result["distortion"],
                rvec,
                tvec
            )

            fold_results.append({
                "pose": held_name,
                "rms": error,
                "success": True
            })

            print(
                f"{held_name:12s}: {error:.4f} px"
            )

        except Exception as error:

            print(
                f"{held_name:12s}: FAILED - {error}"
            )

            fold_results.append({
                "pose": held_name,
                "rms": None,
                "success": False
            })

    successful = [
        item["rms"]
        for item in fold_results
        if item["success"]
    ]

    if not successful:

        raise RuntimeError(
            f"All validation folds failed for {model_name}"
        )

    successful = np.asarray(successful)

    return {
        "folds": fold_results,
        "mean": float(np.mean(successful)),
        "median": float(np.median(successful)),
        "maximum": float(np.max(successful)),
        "aggregate": float(
            np.sqrt(np.mean(successful ** 2))
        ),
        "successful_folds": len(successful)
    }


# ============================================================
# FULL DATASET PER-POSE ERRORS
# ============================================================

def calculate_per_pose_errors(
    result,
    poses
):

    errors = []

    for i, pose in enumerate(poses):

        obj, img, name = pose

        error = calculate_reprojection_error(
            obj,
            img,
            result["camera_matrix"],
            result["distortion"],
            result["rvecs"][i],
            result["tvecs"][i]
        )

        errors.append({
            "pose": name,
            "rms": error
        })

    return errors


# ============================================================
# FIELD OF VIEW
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
# DISTORTION DIAGNOSTICS
# ============================================================

def calculate_radial_range(
    camera_matrix,
    distortion
):

    fx = camera_matrix[0, 0]

    fy = camera_matrix[1, 1]

    cx = camera_matrix[0, 2]

    cy = camera_matrix[1, 2]

    k1, k2, p1, p2, k3 = distortion.flatten()[:5]

    x_pixels = np.linspace(
        0,
        PROJECTOR_WIDTH - 1,
        100
    )

    y_pixels = np.linspace(
        0,
        PROJECTOR_HEIGHT - 1,
        100
    )

    xx, yy = np.meshgrid(
        x_pixels,
        y_pixels
    )

    x = (xx - cx) / fx

    y = (yy - cy) / fy

    r2 = x * x + y * y

    radial = (
        1
        +
        k1 * r2
        +
        k2 * r2 ** 2
        +
        k3 * r2 ** 3
    )

    return {
        "minimum": float(radial.min()),
        "maximum": float(radial.max())
    }


# ============================================================
# ANALYSE ONE MODEL
# ============================================================

def analyse_model(
    model_name,
    model_config,
    poses
):

    print("\n")
    print("=" * 65)

    print("MODEL:", model_name)

    print(model_config["description"])

    print("=" * 65)

    object_points = [
        pose[0] for pose in poses
    ]

    image_points = [
        pose[1] for pose in poses
    ]

    # --------------------------------------------------------
    # CALIBRATION USING ALL POSES
    # --------------------------------------------------------

    print("\nCalibrating using all poses...")

    result = calibrate(
        object_points,
        image_points,
        model_config["flags"]
    )

    per_pose_errors = calculate_per_pose_errors(
        result,
        poses
    )

    fov_x, fov_y = calculate_fov(
        result["camera_matrix"]
    )

    radial_range = calculate_radial_range(
        result["camera_matrix"],
        result["distortion"]
    )

    print("\nCamera matrix:")

    print(result["camera_matrix"])

    print("\nDistortion coefficients:")

    print(result["distortion"].flatten())

    print("\nCalibration RMS:")

    print(f"{result['rms']:.4f} px")

    print("\nField of view:")

    print(f"Horizontal: {fov_x:.3f} degrees")

    print(f"Vertical:   {fov_y:.3f} degrees")

    print("\nRadial distortion factor range:")

    print(radial_range)

    print("\nPer-pose calibration errors:")

    for item in per_pose_errors:

        print(
            f"{item['pose']:12s}: "
            f"{item['rms']:.4f} px"
        )

    # --------------------------------------------------------
    # LEAVE-ONE-POSE-OUT VALIDATION
    # --------------------------------------------------------

    validation = validate_leave_one_out(
        poses,
        model_config["flags"],
        model_name
    )

    print("\nVALIDATION SUMMARY")

    print(
        "Mean:",
        f"{validation['mean']:.4f} px"
    )

    print(
        "Median:",
        f"{validation['median']:.4f} px"
    )

    print(
        "Maximum:",
        f"{validation['maximum']:.4f} px"
    )

    print(
        "Aggregate RMS:",
        f"{validation['aggregate']:.4f} px"
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "description": model_config["description"],

        "calibration_rms": result["rms"],

        "camera_matrix": (
            result["camera_matrix"].tolist()
        ),

        "distortion": (
            result["distortion"].flatten().tolist()
        ),

        "fx": float(result["camera_matrix"][0, 0]),

        "fy": float(result["camera_matrix"][1, 1]),

        "cx": float(result["camera_matrix"][0, 2]),

        "cy": float(result["camera_matrix"][1, 2]),

        "horizontal_fov": fov_x,

        "vertical_fov": fov_y,

        "radial_factor_range": radial_range,

        "per_pose_errors": per_pose_errors,

        "validation": validation

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPROJECTOR INTRINSICS DIAGNOSTICS")

    print("=" * 65)

    print("Dataset:", CAPTURE_ROOT)

    (
        object_points,
        projector_points,
        valid,
        pose_names
    ) = load_data()

    poses = []

    print("\nAVAILABLE POSES")

    print("-" * 50)

    for i, pose_name in enumerate(pose_names):

        prepared = prepare_pose(
            object_points,
            projector_points,
            valid,
            i
        )

        if prepared is None:

            print(
                f"{str(pose_name):12s}: skipped"
            )

            continue

        obj, img = prepared

        poses.append((
            obj,
            img,
            str(pose_name)
        ))

        print(
            f"{str(pose_name):12s}: "
            f"{len(obj)} corners"
        )

    if len(poses) < 5:

        raise RuntimeError(
            "Not enough calibration poses"
        )

    print("\nUsable poses:", len(poses))

    # --------------------------------------------------------
    # ANALYSE ALL MODELS
    # --------------------------------------------------------

    results = {}

    for model_name, model_config in MODELS.items():

        results[model_name] = analyse_model(
            model_name,
            model_config,
            poses
        )

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    print("\n")
    print("=" * 85)

    print("DISTORTION MODEL COMPARISON")

    print("=" * 85)

    print(
        f"{'Model':18s}"
        f"{'Calibration':>15s}"
        f"{'Validation':>15s}"
        f"{'Median':>15s}"
        f"{'Maximum':>15s}"
    )

    print("-" * 85)

    for name, result in results.items():

        validation = result["validation"]

        print(
            f"{name:18s}"
            f"{result['calibration_rms']:15.4f}"
            f"{validation['aggregate']:15.4f}"
            f"{validation['median']:15.4f}"
            f"{validation['maximum']:15.4f}"
        )

    # --------------------------------------------------------
    # RANK MODELS BY VALIDATION PERFORMANCE
    # --------------------------------------------------------

    ranked = sorted(
        results.items(),
        key=lambda item: item[1]["validation"]["aggregate"]
    )

    print("\nRANKING BY VALIDATION ERROR")

    print("-" * 50)

    for index, (name, result) in enumerate(ranked, 1):

        print(
            f"{index}. {name}: "
            f"{result['validation']['aggregate']:.4f} px"
        )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    output = {

        "projector_resolution": [
            PROJECTOR_WIDTH,
            PROJECTOR_HEIGHT
        ],

        "number_of_poses": len(poses),

        "number_of_correspondences": sum(
            len(pose[0]) for pose in poses
        ),

        "validation_method": "leave_one_pose_out",

        "models": results,

        "ranking": [
            name for name, _ in ranked
        ]

    }

    with open(OUTPUT_JSON, "w") as f:

        json.dump(
            output,
            f,
            indent=4
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
            "model",
            "pose",
            "calibration_rms",
            "validation_rms",
            "validation_success"
        ])

        for model_name, result in results.items():

            validation_lookup = {
                item["pose"]: item
                for item in result["validation"]["folds"]
            }

            for item in result["per_pose_errors"]:

                validation_item = validation_lookup[
                    item["pose"]
                ]

                writer.writerow([
                    model_name,
                    item["pose"],
                    item["rms"],
                    validation_item["rms"],
                    validation_item["success"]
                ])

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)

    print("DIAGNOSTICS COMPLETE")

    print("=" * 65)

    print("JSON:", OUTPUT_JSON)

    print("CSV:", OUTPUT_CSV)

    print("\nBest validation model:")

    print(ranked[0][0])

    print(
        "Validation RMS:",
        f"{ranked[0][1]['validation']['aggregate']:.4f} px"
    )

    print("=" * 65)


if __name__ == "__main__":

    main()
