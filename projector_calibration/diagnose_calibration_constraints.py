import json
import csv

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT /
    "calibration_data" /
    "projector"
)

INPUT_NPZ = (
    DATASET_DIR /
    "projector_correspondences.npz"
)

OUTPUT_DIR = (
    DATASET_DIR /
    "constraint_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WIDTH = 1920
HEIGHT = 1080

IMAGE_SIZE = (WIDTH, HEIGHT)

MINIMUM_CORNERS = 12

# Checkerboard observations must be finite.
# All models use k1-only distortion.


# ============================================================
# CALIBRATION CONFIGURATIONS
# ============================================================

CONFIGURATIONS = {

    "A_free": {
        "description": "Free aspect ratio, free principal point",
        "aspect_fixed": False,
        "principal_fixed": False
    },

    "B_square_pixels": {
        "description": "Square pixels, free principal point",
        "aspect_fixed": True,
        "principal_fixed": False
    },

    "C_centered": {
        "description": "Free aspect ratio, centered principal point",
        "aspect_fixed": False,
        "principal_fixed": True
    },

    "D_square_centered": {
        "description": "Square pixels, centered principal point",
        "aspect_fixed": True,
        "principal_fixed": True
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

    poses = []

    for i, name in enumerate(pose_names):

        mask = valid[i].astype(bool)

        obj = object_points[i][mask]

        img = projector_points[i][mask]

        finite = (
            np.isfinite(obj).all(axis=1)
            &
            np.isfinite(img).all(axis=1)
        )

        obj = obj[finite]

        img = img[finite]

        if len(obj) < MINIMUM_CORNERS:
            continue

        poses.append({

            "name": str(name),

            "object": obj.astype(
                np.float32
            ).reshape(-1, 1, 3),

            "image": img.astype(
                np.float32
            ).reshape(-1, 1, 2)

        })

    return poses


# ============================================================
# INITIAL CAMERA MATRIX
# ============================================================

def create_initial_matrix(poses, config):

    object_points = [
        p["object"] for p in poses
    ]

    image_points = [
        p["image"] for p in poses
    ]

    K = cv2.initCameraMatrix2D(
        object_points,
        image_points,
        IMAGE_SIZE,
        0
    )

    # Square-pixel constraint.
    #
    # CALIB_FIX_ASPECT_RATIO preserves fx/fy.
    # Therefore we explicitly initialize both equally.

    if config["aspect_fixed"]:

        focal = np.sqrt(
            K[0, 0] * K[1, 1]
        )

        K[0, 0] = focal
        K[1, 1] = focal

    # Principal-point constraint.

    if config["principal_fixed"]:

        K[0, 2] = WIDTH / 2
        K[1, 2] = HEIGHT / 2

    return K


# ============================================================
# CALIBRATION
# ============================================================

def calibrate(poses, config):

    K = create_initial_matrix(
        poses,
        config
    )

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        |
        cv2.CALIB_ZERO_TANGENT_DIST
        |
        cv2.CALIB_FIX_K2
        |
        cv2.CALIB_FIX_K3
    )

    if config["aspect_fixed"]:

        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    if config["principal_fixed"]:

        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    object_points = [
        p["object"] for p in poses
    ]

    image_points = [
        p["image"] for p in poses
    ]

    criteria = (
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_MAX_ITER,
        500,
        1e-10
    )

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        IMAGE_SIZE,
        K,
        np.zeros((5, 1), dtype=np.float64),
        flags=flags,
        criteria=criteria
    )

    return {

        "rms": float(rms),

        "K": K,

        "D": D,

        "rvecs": rvecs,

        "tvecs": tvecs

    }


# ============================================================
# REPROJECTION ERROR
# ============================================================

def calculate_error(
    pose,
    K,
    D,
    rvec,
    tvec
):

    projected, _ = cv2.projectPoints(
        pose["object"],
        rvec,
        tvec,
        K,
        D
    )

    projected = projected.reshape(-1, 2)

    observed = pose["image"].reshape(-1, 2)

    residuals = projected - observed

    distances = np.linalg.norm(
        residuals,
        axis=1
    )

    return float(
        np.sqrt(np.mean(distances ** 2))
    )


# ============================================================
# FIELD OF VIEW
# ============================================================

def calculate_fov(K):

    fx = K[0, 0]
    fy = K[1, 1]

    fov_x = np.degrees(
        2 * np.arctan(WIDTH / (2 * fx))
    )

    fov_y = np.degrees(
        2 * np.arctan(HEIGHT / (2 * fy))
    )

    return float(fov_x), float(fov_y)


# ============================================================
# LEAVE-ONE-POSE-OUT VALIDATION
# ============================================================

def validate(poses, config, model_name):

    results = []

    print("\nLEAVE-ONE-POSE-OUT:", model_name)

    print("-" * 65)

    for i, held_out in enumerate(poses):

        training = [
            p for j, p in enumerate(poses)
            if j != i
        ]

        try:

            result = calibrate(
                training,
                config
            )

            success, rvec, tvec = cv2.solvePnP(
                held_out["object"],
                held_out["image"],
                result["K"],
                result["D"],
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                raise RuntimeError("solvePnP failed")

            error = calculate_error(
                held_out,
                result["K"],
                result["D"],
                rvec,
                tvec
            )

            record = {

                "pose": held_out["name"],

                "rms": error,

                "fx": float(result["K"][0, 0]),

                "fy": float(result["K"][1, 1]),

                "cx": float(result["K"][0, 2]),

                "cy": float(result["K"][1, 2]),

                "k1": float(result["D"][0, 0])

            }

            results.append(record)

            print(
                f"{held_out['name']:12s} "
                f"RMS={error:.4f} "
                f"fx={record['fx']:.1f} "
                f"fy={record['fy']:.1f} "
                f"cy={record['cy']:.1f}"
            )

        except (cv2.error, RuntimeError, ValueError) as e:

            print(
                f"{held_out['name']:12s} FAILED: {e}"
            )

            results.append({

                "pose": held_out["name"],

                "rms": None,

                "fx": None,

                "fy": None,

                "cx": None,

                "cy": None,

                "k1": None

            })

    successful = [
        r for r in results
        if r["rms"] is not None
    ]

    if not successful:

        raise RuntimeError(
            f"All validation folds failed: {model_name}"
        )

    errors = np.array([
        r["rms"] for r in successful
    ])

    summary = {

        "successful_folds": len(successful),

        "failed_folds": len(results) - len(successful),

        "mean": float(np.mean(errors)),

        "median": float(np.median(errors)),

        "maximum": float(np.max(errors)),

        "aggregate_rms": float(
            np.sqrt(np.mean(errors ** 2))
        )

    }

    return results, summary


# ============================================================
# PARAMETER STABILITY
# ============================================================

def calculate_stability(records):

    parameters = [
        "fx",
        "fy",
        "cx",
        "cy",
        "k1"
    ]

    summary = {}

    for parameter in parameters:

        values = np.array([
            r[parameter]
            for r in records
            if r[parameter] is not None
        ])

        summary[parameter] = {

            "mean": float(np.mean(values)),

            "std": float(np.std(values)),

            "minimum": float(np.min(values)),

            "maximum": float(np.max(values)),

            "range": float(
                np.max(values) - np.min(values)
            )

        }

    return summary


# ============================================================
# PER-POSE CALIBRATION ERRORS
# ============================================================

def calculate_per_pose_errors(result, poses):

    errors = []

    for i, pose in enumerate(poses):

        error = calculate_error(
            pose,
            result["K"],
            result["D"],
            result["rvecs"][i],
            result["tvecs"][i]
        )

        errors.append({

            "pose": pose["name"],

            "rms": error

        })

    return errors


# ============================================================
# ANALYSE ONE CONFIGURATION
# ============================================================

def analyse_configuration(
    model_name,
    config,
    poses
):

    print("\n")
    print("=" * 70)

    print("CONFIGURATION:", model_name)

    print(config["description"])

    print("=" * 70)

    # --------------------------------------------------------
    # FULL CALIBRATION
    # --------------------------------------------------------

    full = calibrate(
        poses,
        config
    )

    K = full["K"]

    D = full["D"].flatten()

    fov_x, fov_y = calculate_fov(K)

    print("\nFULL CALIBRATION")

    print("RMS:", full["rms"])

    print("\nCamera matrix:")

    print(K)

    print("\nDistortion:")

    print(D)

    print("\nPrincipal point:")

    print(
        f"cx={K[0, 2]:.3f}, cy={K[1, 2]:.3f}"
    )

    print("\nFocal lengths:")

    print(
        f"fx={K[0, 0]:.3f}, fy={K[1, 1]:.3f}"
    )

    print("\nFOV:")

    print(
        f"Horizontal: {fov_x:.3f} degrees"
    )

    print(
        f"Vertical: {fov_y:.3f} degrees"
    )

    per_pose = calculate_per_pose_errors(
        full,
        poses
    )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    validation_records, validation_summary = validate(
        poses,
        config,
        model_name
    )

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    stability = calculate_stability(
        validation_records
    )

    print("\nVALIDATION SUMMARY")

    for key, value in validation_summary.items():

        print(f"{key}: {value}")

    print("\nINTRINSIC STABILITY")

    for parameter, stats in stability.items():

        print(
            f"{parameter:5s}: "
            f"mean={stats['mean']:.3f}, "
            f"std={stats['std']:.3f}, "
            f"range={stats['range']:.3f}"
        )

    return {

        "description": config["description"],

        "calibration_rms": full["rms"],

        "camera_matrix": K.tolist(),

        "distortion": D.tolist(),

        "fx": float(K[0, 0]),

        "fy": float(K[1, 1]),

        "cx": float(K[0, 2]),

        "cy": float(K[1, 2]),

        "horizontal_fov": fov_x,

        "vertical_fov": fov_y,

        "principal_point_inside_image": bool(
            0 <= K[0, 2] < WIDTH
            and
            0 <= K[1, 2] < HEIGHT
        ),

        "per_pose_errors": per_pose,

        "validation": {

            "summary": validation_summary,

            "folds": validation_records

        },

        "stability": stability

    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):

    output = OUTPUT_DIR / "constraint_comparison.csv"

    with open(
        output,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            "model",
            "pose",
            "validation_rms",
            "fx",
            "fy",
            "cx",
            "cy",
            "k1"

        ])

        for model_name, result in results.items():

            for fold in result["validation"]["folds"]:

                writer.writerow([

                    model_name,

                    fold["pose"],

                    fold["rms"],

                    fold["fx"],

                    fold["fy"],

                    fold["cx"],

                    fold["cy"],

                    fold["k1"]

                ])

    print("\nSaved:", output)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPROJECTOR CALIBRATION CONSTRAINT DIAGNOSTICS")

    print("=" * 70)

    print("Dataset:", DATASET_DIR)

    poses = load_data()

    print("\nUsable poses:", len(poses))

    print(
        "Total correspondences:",
        sum(len(p["object"]) for p in poses)
    )

    if len(poses) < 5:

        raise RuntimeError(
            "Insufficient calibration poses"
        )

    results = {}

    # --------------------------------------------------------
    # ANALYSE FOUR CONFIGURATIONS
    # --------------------------------------------------------

    for model_name, config in CONFIGURATIONS.items():

        results[model_name] = analyse_configuration(
            model_name,
            config,
            poses
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)

    print("CONFIGURATION COMPARISON")

    print("=" * 90)

    print(
        f"{'Model':22s}"
        f"{'Calib RMS':>13s}"
        f"{'Val RMS':>13s}"
        f"{'Val Median':>13s}"
        f"{'fx std':>12s}"
        f"{'cy std':>12s}"
    )

    print("-" * 90)

    for name, result in results.items():

        validation = result["validation"]["summary"]

        stability = result["stability"]

        print(
            f"{name:22s}"
            f"{result['calibration_rms']:13.4f}"
            f"{validation['aggregate_rms']:13.4f}"
            f"{validation['median']:13.4f}"
            f"{stability['fx']['std']:12.2f}"
            f"{stability['cy']['std']:12.2f}"
        )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    ranking = sorted(
        results.items(),
        key=lambda item: (
            item[1]["validation"]["summary"]["aggregate_rms"]
        )
    )

    print("\nRANKING BY VALIDATION RMS")

    for i, (name, result) in enumerate(ranking, 1):

        print(
            f"{i}. {name}: "
            f"{result['validation']['summary']['aggregate_rms']:.4f} px"
        )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    report = {

        "projector_resolution": [
            WIDTH,
            HEIGHT
        ],

        "number_of_poses": len(poses),

        "number_of_correspondences": sum(
            len(p["object"]) for p in poses
        ),

        "configurations": results,

        "ranking": [
            name for name, _ in ranking
        ]

    }

    json_path = (
        OUTPUT_DIR /
        "constraint_diagnostics.json"
    )

    with open(json_path, "w") as f:

        json.dump(
            report,
            f,
            indent=4
        )

    save_csv(results)

    print("\n")
    print("=" * 70)

    print("DIAGNOSTICS COMPLETE")

    print("=" * 70)

    print("Output directory:", OUTPUT_DIR)

    print("JSON:", json_path)

    print("=" * 70)


if __name__ == "__main__":

    main()
