from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = PROJECT_ROOT / "calibration_data" / "projector"

CORRESPONDENCE_FILE = DATASET / "projector_correspondences.npz"

OUTPUT_DIR = PROJECT_ROOT / "calibration_data" / "extrinsics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CAMERA INTRINSICS
# ============================================================

CAMERA_MATRIX = np.array([
    [1060.53706, 0.0, 663.877816],
    [0.0, 1060.68969, 519.283264],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

CAMERA_DISTORTION = np.array([
    0.05706026,
    -0.26159217,
    0.00266085,
    0.00262938,
    0.24140464
], dtype=np.float64)


# ============================================================
# PROJECTOR INTRINSICS
# Model D_simple: best cross-validation result
# ============================================================

PROJECTOR_MATRIX = np.array([
    [1265.86260, 0.0, 965.747930],
    [0.0, 1255.15936, -118.757617],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

PROJECTOR_DISTORTION = np.array([
    0.17089973,
    0.0,
    0.0,
    0.0,
    0.0
], dtype=np.float64)


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

CAMERA_SIZE = (1280, 1024)

PROJECTOR_SIZE = (1920, 1080)


# ============================================================
# SETTINGS
# ============================================================

MIN_POINTS = 12

# Outlier rejection thresholds
MIN_TRANSLATION_THRESHOLD_MM = 15.0
MIN_ROTATION_THRESHOLD_DEG = 2.0

# Iterative rotation averaging
ROTATION_ITERATIONS = 50

CRITERIA = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-10
)


# ============================================================
# LOAD CORRESPONDENCES
# ============================================================

def load_data():

    with np.load(CORRESPONDENCE_FILE) as data:

        required = [
            "object_points",
            "camera_points",
            "projector_points",
            "valid",
            "pose_names"
        ]

        for key in required:

            if key not in data.files:
                raise RuntimeError(
                    f"Missing array: {key}"
                )

        return {
            key: data[key].copy()
            for key in required
        }


# ============================================================
# PREPARE POSES
# ============================================================

def prepare_poses(data):

    poses = []

    print("\nAVAILABLE CALIBRATION POSES")
    print("=" * 65)

    for i, name in enumerate(data["pose_names"]):

        obj = data["object_points"][i]

        cam = data["camera_points"][i]

        proj = data["projector_points"][i]

        valid = data["valid"][i].astype(bool)

        valid &= np.isfinite(obj).all(axis=1)
        valid &= np.isfinite(cam).all(axis=1)
        valid &= np.isfinite(proj).all(axis=1)

        obj = obj[valid].astype(np.float64)
        cam = cam[valid].astype(np.float64)
        proj = proj[valid].astype(np.float64)

        if len(obj) < MIN_POINTS:

            print(f"{name}: SKIPPED")

            continue

        poses.append({

            "name": str(name),

            "object": obj,

            "camera": cam,

            "projector": proj

        })

        print(
            f"{name:12s}: {len(obj)} points"
        )

    if len(poses) < 3:

        raise RuntimeError(
            "Insufficient calibration poses."
        )

    print("\nUsable poses:", len(poses))

    return poses

def project_points(obj, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(
        obj, rvec, tvec, K, dist
    )
    return projected.reshape(-1, 2)


def optimize_joint_stereo(poses, initial_R, initial_T):

    print("\nJOINT NONLINEAR STEREO OPTIMIZATION")
    print("=" * 65)

    # Initialize each checkerboard pose from camera observations.
    board_poses = []

    for pose in poses:
        result = solve_planar_pnp(
            pose["object"],
            pose["camera"],
            CAMERA_MATRIX,
            CAMERA_DISTORTION
        )

        board_poses.append((
            result["rvec"].reshape(3),
            result["tvec"].reshape(3)
        ))

    global_rvec, _ = cv2.Rodrigues(initial_R)

    # Parameter vector:
    # global rotation (3)
    # global translation (3)
    # individual board poses (6 per pose)

    x0 = np.concatenate([
        global_rvec.reshape(3),
        initial_T.reshape(3),
        *[
            np.concatenate([r, t])
            for r, t in board_poses
        ]
    ])

    def residuals(parameters):

        rvec_pc = parameters[:3]
        t_pc = parameters[3:6].reshape(3, 1)

        R_pc, _ = cv2.Rodrigues(rvec_pc)

        all_residuals = []

        for i, pose in enumerate(poses):

            offset = 6 + i * 6

            rvec_cb = parameters[offset:offset + 3]
            t_cb = parameters[offset + 3:offset + 6].reshape(3, 1)

            obj = pose["object"]

            # Camera prediction.
            predicted_camera = project_points(
                obj,
                rvec_cb,
                t_cb,
                CAMERA_MATRIX,
                CAMERA_DISTORTION
            )

            # Transform checkerboard into projector coordinates.
            R_cb, _ = cv2.Rodrigues(rvec_cb)

            R_pb = R_pc @ R_cb
            t_pb = R_pc @ t_cb + t_pc

            rvec_pb, _ = cv2.Rodrigues(R_pb)

            predicted_projector = project_points(
                obj,
                rvec_pb,
                t_pb,
                PROJECTOR_MATRIX,
                PROJECTOR_DISTORTION
            )

            # Normalize by image measurement uncertainty.
            camera_error = (
                predicted_camera - pose["camera"]
            ).reshape(-1)

            projector_error = (
                predicted_projector - pose["projector"]
            ).reshape(-1)

            all_residuals.extend(camera_error)
            all_residuals.extend(projector_error)

        return np.asarray(all_residuals)

    print("Initial residual evaluation...")

    initial_rms = np.sqrt(
        np.mean(residuals(x0) ** 2)
    )

    print(f"Initial RMS: {initial_rms:.4f} px")
    print("Running nonlinear optimization...")

    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=3000,
        verbose=1
    )

    if not result.success:
        print("WARNING:", result.message)

    parameters = result.x

    R_pc, _ = cv2.Rodrigues(
        parameters[:3]
    )

    T_pc = parameters[3:6].reshape(3)

    print("\nOptimization complete.")
    print("Converged:", result.success)
    print("Function evaluations:", result.nfev)
    print("Final robust cost:", result.cost)

    return {
        "R": R_pc,
        "T": T_pc,
        "parameters": parameters,
        "optimization": result,
        "initial_rms": float(initial_rms),
        "final_rms": float(
            np.sqrt(np.mean(residuals(parameters) ** 2))
        )
    }


def validate_joint_solution(poses, solution):

    parameters = solution["parameters"]

    R_pc = solution["R"]
    T_pc = solution["T"].reshape(3, 1)

    results = []

    camera_squared = 0.0
    projector_squared = 0.0
    total_points = 0

    print("\nJOINT TRANSFORMATION VALIDATION")
    print("=" * 65)

    for i, pose in enumerate(poses):

        offset = 6 + i * 6

        rvec_cb = parameters[offset:offset + 3]
        t_cb = parameters[offset + 3:offset + 6].reshape(3, 1)

        obj = pose["object"]

        predicted_camera = project_points(
            obj,
            rvec_cb,
            t_cb,
            CAMERA_MATRIX,
            CAMERA_DISTORTION
        )

        R_cb, _ = cv2.Rodrigues(rvec_cb)

        R_pb = R_pc @ R_cb
        t_pb = R_pc @ t_cb + T_pc

        rvec_pb, _ = cv2.Rodrigues(R_pb)

        predicted_projector = project_points(
            obj,
            rvec_pb,
            t_pb,
            PROJECTOR_MATRIX,
            PROJECTOR_DISTORTION
        )

        camera_errors = np.linalg.norm(
            predicted_camera - pose["camera"],
            axis=1
        )

        projector_errors = np.linalg.norm(
            predicted_projector - pose["projector"],
            axis=1
        )

        camera_rms = float(
            np.sqrt(np.mean(camera_errors ** 2))
        )

        projector_rms = float(
            np.sqrt(np.mean(projector_errors ** 2))
        )

        camera_squared += np.sum(camera_errors ** 2)
        projector_squared += np.sum(projector_errors ** 2)
        total_points += len(obj)

        results.append({
            "pose": pose["name"],
            "points": len(obj),
            "camera_rms_px": camera_rms,
            "projector_rms_px": projector_rms,
            "camera_max_px": float(camera_errors.max()),
            "projector_max_px": float(projector_errors.max())
        })

        print(
            f"{pose['name']:12s} "
            f"Camera: {camera_rms:.3f} px | "
            f"Projector: {projector_rms:.3f} px"
        )

    camera_rms = np.sqrt(camera_squared / total_points)
    projector_rms = np.sqrt(projector_squared / total_points)

    combined_rms = np.sqrt(
        (camera_squared + projector_squared)
        / (2 * total_points)
    )

    print("\nVALIDATION SUMMARY")
    print("-" * 65)
    print(f"Camera RMS:     {camera_rms:.4f} px")
    print(f"Projector RMS:  {projector_rms:.4f} px")
    print(f"Combined RMS:   {combined_rms:.4f} px")

    return {
        "poses": results,
        "camera_rms_px": float(camera_rms),
        "projector_rms_px": float(projector_rms),
        "combined_rms_px": float(combined_rms)
    }


# ============================================================
# JOINT CAMERA-PROJECTOR STEREO CALIBRATION
# ============================================================

def calibrate_stereo(poses):

    print("\n")
    print("=" * 65)
    print("JOINT CAMERA-PROJECTOR STEREO CALIBRATION")
    print("=" * 65)

    object_points = []
    camera_points = []
    projector_points = []

    for pose in poses:

        object_points.append(
            pose["object"].astype(np.float32).reshape(-1, 1, 3)
        )

        camera_points.append(
            pose["camera"].astype(np.float32).reshape(-1, 1, 2)
        )

        projector_points.append(
            pose["projector"].astype(np.float32).reshape(-1, 1, 2)
        )

    camera_matrix = CAMERA_MATRIX.copy()
    camera_distortion = CAMERA_DISTORTION.reshape(1, -1).copy()

    projector_matrix = PROJECTOR_MATRIX.copy()
    projector_distortion = PROJECTOR_DISTORTION.reshape(1, -1).copy()

    flags = cv2.CALIB_FIX_INTRINSIC

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-10
    )

    print("Calibration poses:", len(poses))
    print("Camera resolution:", CAMERA_SIZE)
    print("Projector resolution:", PROJECTOR_SIZE)
    print("Intrinsics: FIXED")
    print("Optimization: JOINT STEREO CALIBRATION")

    result = cv2.stereoCalibrate(
        object_points,
        camera_points,
        projector_points,

        camera_matrix,
        camera_distortion,

        projector_matrix,
        projector_distortion,

        CAMERA_SIZE,

        criteria=criteria,
        flags=flags
    )

    (
        rms,
        K1,
        D1,
        K2,
        D2,
        R,
        T,
        E,
        F
    ) = result

    # Verify that OpenCV did not alter the fixed intrinsics.
    if not np.allclose(K1, CAMERA_MATRIX):
        raise RuntimeError("Camera intrinsics unexpectedly changed.")

    if not np.allclose(K2, PROJECTOR_MATRIX):
        raise RuntimeError("Projector intrinsics unexpectedly changed.")

    projector_position = -R.T @ T.reshape(3)

    print("\nSTEREO CALIBRATION RESULT")
    print("-" * 65)

    print("Stereo RMS:", rms)

    print("\nCamera -> Projector rotation:")
    print(R)

    print("\nCamera -> Projector translation (mm):")
    print(T.reshape(3))

    print("\nProjector optical centre in camera coordinates (mm):")
    print(projector_position)

    return {
        "R": R,
        "T": T.reshape(3),
        "E": E,
        "F": F,
        "rms": float(rms),
        "projector_position": projector_position
    }


# ============================================================
# PLANAR PNP CANDIDATES
# ============================================================

def get_pnp_candidates(obj, img, K, D):

    result = cv2.solvePnPGeneric(
        obj,
        img,
        K,
        D,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not result[0]:
        raise RuntimeError("IPPE failed.")

    candidates = []

    for rvec, tvec in zip(result[1], result[2]):

        rvec = np.asarray(rvec).reshape(3, 1)
        tvec = np.asarray(tvec).reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)

        points_3d = R @ obj.T + tvec

        if np.any(points_3d[2] <= 0):
            continue

        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            K,
            D
        )

        errors = np.linalg.norm(
            projected.reshape(-1, 2) - img,
            axis=1
        )

        candidates.append({
            "R": R,
            "rvec": rvec,
            "tvec": tvec,
            "rms": float(np.sqrt(np.mean(errors ** 2)))
        })

    if not candidates:
        raise RuntimeError("No valid PnP solutions.")

    return candidates


# ============================================================
# VALIDATE JOINT TRANSFORMATION
# ============================================================

def validate_stereo(poses, calibration):

    R_PC = calibration["R"]
    t_PC = calibration["T"].reshape(3, 1)

    results = []

    total_squared_error = 0.0
    total_points = 0

    print("\n")
    print("=" * 65)
    print("JOINT TRANSFORMATION VALIDATION")
    print("=" * 65)

    for pose in poses:

        name = pose["name"]

        obj = pose["object"]
        cam = pose["camera"]
        proj = pose["projector"]

        candidates = get_pnp_candidates(
            obj,
            cam,
            CAMERA_MATRIX,
            CAMERA_DISTORTION
        )

        best = None

        for candidate in candidates:

            R_CB = candidate["R"]
            t_CB = candidate["tvec"]

            # Board pose in projector coordinates.
            R_PB = R_PC @ R_CB
            t_PB = R_PC @ t_CB + t_PC

            if np.any((R_PB @ obj.T + t_PB)[2] <= 0):
                continue

            rvec_PB, _ = cv2.Rodrigues(R_PB)

            predicted_camera, _ = cv2.projectPoints(
                obj,
                candidate["rvec"],
                t_CB,
                CAMERA_MATRIX,
                CAMERA_DISTORTION
            )

            predicted_projector, _ = cv2.projectPoints(
                obj,
                rvec_PB,
                t_PB,
                PROJECTOR_MATRIX,
                PROJECTOR_DISTORTION
            )

            camera_errors = np.linalg.norm(
                predicted_camera.reshape(-1, 2) - cam,
                axis=1
            )

            projector_errors = np.linalg.norm(
                predicted_projector.reshape(-1, 2) - proj,
                axis=1
            )

            camera_rms = float(
                np.sqrt(np.mean(camera_errors ** 2))
            )

            projector_rms = float(
                np.sqrt(np.mean(projector_errors ** 2))
            )

            # Combined score for selecting the planar PnP solution.
            combined_rms = float(
                np.sqrt(
                    (
                        np.sum(camera_errors ** 2) +
                        np.sum(projector_errors ** 2)
                    ) /
                    (2 * len(obj))
                )
            )

            if best is None or combined_rms < best["combined_rms"]:

                best = {
                    "camera_rms": camera_rms,
                    "projector_rms": projector_rms,
                    "combined_rms": combined_rms,
                    "camera_errors": camera_errors,
                    "projector_errors": projector_errors
                }

        if best is None:
            raise RuntimeError(
                f"No valid joint PnP solution for {name}"
            )

        n = len(obj)

        total_squared_error += (
            np.sum(best["camera_errors"] ** 2) +
            np.sum(best["projector_errors"] ** 2)
        )

        total_points += 2 * n

        results.append({
            "pose": name,
            "points": n,
            "camera_rms_px": best["camera_rms"],
            "projector_rms_px": best["projector_rms"],
            "combined_rms_px": best["combined_rms"]
        })

        print(
            f"{name:12s} "
            f"Camera: {best['camera_rms']:.3f} px | "
            f"Projector: {best['projector_rms']:.3f} px | "
            f"Combined: {best['combined_rms']:.3f} px"
        )

    aggregate_rms = float(
        np.sqrt(total_squared_error / total_points)
    )

    print("\nAggregate combined reprojection RMS:")
    print(f"{aggregate_rms:.4f} px")

    return results, aggregate_rms



# ============================================================
# PLANAR PNP
# ============================================================

def solve_planar_pnp(
    object_points,
    image_points,
    camera_matrix,
    distortion
):

    obj = np.asarray(
        object_points,
        dtype=np.float64
    ).reshape(-1, 3)

    img = np.asarray(
        image_points,
        dtype=np.float64
    ).reshape(-1, 2)

    # IPPE requires planar object points.
    centered = obj - obj.mean(axis=0)

    singular_values = np.linalg.svd(
        centered,
        compute_uv=False
    )

    if singular_values[0] == 0:

        raise RuntimeError(
            "Degenerate object points."
        )

    if singular_values[-1] > singular_values[0] * 1e-5:

        raise RuntimeError(
            "Checkerboard object points are not planar."
        )

    # Generate both planar pose solutions.
    result = cv2.solvePnPGeneric(
        obj,
        img,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not result[0]:

        raise RuntimeError(
            "IPPE solvePnP failed."
        )

    candidates = []

    for rvec, tvec in zip(result[1], result[2]):

        rvec = np.asarray(rvec).reshape(3, 1)

        tvec = np.asarray(tvec).reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)

        camera_points_3d = (
            R @ obj.T + tvec
        )

        # Reject solutions with points behind the camera.
        if np.any(camera_points_3d[2] <= 0):

            continue

        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            camera_matrix,
            distortion
        )

        projected = projected.reshape(-1, 2)

        errors = np.linalg.norm(
            projected - img,
            axis=1
        )

        rms = float(
            np.sqrt(np.mean(errors ** 2))
        )

        candidates.append({
            "rvec": rvec,
            "tvec": tvec,
            "R": R,
            "rms": rms
        })

    if not candidates:

        raise RuntimeError(
            "No physically valid IPPE solution."
        )

    return min(
        candidates,
        key=lambda x: x["rms"]
    )


# ============================================================
# CALCULATE INDEPENDENT TRANSFORMATIONS
# ============================================================

def calculate_pose_transformations(poses):

    results = []

    print("\nINDEPENDENT POSE CALIBRATION")
    print("=" * 65)

    for pose in poses:

        name = pose["name"]

        obj = pose["object"]

        cam = pose["camera"]

        proj = pose["projector"]

        try:

            camera_pnp = solve_planar_pnp(
                obj,
                cam,
                CAMERA_MATRIX,
                CAMERA_DISTORTION
            )

            projector_pnp = solve_planar_pnp(
                obj,
                proj,
                PROJECTOR_MATRIX,
                PROJECTOR_DISTORTION
            )

            R_CB = camera_pnp["R"]

            t_CB = camera_pnp["tvec"]

            R_PB = projector_pnp["R"]

            t_PB = projector_pnp["tvec"]

            # Camera -> Projector
            R_PC = R_PB @ R_CB.T

            t_PC = t_PB - R_PC @ t_CB

            # Projector optical centre in camera coordinates.
            projector_position = (
                -R_PC.T @ t_PC
            ).reshape(3)

            result = {

                "name": name,

                "R": R_PC,

                "T": t_PC.reshape(3),

                "projector_position": projector_position,

                "camera_rms": camera_pnp["rms"],

                "projector_rms": projector_pnp["rms"]

            }

            results.append(result)

            print(
                f"{name:12s} "
                f"Camera: {camera_pnp['rms']:.3f} px | "
                f"Projector: {projector_pnp['rms']:.3f} px"
            )

            print(
                "  Translation:",
                np.round(result["T"], 2)
            )

        except Exception as exc:

            print(
                f"{name}: FAILED - {exc}"
            )

    if len(results) < 3:

        raise RuntimeError(
            "Too few successful pose transformations."
        )

    return results


# ============================================================
# ROTATION DISTANCE
# ============================================================

def rotation_distance_deg(R1, R2):

    relative = R1 @ R2.T

    cos_angle = (
        np.trace(relative) - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    return float(
        np.degrees(np.arccos(cos_angle))
    )


# ============================================================
# ROBUST MEDIAN
# ============================================================

def robust_threshold(values, minimum_threshold):

    values = np.asarray(values)

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    robust_sigma = 1.4826 * mad

    threshold = median + max(
        3.0 * robust_sigma,
        minimum_threshold
    )

    return float(threshold)


# ============================================================
# ROBUST ROTATION AVERAGING
# ============================================================

def average_rotations(rotations):

    rotations = list(rotations)

    # Find the rotation medoid.
    scores = []

    for i, R in enumerate(rotations):

        score = sum(
            rotation_distance_deg(R, other)
            for j, other in enumerate(rotations)
            if i != j
        )

        scores.append(score)

    R_mean = rotations[int(np.argmin(scores))].copy()

    # Iterative SO(3) averaging.
    for _ in range(ROTATION_ITERATIONS):

        residuals = []

        for R in rotations:

            delta = R @ R_mean.T

            rvec, _ = cv2.Rodrigues(delta)

            residuals.append(rvec.reshape(3))

        update = np.mean(
            residuals,
            axis=0
        ).reshape(3, 1)

        delta_R, _ = cv2.Rodrigues(update)

        R_mean = delta_R @ R_mean

        if np.linalg.norm(update) < 1e-12:

            break

    return R_mean


# ============================================================
# ROBUST CONSENSUS
# ============================================================

def calculate_consensus(results):

    translations = np.array([
        r["T"] for r in results
    ])

    rotations = [
        r["R"] for r in results
    ]

    # Initial robust translation.
    median_translation = np.median(
        translations,
        axis=0
    )

    translation_distances = np.linalg.norm(
        translations - median_translation,
        axis=1
    )

    # Initial robust rotation.
    initial_rotation = average_rotations(
        rotations
    )

    rotation_distances = np.array([
        rotation_distance_deg(
            R,
            initial_rotation
        )
        for R in rotations
    ])

    translation_threshold = robust_threshold(
        translation_distances,
        MIN_TRANSLATION_THRESHOLD_MM
    )

    rotation_threshold = robust_threshold(
        rotation_distances,
        MIN_ROTATION_THRESHOLD_DEG
    )

    inliers = (
        (translation_distances <= translation_threshold)
        &
        (rotation_distances <= rotation_threshold)
    )

    print("\n")
    print("=" * 65)

    print("POSE CONSENSUS AND OUTLIER DETECTION")

    print("=" * 65)

    print(
        "Translation threshold:",
        f"{translation_threshold:.2f} mm"
    )

    print(
        "Rotation threshold:",
        f"{rotation_threshold:.2f} degrees"
    )

    for i, result in enumerate(results):

        status = "INLIER" if inliers[i] else "OUTLIER"

        print(
            f"{result['name']:12s} "
            f"Translation deviation: "
            f"{translation_distances[i]:8.2f} mm | "
            f"Rotation deviation: "
            f"{rotation_distances[i]:6.2f} deg | "
            f"{status}"
        )

    if np.sum(inliers) < 3:

        raise RuntimeError(
            "Insufficient consensus poses."
        )

    # Recalculate using inliers only.
    final_translations = translations[inliers]

    final_rotations = [
        rotations[i]
        for i in range(len(results))
        if inliers[i]
    ]

    final_T = np.median(
        final_translations,
        axis=0
    )

    final_R = average_rotations(
        final_rotations
    )

    return {

        "R": final_R,

        "T": final_T,

        "inliers": inliers,

        "translation_distances": translation_distances,

        "rotation_distances": rotation_distances,

        "translation_threshold": translation_threshold,

        "rotation_threshold": rotation_threshold

    }


# ============================================================
# FINAL REPROJECTION VALIDATION
# ============================================================

def validate_final_transformation(
    poses,
    consensus,
    pose_comparison
):

    R_PC = consensus["R"]

    t_PC = consensus["T"].reshape(3, 1)

    results = []

    print("\n")
    print("=" * 65)

    print("FINAL TRANSFORMATION VALIDATION")

    print("=" * 65)

    for pose in poses:

        name = pose["name"]

        obj = pose["object"]

        cam = pose["camera"]

        proj = pose["projector"]

        camera_pnp = solve_planar_pnp(
            obj,
            cam,
            CAMERA_MATRIX,
            CAMERA_DISTORTION
        )

        R_CB = camera_pnp["R"]

        t_CB = camera_pnp["tvec"]

        # Board pose in projector coordinates.
        R_PB = R_PC @ R_CB

        t_PB = R_PC @ t_CB + t_PC

        rvec_PB, _ = cv2.Rodrigues(R_PB)

        predicted, _ = cv2.projectPoints(
            obj,
            rvec_PB,
            t_PB,
            PROJECTOR_MATRIX,
            PROJECTOR_DISTORTION
        )

        predicted = predicted.reshape(-1, 2)

        errors = np.linalg.norm(
            predicted - proj,
            axis=1
        )

        rms = float(
            np.sqrt(np.mean(errors ** 2))
        )

        results.append({

            "pose": name,

            "points": len(obj),

            "camera_rms_px": camera_pnp["rms"],

            "projector_rms_px": rms,

            "projector_max_px": float(errors.max()),

            "inlier": bool(
                consensus["inliers"][
                    next(
                        i for i, r in enumerate(pose_comparison)
                        if r["name"] == name
                    )
                ]
            )

        })

        print(
            f"{name:12s} "
            f"Projector RMS: {rms:.3f} px"
        )

    return results


# ============================================================
# TRANSFORMATION MATRICES
# ============================================================

def calculate_matrices(R, T):

    T_camera_to_projector = np.eye(4)

    T_camera_to_projector[:3, :3] = R

    T_camera_to_projector[:3, 3] = T

    R_inverse = R.T

    T_inverse = -R_inverse @ T

    T_projector_to_camera = np.eye(4)

    T_projector_to_camera[:3, :3] = R_inverse

    T_projector_to_camera[:3, 3] = T_inverse

    return (
        T_camera_to_projector,
        T_projector_to_camera,
        T_inverse
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    consensus,
    # pose_results,
    validation_results,
    pose_comparison
):

    R = consensus["R"]

    T = consensus["T"]

    T_cp, T_pc, projector_position = (
        calculate_matrices(R, T)
    )

    inlier_results = [
        r for r, valid in zip(
            pose_comparison,
            consensus["inliers"]
        )
        if valid
    ]

    report = {

        "method": "Independent per-pose PnP with robust consensus",

        "intrinsic_model": "D_simple",

        "camera_to_projector_rotation": R.tolist(),

        "camera_to_projector_translation_mm": T.tolist(),

        "camera_to_projector_matrix": T_cp.tolist(),

        "projector_to_camera_matrix": T_pc.tolist(),

        "projector_position_in_camera_mm": (
            projector_position.tolist()
        ),

        "number_of_poses": len(pose_comparison),

        "number_of_inliers": int(
            np.sum(consensus["inliers"])
        ),

        "number_of_outliers": int(
            len(consensus["inliers"]) -
            np.sum(consensus["inliers"])
        ),

        "translation_threshold_mm": (
            consensus["translation_threshold"]
        ),

        "rotation_threshold_deg": (
            consensus["rotation_threshold"]
        ),

        "pose_comparison": pose_comparison,

        "final_reprojection_validation": validation_results,

        "coordinate_convention": {

            "camera": "OpenCV X right, Y down, Z forward",

            "projector": "OpenCV X right, Y down, Z forward",

            "translation_units": "millimetres",

            "xtion_mounting_correction_applied": False

        }

    }

    json_path = OUTPUT_DIR / "camera_to_projector.json"

    with open(json_path, "w") as f:

        def json_converter(obj):

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, np.integer):
                return int(obj)

            if isinstance(obj, np.floating):
                return float(obj)

            if isinstance(obj, np.bool_):
                return bool(obj)

            raise TypeError(
                f"Object of type {type(obj).__name__} "
                "is not JSON serializable"
            )
        json.dump(
            report,
            f,
            indent=4,
            default=json_converter
        )

    np.savez_compressed(

        OUTPUT_DIR / "camera_to_projector.npz",

        R_camera_to_projector=R,

        T_camera_to_projector_mm=T,

        T_camera_to_projector=T_cp,

        T_projector_to_camera=T_pc,

        projector_position_camera_mm=projector_position,

        inliers=consensus["inliers"],

        camera_matrix=CAMERA_MATRIX,

        camera_distortion=CAMERA_DISTORTION,

        projector_matrix=PROJECTOR_MATRIX,

        projector_distortion=PROJECTOR_DISTORTION

    )

    # Save individual transformations.
    np.savez_compressed(

        OUTPUT_DIR / "extrinsic_transforms_per_pose.npz",

        pose_names=np.array([
            r["name"] for r in pose_comparison
        ]),

        rotations=np.array([
            r["R"] for r in pose_comparison
        ]),

        translations=np.array([
            r["T"] for r in pose_comparison
        ]),

        projector_positions=np.array([
            r["projector_position"]
            for r in pose_comparison
        ]),

        inliers=consensus["inliers"]

    )

    # Save CSV.
    csv_path = OUTPUT_DIR / "extrinsic_pose_comparison.csv"

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "pose",
            "camera_rms_px",
            "projector_rms_px",
            "tx_mm",
            "ty_mm",
            "tz_mm",
            "projector_x_camera_mm",
            "projector_y_camera_mm",
            "projector_z_camera_mm",
            "translation_deviation_mm",
            "rotation_deviation_deg",
            "inlier"
        ])

        for i, r in enumerate(pose_comparison):

            writer.writerow([

                r["name"],

                r["camera_rms"],

                r["projector_rms"],

                *r["T"],

                *r["projector_position"],

                consensus["translation_distances"][i],

                consensus["rotation_distances"][i],

                consensus["inliers"][i]

            ])

    print("\nSaved:")
    print(json_path)

    print(OUTPUT_DIR / "camera_to_projector.npz")

    print(OUTPUT_DIR / "extrinsic_transforms_per_pose.npz")

    print(csv_path)


# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

def create_plot(
    pose_comparison,
    validation_results,
    consensus
):

    names = [
        r["name"] for r in pose_comparison
    ]

    translations = np.array([
        r["T"] for r in pose_comparison
    ])

    translation_deviation = (
        consensus["translation_distances"]
    )

    rotation_deviation = (
        consensus["rotation_distances"]
    )

    validation_errors = np.array([
        r["projector_rms_px"]
        for r in validation_results
    ])

    inliers = consensus["inliers"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10)
    )

    # Translation estimates.
    ax = axes[0, 0]

    for i, label in enumerate(["X", "Y", "Z"]):

        ax.plot(
            translations[:, i],
            "o-",
            label=label
        )

    ax.set_title(
        "Camera-to-projector translation estimates"
    )

    ax.set_ylabel("Millimetres")

    ax.set_xticks(range(len(names)))

    ax.set_xticklabels(
        names,
        rotation=45
    )

    ax.legend()

    ax.grid(alpha=0.3)

    # Translation deviation.
    ax = axes[0, 1]

    colors = [
        "green" if x else "red"
        for x in inliers
    ]

    ax.bar(
        names,
        translation_deviation,
        color=colors
    )

    ax.axhline(
        consensus["translation_threshold"],
        color="black",
        linestyle="--"
    )

    ax.set_title(
        "Translation deviation"
    )

    ax.set_ylabel("Millimetres")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    # Rotation deviation.
    ax = axes[1, 0]

    ax.bar(
        names,
        rotation_deviation,
        color=colors
    )

    ax.axhline(
        consensus["rotation_threshold"],
        color="black",
        linestyle="--"
    )

    ax.set_title(
        "Rotation deviation"
    )

    ax.set_ylabel("Degrees")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    # Final reprojection errors.
    ax = axes[1, 1]

    ax.bar(
        names,
        validation_errors,
        color=colors
    )

    ax.set_title(
        "Final projector reprojection errors"
    )

    ax.set_ylabel("RMS pixels")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    for ax in axes.flat:

        ax.grid(
            axis="y",
            alpha=0.3
        )

    plt.tight_layout()

    path = OUTPUT_DIR / "extrinsic_reprojection.png"

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()

    print("\nSaved diagnostic plot:", path)


# ============================================================
# MAIN
# ============================================================

# def main():

#     # global consensus_results

#     print("\n")
#     print("=" * 65)

#     print("CAMERA-PROJECTOR PER-POSE EXTRINSIC CALIBRATION")

#     print("=" * 65)

#     print("Dataset:", DATASET)

#     print("Projector intrinsic model: D_simple")

#     data = load_data()

#     poses = prepare_poses(data)

#     pose_comparison = calculate_pose_transformations(
#         poses
#     )

#     # consensus_results = pose_comparison

#     consensus = calculate_consensus(
#         pose_comparison
#     )

#     print("\n")
#     print("=" * 65)

#     print("FINAL CAMERA-TO-PROJECTOR TRANSFORMATION")

#     print("=" * 65)

#     print("\nRotation:")

#     print(consensus["R"])

#     print("\nTranslation (mm):")

#     print(consensus["T"])

#     _, _, projector_position = calculate_matrices(
#         consensus["R"],
#         consensus["T"]
#     )

#     print("\nProjector optical centre in camera coordinates:")

#     print(projector_position)

#     print("\nAccepted poses:")

#     print(np.sum(consensus["inliers"]))

#     print("Rejected poses:")

#     print(
#         len(consensus["inliers"]) -
#         np.sum(consensus["inliers"])
#     )

#     # validation_results = validate_final_transformation(
#     #     poses,
#     #     consensus
#     # )

#     successful_names = {
#         r["name"] for r in pose_comparison
#     }

#     successful_poses = [
#         p for p in poses
#         if p["name"] in successful_names
#     ]

#     validation_results = validate_final_transformation(
#         successful_poses,
#         consensus,
#         pose_comparison
#     )

#     save_results(
#         consensus,
#         # validation_results,
#         validation_results,
#         pose_comparison
#     )

#     create_plot(
#         pose_comparison,
#         validation_results,
#         consensus
#     )

#     print("\n")
#     print("=" * 65)

#     print("PER-POSE EXTRINSIC CALIBRATION COMPLETE")

#     print("=" * 65)

#     print("Output:", OUTPUT_DIR)


def main():

    print("\n")
    print("=" * 65)
    print("CAMERA-PROJECTOR JOINT EXTRINSIC CALIBRATION")
    print("=" * 65)

    print("Dataset:", DATASET)

    data = load_data()
    poses = prepare_poses(data)

    # Initial estimate from existing independent PnP method.
    pose_results = calculate_pose_transformations(poses)

    initial_consensus = calculate_consensus(
        pose_results
    )

    print("\nINITIAL TRANSFORMATION")
    print(initial_consensus["R"])
    print(initial_consensus["T"])

    # Joint optimization.
    solution = optimize_joint_stereo(
        poses,
        initial_consensus["R"],
        initial_consensus["T"]
    )

    # Independent validation of the optimized result.
    validation = validate_joint_solution(
        poses,
        solution
    )

    R = solution["R"]
    T = solution["T"]

    T_cp, T_pc, projector_position = calculate_matrices(
        R, T
    )

    print("\nFINAL CAMERA-TO-PROJECTOR TRANSFORMATION")
    print("=" * 65)

    print("\nRotation:")
    print(R)

    print("\nTranslation (mm):")
    print(T)

    print("\nProjector optical centre in camera coordinates:")
    print(projector_position)

    print("\nSAVING RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_DIR / "camera_to_projector.npz",

        R_camera_to_projector=R,
        T_camera_to_projector_mm=T,

        T_camera_to_projector=T_cp,
        T_projector_to_camera=T_pc,

        projector_position_camera_mm=projector_position,

        camera_matrix=CAMERA_MATRIX,
        camera_distortion=CAMERA_DISTORTION,

        projector_matrix=PROJECTOR_MATRIX,
        projector_distortion=PROJECTOR_DISTORTION,

        optimization_parameters=solution["parameters"],

        xtion_mounting_correction_applied=False
    )

    report = {
        "method": "Joint nonlinear camera-projector optimization",
        "intrinsic_model": "D_simple",

        "camera_to_projector_rotation": R.tolist(),
        "camera_to_projector_translation_mm": T.tolist(),

        "camera_to_projector_matrix": T_cp.tolist(),
        "projector_to_camera_matrix": T_pc.tolist(),

        "projector_position_in_camera_mm":
            projector_position.tolist(),

        "number_of_poses": len(poses),

        "initial_rms_px": solution["initial_rms"],
        "final_rms_px": solution["final_rms"],

        "optimization_success":
            bool(solution["optimization"].success),

        "optimization_message":
            str(solution["optimization"].message),

        "validation": validation,

        "coordinate_convention": {
            "camera": "OpenCV X right, Y down, Z forward",
            "projector": "OpenCV X right, Y down, Z forward",
            "translation_units": "millimetres",
            "xtion_mounting_correction_applied": False
        }
    }

    with open(
        OUTPUT_DIR / "camera_to_projector.json",
        "w"
    ) as f:
        json.dump(report, f, indent=4)

    with open(
        OUTPUT_DIR / "stereo_validation.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=validation["poses"][0].keys()
        )

        writer.writeheader()
        writer.writerows(validation["poses"])

    print("\nSaved files:")
    print(OUTPUT_DIR / "camera_to_projector.json")
    print(OUTPUT_DIR / "camera_to_projector.npz")
    print(OUTPUT_DIR / "stereo_validation.csv")

    print("\n" + "=" * 65)
    print("JOINT CALIBRATION COMPLETE")
    print("=" * 65)

    print("Initial RMS:", solution["initial_rms"])
    print("Final RMS:", solution["final_rms"])
    print("Validation RMS:", validation["combined_rms_px"])

    print("Xtion mounting correction: NOT APPLIED")




if __name__ == "__main__":

    main()
