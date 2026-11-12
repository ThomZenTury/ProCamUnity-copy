#!/usr/bin/env python3
"""
ProCamUnity calibration exporter.

Input:
    calibration_data/rgb_intrinsics.npz
    calibration_data/depth_intrinsics.npz
    calibration_data/rgbd_extrinsics/rgbd_calibration.npz
    calibration_data/projector/projector_intrinsics.npz
    calibration_data/extrinsics/camera_to_projector.npz
    calibration_data/extrinsics/camera_to_projector.json

Output:
    calibration_data/unity_export/calibration.json

Coordinate conventions:
    OpenCV: X right, Y down, Z forward
    Unity:  X right, Y up, Z forward

Translations: millimetres.
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "calibration_data"

RGB_INTRINSICS = DATA / "rgb_intrinsics.npz"
DEPTH_INTRINSICS = DATA / "depth_intrinsics.npz"

RGBD_EXTRINSICS = (
    DATA / "rgbd_extrinsics" / "rgbd_calibration.npz"
)

PROJECTOR_EXTRINSICS_DIR = DATA / "extrinsics"

PROJECTOR_EXTRINSICS = (
    PROJECTOR_EXTRINSICS_DIR / "camera_to_projector.npz"
)

PROJECTOR_EXTRINSICS_JSON = (
    PROJECTOR_EXTRINSICS_DIR / "camera_to_projector.json"
)

OUTPUT_DEFAULT = (
    DATA / "unity_export" / "calibration.json"
)


# ============================================================
# PROJECTOR INTRINSICS
# ============================================================

# Selected D_simple model.
# These values must match the model used during
# camera-projector extrinsic calibration.

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

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

PROJECTOR_MODEL = "D_simple"


# ============================================================
# COORDINATE CONVERSION
# ============================================================

CV_TO_UNITY = np.diag([1.0, -1.0, 1.0])


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_npz(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n{path}"
        )

    with np.load(path, allow_pickle=False) as data:
        return {
            key: data[key].copy()
            for key in data.files
        }


def get_value(data, keys, required=True):

    for key in keys:
        if key in data:
            return data[key]

    if required:
        raise KeyError(
            f"None of these keys found: {keys}\n"
            f"Available keys: {list(data.keys())}"
        )

    return None


def to_float(value):
    return np.asarray(value, dtype=np.float64)


def to_scalar(value):
    return np.asarray(value).item()


def validate_matrix(name, value, shape):

    value = to_float(value)

    if value.shape != shape:
        raise ValueError(
            f"{name}: expected {shape}, got {value.shape}"
        )

    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or Inf")

    return value


def validate_rotation(name, R):

    R = validate_matrix(name, R, (3, 3))

    if not np.isclose(np.linalg.det(R), 1.0, atol=0.02):
        raise ValueError(
            f"{name}: invalid determinant "
            f"{np.linalg.det(R)}"
        )

    if not np.allclose(R.T @ R, np.eye(3), atol=0.02):
        raise ValueError(
            f"{name}: matrix is not orthogonal"
        )

    return R


def validate_translation(name, t):

    t = to_float(t).reshape(-1)

    if t.size != 3:
        raise ValueError(
            f"{name}: expected 3 values, got {t.size}"
        )

    if not np.all(np.isfinite(t)):
        raise ValueError(f"{name} contains NaN or Inf")

    return t


def make_transform(R, t):

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def validate_transform(name, T):

    T = validate_matrix(name, T, (4, 4))

    if not np.allclose(
        T[3],
        [0, 0, 0, 1],
        atol=1e-7
    ):
        raise ValueError(
            f"{name}: invalid homogeneous last row"
        )

    validate_rotation(
        f"{name} rotation",
        T[:3, :3]
    )

    return T


def transform_to_unity(R, t):

    S = CV_TO_UNITY

    R_unity = S @ R @ S
    t_unity = S @ t

    return make_transform(R_unity, t_unity)


# ============================================================
# INTRINSICS
# ============================================================

def load_intrinsics(path, name):

    data = load_npz(path)

    K = validate_matrix(
        f"{name} camera matrix",
        get_value(data, ["camera_matrix", "K"]),
        (3, 3)
    )

    distortion_value = get_value(
        data,
        [
            "distortion",
            "dist_coeffs",
            "distortion_coefficients"
        ],
        required=False
    )

    if distortion_value is None:
        distortion = np.zeros(5, dtype=np.float64)
        distortion_status = "NOT_CALIBRATED"
    else:
        distortion = to_float(
            distortion_value
        ).reshape(-1)
        distortion_status = "CALIBRATED"

    width = int(to_scalar(
        get_value(data, ["image_width", "width"])
    ))

    height = int(to_scalar(
        get_value(data, ["image_height", "height"])
    ))

    return {
        "camera_matrix": K.tolist(),
        "distortion": distortion.tolist(),
        "distortion_status": distortion_status,
        "resolution": {
            "width": width,
            "height": height
        }
    }


def load_projector_intrinsics():

    return {
        "camera_matrix": PROJECTOR_MATRIX.tolist(),
        "distortion": PROJECTOR_DISTORTION.tolist(),
        "distortion_model": PROJECTOR_MODEL,
        "resolution": {
            "width": PROJECTOR_WIDTH,
            "height": PROJECTOR_HEIGHT
        }
    }


# ============================================================
# CAMERA-PROJECTOR EXTRINSICS
# ============================================================

def load_camera_projector_extrinsics():

    print("\nLoading camera-projector extrinsics...")

    data = load_npz(PROJECTOR_EXTRINSICS)

    print("Available NPZ keys:")

    for key, value in data.items():
        print(f"  {key:40s} {value.shape}")

    # Current calibration script saves:
    #
    # R_camera_to_projector       (3,3)
    # T_camera_to_projector_mm    (3,)
    # T_camera_to_projector       (4,4)
    # T_projector_to_camera       (4,4)

    T_value = get_value(
        data,
        [
            "T_camera_to_projector",
            "camera_to_projector_matrix"
        ],
        required=False
    )

    if T_value is not None:

        T = validate_transform(
            "Camera-to-projector transformation",
            T_value
        )

        R = T[:3, :3].copy()
        t = T[:3, 3].copy()

    else:

        R = validate_rotation(
            "Camera-to-projector rotation",
            get_value(
                data,
                [
                    "R_camera_to_projector",
                    "rotation_camera_to_projector",
                    "rotation",
                    "R"
                ]
            )
        )

        t = validate_translation(
            "Camera-to-projector translation",
            get_value(
                data,
                [
                    "T_camera_to_projector_mm",
                    "translation_camera_to_projector_mm",
                    "translation_camera_to_projector",
                    "translation",
                    "t",
                    "tvec"
                ]
            )
        )

        T = make_transform(R, t)

    R_saved = get_value(
        data,
        ["R_camera_to_projector"],
        required=False
    )

    t_saved = get_value(
        data,
        ["T_camera_to_projector_mm"],
        required=False
    )

    if R_saved is not None:
        if not np.allclose(R, R_saved, atol=1e-7):
            raise ValueError(
                "Rotation and transformation matrices disagree."
            )

    if t_saved is not None:
        if not np.allclose(
            t,
            np.asarray(t_saved).reshape(3),
            atol=1e-7
        ):
            raise ValueError(
                "Translation and transformation matrices disagree."
            )

    print("\nCamera-to-projector rotation:")
    print(R)

    print("\nCamera-to-projector translation (mm):")
    print(t)

    return R, t


# ============================================================
# RGB-D EXTRINSICS
# ============================================================

def load_rgbd_extrinsics():

    print("\nLoading RGB-D extrinsics...")

    data = load_npz(RGBD_EXTRINSICS)

    R = validate_rotation(
        "Depth-to-RGB rotation",
        get_value(
            data,
            ["rotation_depth_to_rgb"]
        )
    )

    t = validate_translation(
        "Depth-to-RGB translation",
        get_value(
            data,
            ["translation_depth_to_rgb_mm"]
        )
    )

    depth_scale = float(to_scalar(
        get_value(data, ["depth_scale"])
    ))

    return R, t, depth_scale


# ============================================================
# QUALITY INFORMATION
# ============================================================

def load_quality_report():

    if not PROJECTOR_EXTRINSICS_JSON.exists():
        return {
            "status": "PROVISIONAL",
            "note": "Extrinsics JSON report unavailable."
        }

    with open(
        PROJECTOR_EXTRINSICS_JSON,
        "r",
        encoding="utf-8"
    ) as f:
        report = json.load(f)

    return {
        "status": "PROVISIONAL",
        "method": report.get("method", "Unknown"),
        "intrinsic_model": report.get(
            "intrinsic_model",
            PROJECTOR_MODEL
        ),
        "initial_rms_px": report.get("initial_rms_px"),
        "final_rms_px": report.get("final_rms_px"),
        "validation": report.get("validation", {}),
        "number_of_poses": report.get("number_of_poses"),
        "notes": [
            "Camera-projector extrinsics are provisional.",
            "Independent physical validation is required.",
            "Xtion mounting correction has not been applied."
        ]
    }


# ============================================================
# EXPORT
# ============================================================

def export(output):

    print("\n" + "=" * 65)
    print("UNITY CALIBRATION EXPORT")
    print("=" * 65)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Calibration data: {DATA}")

    # Intrinsics

    print("\nLoading intrinsic calibrations...")

    rgb = load_intrinsics(
        RGB_INTRINSICS,
        "RGB"
    )

    depth = load_intrinsics(
        DEPTH_INTRINSICS,
        "Depth"
    )

    projector = load_projector_intrinsics()

    # Extrinsics

    R_pc, t_pc = load_camera_projector_extrinsics()

    R_rgbd, t_rgbd, depth_scale = load_rgbd_extrinsics()

    # Convert transformations to Unity coordinates.

    T_projector_from_rgb = transform_to_unity(
        R_pc,
        t_pc
    )

    T_rgb_from_depth = transform_to_unity(
        R_rgbd,
        t_rgbd
    )

    T_rgb_from_projector = np.linalg.inv(
        T_projector_from_rgb
    )

    T_depth_from_rgb = np.linalg.inv(
        T_rgb_from_depth
    )

    # Validate transformations.

    transformations = {
        "projector_from_rgb": T_projector_from_rgb,
        "rgb_from_projector": T_rgb_from_projector,
        "rgb_from_depth": T_rgb_from_depth,
        "depth_from_rgb": T_depth_from_rgb
    }

    print("\nTransformation checks:")
    print("-" * 65)

    for name, T in transformations.items():
        validate_transform(name, T)
        print(f"{name:30s}: OK")

    # Verify inverse consistency.

    if not np.allclose(
        T_projector_from_rgb @ T_rgb_from_projector,
        np.eye(4),
        atol=1e-7
    ):
        raise ValueError(
            "RGB/projector inverse check failed."
        )

    if not np.allclose(
        T_rgb_from_depth @ T_depth_from_rgb,
        np.eye(4),
        atol=1e-7
    ):
        raise ValueError(
            "RGB/depth inverse check failed."
        )

    # Assemble output.

    calibration = {

        "schema_version": 1,

        "system": "ProCamUnity",

        "units": {
            "translation": "millimetres",
            "depth": "millimetres",
            "image_coordinates": "pixels"
        },

        "coordinate_convention": {

            "opencv": "X right, Y down, Z forward",

            "unity": "X right, Y up, Z forward",

            "matrix_convention": (
                "4x4 matrices, row-major JSON arrays, "
                "column-vector multiplication"
            )
        },

        "rgb_camera": rgb,

        "depth_camera": depth,

        "projector": projector,

        "depth": {
            "scale": depth_scale,
            "scale_applied": False
        },

        "extrinsics": {

            "status": "PROVISIONAL",

            "projector_from_rgb": (
                T_projector_from_rgb.tolist()
            ),

            "rgb_from_projector": (
                T_rgb_from_projector.tolist()
            ),

            "rgb_from_depth": (
                T_rgb_from_depth.tolist()
            ),

            "depth_from_rgb": (
                T_depth_from_rgb.tolist()
            ),

            "camera_to_projector_source_opencv": {
                "rotation": R_pc.tolist(),
                "translation_mm": t_pc.tolist()
            },

            "depth_to_rgb_source_opencv": {
                "rotation": R_rgbd.tolist(),
                "translation_mm": t_rgbd.tolist()
            }
        },

        "quality": {

            "rgb_intrinsic_rms_px": 0.36416646,

            "rgbd_extrinsic_rms_mm": 4.18838749,

            "projector_intrinsic_model": PROJECTOR_MODEL,

            "projector_intrinsic_rms_px": 0.3720,

            "projector_extrinsics": load_quality_report()
        }
    }

    # Save JSON.

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            calibration,
            f,
            indent=4,
            allow_nan=False
        )

    # Summary.

    print("\n" + "=" * 65)
    print("UNITY EXPORT COMPLETE")
    print("=" * 65)

    print(
        f"RGB: {rgb['resolution']['width']} x "
        f"{rgb['resolution']['height']}"
    )

    print(
        f"Depth: {depth['resolution']['width']} x "
        f"{depth['resolution']['height']}"
    )

    print(
        f"Projector: {PROJECTOR_WIDTH} x "
        f"{PROJECTOR_HEIGHT}"
    )

    print(f"Projector model: {PROJECTOR_MODEL}")

    print(
        f"Depth distortion: {depth['distortion_status']}"
    )

    print("Extrinsics status: PROVISIONAL")

    print(f"\nOutput:\n{output}")

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Export ProCamUnity calibration."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT
    )

    args = parser.parse_args()

    export(args.output)


if __name__ == "__main__":
    main()
