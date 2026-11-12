#!/usr/bin/env python3
"""
Export ProCamUnity calibration data to a Unity-readable JSON file.

Coordinate convention:
    OpenCV: X right, Y down, Z forward
    Unity:  X right, Y up,   Z forward

All translations are in millimetres.
Image coordinates and intrinsics are in pixels.

The camera-to-projector transformation is explicitly marked
PROVISIONAL until independently validated.
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "calibration_data"

OUTPUT_DEFAULT = (
    DATA / "unity_export" / "calibration.json"
)

# Best-performing projector model from diagnostics.
PROJECTOR_SIMPLE_DISTORTION = np.array(
    [0.17089973, 0.0, 0.0, 0.0, 0.0],
    dtype=float
)

# OpenCV -> Unity coordinate basis conversion.
CV_TO_UNITY = np.diag([1.0, -1.0, 1.0])


# ------------------------------------------------------------
# GENERAL HELPERS
# ------------------------------------------------------------

def load_npz(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        return {
            key: data[key].copy()
            for key in data.files
        }


def get_value(data, candidates, required=True):

    for key in candidates:

        if key in data:
            return data[key]

    if required:
        raise KeyError(
            f"None of these keys found: {candidates}\n"
            f"Available keys: {list(data.keys())}"
        )

    return None


def scalar(value):
    return np.asarray(value).item()


def matrix(value):
    return np.asarray(value, dtype=float).tolist()


def vector(value):
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def validate_matrix(name, value, shape):

    value = np.asarray(value, dtype=float)

    if value.shape != shape:
        raise ValueError(
            f"{name}: expected {shape}, got {value.shape}"
        )

    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or Inf")

    return value


def validate_rotation(name, R):

    R = validate_matrix(name, R, (3, 3))

    determinant = np.linalg.det(R)

    if not np.isclose(determinant, 1.0, atol=0.02):
        raise ValueError(
            f"{name} is not a valid rotation matrix. "
            f"det(R)={determinant}"
        )

    if not np.allclose(
        R.T @ R,
        np.eye(3),
        atol=0.02
    ):
        raise ValueError(
            f"{name} is not orthogonal."
        )

    return R


def make_transform(R, t):

    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)

    return T


def cv_transform_to_unity(R, t):

    """
    Convert a transform between two OpenCV coordinate
    systems into Unity coordinate systems.

    X_projector_cv = R * X_camera_cv + t

    X_projector_unity = S * R * S * X_camera_unity + S * t
    """

    S = CV_TO_UNITY

    R_unity = S @ R @ S
    t_unity = S @ np.asarray(t).reshape(3)

    return make_transform(R_unity, t_unity)


def intrinsic_data(path, name):

    data = load_npz(path)

    K = validate_matrix(
        f"{name} camera matrix",
        get_value(data, ["camera_matrix", "K"]),
        (3, 3)
    )

    distortion = vector(
        get_value(data, ["distortion", "dist_coeffs"])
    )

    width = int(scalar(
        get_value(data, ["image_width", "width"])
    ))

    height = int(scalar(
        get_value(data, ["image_height", "height"])
    ))

    return {
        "camera_matrix": matrix(K),
        "distortion": distortion,
        "resolution": {
            "width": width,
            "height": height
        }
    }


# ------------------------------------------------------------
# TRANSFORMATION LOADING
# ------------------------------------------------------------

def load_camera_projector_extrinsics():

    path = DATA / "extrinsics" / "camera_to_projector.npz"

    data = load_npz(path)

    R = get_value(
        data,
        [
            "rotation_camera_to_projector",
            "rotation",
            "R",
            "R_camera_to_projector"
        ]
    )

    t = get_value(
        data,
        [
            "translation_camera_to_projector",
            "translation",
            "t",
            "T_camera_to_projector",
            "tvec"
        ]
    )

    R = validate_rotation(
        "Camera-to-projector rotation", R
    )

    t = np.asarray(t, dtype=float).reshape(3)

    if not np.all(np.isfinite(t)):
        raise ValueError("Invalid camera-to-projector translation")

    return R, t


def load_rgbd_extrinsics():

    path = (
        DATA /
        "rgbd_extrinsics" /
        "rgbd_calibration.npz"
    )

    data = load_npz(path)

    R = validate_rotation(
        "Depth-to-RGB rotation",
        get_value(
            data,
            ["rotation_depth_to_rgb"]
        )
    )

    t = np.asarray(
        get_value(
            data,
            ["translation_depth_to_rgb_mm"]
        ),
        dtype=float
    ).reshape(3)

    scale = float(
        scalar(
            get_value(data, ["depth_scale"])
        )
    )

    return R, t, scale


# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

def export(output, projector_model):

    print("\nUNITY CALIBRATION EXPORT")
    print("=" * 60)

    # Intrinsics

    rgb = intrinsic_data(
        DATA / "rgb_intrinsics.npz",
        "RGB"
    )

    depth = intrinsic_data(
        DATA / "depth_intrinsics.npz",
        "Depth"
    )

    projector_npz = load_npz(
        DATA / "projector" / "projector_intrinsics.npz"
    )

    projector_K = validate_matrix(
        "Projector camera matrix",
        get_value(
            projector_npz,
            ["camera_matrix", "K"]
        ),
        (3, 3)
    )

    projector_width = int(scalar(
        get_value(
            projector_npz,
            ["image_width", "width"]
        )
    ))

    projector_height = int(scalar(
        get_value(
            projector_npz,
            ["image_height", "height"]
        )
    ))

    if projector_model == "D_simple":

        projector_distortion = (
            PROJECTOR_SIMPLE_DISTORTION.copy()
        )

    else:

        projector_distortion = np.asarray(
            get_value(
                projector_npz,
                ["distortion", "dist_coeffs"]
            ),
            dtype=float
        ).reshape(-1)

    projector = {
        "camera_matrix": matrix(projector_K),
        "distortion": vector(projector_distortion),
        "distortion_model": projector_model,
        "resolution": {
            "width": projector_width,
            "height": projector_height
        }
    }

    # Extrinsics

    R_pc, t_pc = load_camera_projector_extrinsics()

    R_rgbd, t_rgbd, depth_scale = load_rgbd_extrinsics()

    T_projector_from_rgb = cv_transform_to_unity(
        R_pc, t_pc
    )

    T_rgb_from_depth = cv_transform_to_unity(
        R_rgbd, t_rgbd
    )

    T_rgb_from_projector = np.linalg.inv(
        T_projector_from_rgb
    )

    T_depth_from_rgb = np.linalg.inv(
        T_rgb_from_depth
    )

    # Calibration quality

    extrinsics_status = "PROVISIONAL"

    calibration = {

        "schema_version": 1,

        "system": "ProCamUnity",

        "units": {
            "translation": "millimetres",
            "depth": "millimetres",
            "image_coordinates": "pixels"
        },

        "coordinate_convention": {
            "camera": "Unity: X right, Y up, Z forward",
            "projector": "Unity: X right, Y up, Z forward",
            "matrix_convention": (
                "4x4 row arrays; column-vector multiplication"
            ),
            "source_conversion": (
                "OpenCV to Unity using diag(1,-1,1) "
                "for both coordinate frames"
            )
        },

        "rgb_camera": rgb,

        "depth_camera": depth,

        "projector": projector,

        "depth": {
            "scale": depth_scale,
            "scale_applied": False,
            "note": (
                "Apply the depth scale when converting "
                "raw sensor values to millimetres."
            )
        },

        "extrinsics": {

            "status": extrinsics_status,

            "rgb_from_depth": matrix(
                T_rgb_from_depth
            ),

            "depth_from_rgb": matrix(
                T_depth_from_rgb
            ),

            "projector_from_rgb": matrix(
                T_projector_from_rgb
            ),

            "rgb_from_projector": matrix(
                T_rgb_from_projector
            ),

            "camera_to_projector_source": {
                "rotation_opencv": matrix(R_pc),
                "translation_opencv_mm": vector(t_pc)
            },

            "depth_to_rgb_source": {
                "rotation_opencv": matrix(R_rgbd),
                "translation_opencv_mm": vector(t_rgbd)
            }
        },

        "quality": {

            "rgb_intrinsic_rms_px": 0.36416646,

            "projector_intrinsic_rms_px": 0.49097510,

            "rgbd_extrinsic_rms_mm": 4.18838749,

            "projector_extrinsic_validation_rms_px": 11.70417,

            "projector_extrinsic_validated": False,

            "notes": [
                "Camera-to-projector transformation is provisional.",
                "Do not interpret provisional extrinsics as "
                "a validated physical calibration.",
                "Projector distortion model must match "
                "the model used to generate projector coordinates."
            ]
        }
    }

    # Final consistency checks

    for name, transform in [
        ("RGB from depth", T_rgb_from_depth),
        ("Projector from RGB", T_projector_from_rgb)
    ]:

        if not np.all(np.isfinite(transform)):
            raise ValueError(
                f"Invalid transformation: {name}"
            )

        if not np.allclose(
            transform[3],
            [0, 0, 0, 1],
            atol=1e-8
        ):
            raise ValueError(
                f"Invalid homogeneous matrix: {name}"
            )

    # Save

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output, "w") as f:

        json.dump(
            calibration,
            f,
            indent=4,
            allow_nan=False
        )

    print("\nLOADED CALIBRATION")
    print("-" * 60)

    print(
        f"RGB camera: "
        f"{rgb['resolution']['width']} x "
        f"{rgb['resolution']['height']}"
    )

    print(
        f"Depth camera: "
        f"{depth['resolution']['width']} x "
        f"{depth['resolution']['height']}"
    )

    print(
        f"Projector: "
        f"{projector_width} x {projector_height}"
    )

    print(
        f"Depth scale: {depth_scale}"
    )

    print("\nEXTRINSICS")
    print("-" * 60)

    print("Projector from RGB:")
    print(np.array(T_projector_from_rgb))

    print("\nRGB from depth:")
    print(np.array(T_rgb_from_depth))

    print("\nCALIBRATION STATUS")
    print("-" * 60)

    print("Camera-to-projector: PROVISIONAL")
    print("Projector distortion:", projector_model)

    print("\nEXPORT COMPLETE")
    print("=" * 60)
    print(f"Output: {output}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT
    )

    parser.add_argument(
        "--projector-model",
        choices=["D_simple", "A_full"],
        default="D_simple"
    )

    args = parser.parse_args()

    export(
        args.output,
        args.projector_model
    )


if __name__ == "__main__":
    main()
