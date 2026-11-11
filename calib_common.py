"""Shared geometry and calibration helpers.

Coordinate conventions
----------------------
* Camera coordinates use the OpenCV convention: X right, Y down, Z forward.
* Board coordinates place the checkerboard in the Z=0 plane.  The origin is
  the first internal corner returned by OpenCV.  X runs along the
  ``BOARD_COLUMNS`` direction and Y along the ``BOARD_ROWS`` direction.
* Planes are stored as ``n . X + d = 0``.  ``n`` is always oriented toward
  the camera, so ``d`` is positive.
* The RGB-D transform maps depth-camera points into RGB-camera points::

      X_rgb = R @ X_depth + t

  ``R`` is ``rotation_depth_to_rgb`` and ``t`` is
  ``translation_depth_to_rgb_mm``.

Checkerboard geometry
---------------------
An 8x5 internal-corner board has 9x6 squares.  Relative to the first
internal corner, the printed pattern extends one full square (29 mm) beyond
the outermost internal corners.  The pattern therefore spans::

    x in [-29, 232] mm
    y in [-29, 145] mm

The paper and wooden borders are deliberately *not* part of this geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------

BOARD_COLUMNS = 8
BOARD_ROWS = 5
SQUARE_SIZE_MM = 29.0

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = BASE_DIR / "calibration_data"
RGBD_OUTPUT_DIR = CALIBRATION_DIR / "rgbd_extrinsics"

RGB_INTRINSICS_PATH = CALIBRATION_DIR / "rgb_intrinsics.npz"
DEPTH_INTRINSICS_PATH = CALIBRATION_DIR / "depth_intrinsics.npz"

#: Default valid depth range for the Asus Xtion Pro Live.
DEFAULT_MIN_DEPTH_MM = 300.0
DEFAULT_MAX_DEPTH_MM = 5000.0


def create_object_points(
    columns: int = BOARD_COLUMNS,
    rows: int = BOARD_ROWS,
    square_size_mm: float = SQUARE_SIZE_MM,
) -> np.ndarray:
    """Return the internal checkerboard corners in board coordinates."""

    points = np.zeros((columns * rows, 3), dtype=np.float32)

    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    points *= float(square_size_mm)

    return points


def pattern_outer_corners_mm(
    columns: int = BOARD_COLUMNS,
    rows: int = BOARD_ROWS,
    square_size_mm: float = SQUARE_SIZE_MM,
    extra_margin_mm: float = 0.0,
) -> np.ndarray:
    """Return the four outer corners of the printed checkerboard pattern.

    The pattern has one full square outside the outermost internal corners.
    ``extra_margin_mm`` expands the rectangle, which is useful when building
    a slightly generous region of interest.
    """

    margin = float(square_size_mm) + float(extra_margin_mm)

    max_x = (columns - 1) * float(square_size_mm) + margin
    max_y = (rows - 1) * float(square_size_mm) + margin

    return np.array(
        [
            [-margin, -margin, 0.0],
            [max_x, -margin, 0.0],
            [max_x, max_y, 0.0],
            [-margin, max_y, 0.0],
        ],
        dtype=np.float64,
    )


def pattern_size_mm(
    columns: int = BOARD_COLUMNS,
    rows: int = BOARD_ROWS,
    square_size_mm: float = SQUARE_SIZE_MM,
) -> tuple[float, float]:
    """Return the width and height of the printed pattern in millimetres."""

    return (
        (columns + 1) * float(square_size_mm),
        (rows + 1) * float(square_size_mm),
    )


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


@dataclass
class CameraIntrinsics:
    """Camera matrix, distortion coefficients, and image size."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(
            self.camera_matrix, dtype=np.float64
        ).reshape(3, 3)

        self.distortion = np.asarray(
            self.distortion, dtype=np.float64
        ).reshape(-1)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


def load_intrinsics(
    path: str | Path,
    default_width: int = 0,
    default_height: int = 0,
) -> CameraIntrinsics:
    """Load intrinsics from one of the project's ``.npz`` files."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")

    with np.load(path) as data:
        files = set(data.files)

        if "camera_matrix" not in files:
            raise KeyError(f"No camera_matrix in {path}")

        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)

        if "distortion" in files:
            distortion = np.asarray(data["distortion"], dtype=np.float64)
        elif "distortion_coefficients" in files:
            distortion = np.asarray(
                data["distortion_coefficients"], dtype=np.float64
            )
        else:
            distortion = np.zeros(5, dtype=np.float64)

        if "image_width" in files:
            width = int(data["image_width"])
        elif "width" in files:
            width = int(data["width"])
        else:
            width = int(default_width)

        if "image_height" in files:
            height = int(data["image_height"])
        elif "height" in files:
            height = int(data["height"])
        else:
            height = int(default_height)

    return CameraIntrinsics(camera_matrix, distortion, width, height)


def load_rgb_intrinsics(
    path: str | Path = RGB_INTRINSICS_PATH,
) -> CameraIntrinsics:
    return load_intrinsics(path, 1280, 1024)


def load_depth_intrinsics(
    path: str | Path = DEPTH_INTRINSICS_PATH,
) -> CameraIntrinsics:
    return load_intrinsics(path, 640, 480)


# ---------------------------------------------------------------------------
# Checkerboard detection and pose
# ---------------------------------------------------------------------------


def detect_checkerboard(
    image: np.ndarray,
    columns: int = BOARD_COLUMNS,
    rows: int = BOARD_ROWS,
    refine: bool = True,
    fast: bool = False,
) -> tuple[bool, Optional[np.ndarray], np.ndarray]:
    """Detect an internal-corner checkerboard.

    ``image`` may be grayscale or BGR.  The returned corners have shape
    ``(N, 1, 2)`` and are ordered row by row, matching
    :func:`create_object_points`.

    ``fast=True`` uses the classic detector, which is much faster and is
    intended for the live preview.  The capture path should use the slower,
    more accurate sector-based detector (``fast=False``).
    """

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    pattern_size = (int(columns), int(rows))

    if fast:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size, flags=flags
        )
    else:
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=flags
        )

        if not found:
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            found, corners = cv2.findChessboardCorners(
                gray, pattern_size, flags=flags
            )

    if found and refine:
        window = (11, 11) if fast else (5, 5)

        corners = cv2.cornerSubPix(
            gray,
            corners,
            window,
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_MAX_ITER,
                30,
                1e-3,
            ),
        )

    if not found:
        return False, None, gray

    return True, corners.reshape(-1, 1, 2).astype(np.float32), gray


@dataclass
class BoardPose:
    """Pose of the checkerboard in RGB-camera coordinates."""

    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    normal: np.ndarray
    offset: float
    corners: np.ndarray
    reprojection_rms_px: float

    @property
    def distance_mm(self) -> float:
        return float(np.linalg.norm(self.translation))

    def transform_to_rgb(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.rotation, self.translation)


def solve_board_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: Optional[np.ndarray] = None,
    object_points: Optional[np.ndarray] = None,
    refine: bool = True,
) -> Optional[BoardPose]:
    """Solve the board pose from detected internal corners."""

    if object_points is None:
        object_points = create_object_points()

    if distortion is None:
        distortion = np.zeros(5, dtype=np.float64)

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)

    image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not ok:
        return None

    if refine:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)

    normal, offset = plane_from_pose(rotation, translation)

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )

    errors = np.linalg.norm(
        projected.reshape(-1, 2) - image_points.reshape(-1, 2),
        axis=1,
    )

    rms = float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan")

    return BoardPose(
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
        rotation=rotation,
        translation=translation,
        normal=normal,
        offset=offset,
        corners=image_points,
        reprojection_rms_px=rms,
    )


# ---------------------------------------------------------------------------
# Plane geometry
# ---------------------------------------------------------------------------


def canonicalize_plane(
    normal: np.ndarray, offset: float
) -> tuple[np.ndarray, float]:
    """Normalize a plane and orient its normal toward the camera."""

    normal = np.asarray(normal, dtype=np.float64).copy()

    length = float(np.linalg.norm(normal))

    if length < 1e-12:
        raise ValueError("Plane normal has zero length")

    normal /= length
    offset = float(offset)

    if offset < 0.0:
        normal = -normal
        offset = -offset

    return normal, offset


def plane_from_pose(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the board plane from a board pose."""

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)

    normal = rotation[:, 2].copy()
    offset = -float(np.dot(normal, translation))

    return canonicalize_plane(normal, offset)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares plane fit using SVD.

    Returns ``normal, offset, signed_residuals`` where
    ``residual = points @ normal + offset``.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    if len(points) < 3:
        raise ValueError("At least three points are required to fit a plane")

    center = np.mean(points, axis=0)

    _, _, vh = np.linalg.svd(points - center, full_matrices=False)

    normal = vh[-1]
    offset = -float(np.dot(normal, center))

    normal, offset = canonicalize_plane(normal, offset)

    residuals = points @ normal + offset

    return normal, offset, residuals


def plane_angle_deg(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    """Angle between two plane normals in degrees (sign independent)."""

    a = np.asarray(normal_a, dtype=np.float64).reshape(3)
    b = np.asarray(normal_b, dtype=np.float64).reshape(3)

    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)

    cosine = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


# ---------------------------------------------------------------------------
# Transforms and projection
# ---------------------------------------------------------------------------


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply ``X_out = R @ X_in + t`` to an array of 3-D points."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)

    return points @ rotation.T + translation


def invert_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the inverse of ``X_out = R @ X_in + t``."""

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)

    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation

    return rotation_inv, translation_inv


def project_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: Optional[np.ndarray] = None,
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project 3-D points into a camera image."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    if distortion is None:
        distortion = np.zeros(5, dtype=np.float64)

    if rvec is None:
        rvec = np.zeros(3, dtype=np.float64)

    if tvec is None:
        tvec = np.zeros(3, dtype=np.float64)

    projected, _ = cv2.projectPoints(
        points,
        np.asarray(rvec, dtype=np.float64).reshape(3),
        np.asarray(tvec, dtype=np.float64).reshape(3),
        camera_matrix,
        np.asarray(distortion, dtype=np.float64).reshape(-1),
    )

    return projected.reshape(-1, 2)


def depth_to_points(
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    min_depth_mm: float = DEFAULT_MIN_DEPTH_MM,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
    stride: int = 1,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a depth image to 3-D points in depth-camera coordinates.

    Returns ``points`` (N, 3) and ``pixels`` (N, 2) where pixels are
    ``(u, v)`` coordinates in the depth image.
    """

    depth = np.asarray(depth)

    if depth.ndim != 2:
        raise ValueError("Depth image must be two-dimensional")

    stride = int(stride)

    if stride < 1:
        raise ValueError("stride must be at least 1")

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    if fx == 0.0 or fy == 0.0:
        raise ValueError("Invalid depth focal length")

    v_grid, u_grid = np.mgrid[
        0 : depth.shape[0] : stride,
        0 : depth.shape[1] : stride,
    ]

    z = depth[::stride, ::stride].astype(np.float64)

    valid = (z >= float(min_depth_mm)) & (z <= float(max_depth_mm))

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != depth.shape:
            raise ValueError("mask must have the same shape as depth")

        valid &= mask[::stride, ::stride]

    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.column_stack([x, y, z])
    pixels = np.column_stack([u, v])

    return points, pixels


def board_polygon_in_depth(
    pose: BoardPose,
    depth_camera_matrix: np.ndarray,
    rotation_depth_to_rgb: Optional[np.ndarray] = None,
    translation_depth_to_rgb_mm: Optional[np.ndarray] = None,
    extra_margin_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the printed board pattern into the depth image.

    If a prior RGB-D transform is supplied, it is used to move the board
    corners from RGB coordinates into depth coordinates.  Otherwise the
    cameras are assumed to be co-located, which is accurate enough for a
    region of interest because the two cameras are physically close.
    """

    corners_board = pattern_outer_corners_mm(
        extra_margin_mm=extra_margin_mm
    )

    corners_rgb = pose.transform_to_rgb(corners_board)

    if (
        rotation_depth_to_rgb is not None
        and translation_depth_to_rgb_mm is not None
    ):
        rotation_inv, translation_inv = invert_transform(
            rotation_depth_to_rgb,
            translation_depth_to_rgb_mm,
        )

        corners_depth = transform_points(
            corners_rgb, rotation_inv, translation_inv
        )
    else:
        corners_depth = corners_rgb

    polygon = project_points(
        corners_depth,
        depth_camera_matrix,
        distortion=None,
    )

    return polygon, corners_depth


def polygon_mask(
    image_shape: Sequence[int], polygon: np.ndarray
) -> np.ndarray:
    """Rasterize a polygon into a boolean mask."""

    mask = np.zeros((int(image_shape[0]), int(image_shape[1])), np.uint8)

    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)

    cv2.fillPoly(
        mask,
        [np.round(points).astype(np.int32)],
        255,
        lineType=cv2.LINE_AA,
    )

    return mask > 0


def select_plane(
    points: np.ndarray,
    predicted_normal: Optional[np.ndarray] = None,
    predicted_offset: Optional[float] = None,
    normal_tolerance_deg: float = 35.0,
    distance_threshold_mm: float = 8.0,
    seed_threshold_mm: float = 60.0,
    max_iterations: int = 1200,
    min_inliers: int = 50,
    random_seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Find the board plane in a set of depth points.

    The search is normal-guided.  If a predicted plane is supplied, points
    close to it are used as a seed and refined iteratively.  A normal-
    constrained RANSAC fallback handles a poor or missing prediction.

    Returns ``normal, offset, residuals, inlier_mask``.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    if len(points) < 3:
        raise ValueError("At least three points are required")

    if len(points) < min_inliers:
        raise RuntimeError(
            f"Only {len(points)} points available, need {min_inliers}"
        )

    rng = np.random.default_rng(int(random_seed))

    predicted = None

    if predicted_normal is not None:
        predicted = np.asarray(predicted_normal, dtype=np.float64).reshape(3)

        length = float(np.linalg.norm(predicted))

        if length > 1e-12:
            predicted = predicted / length
        else:
            predicted = None

    # ------------------------------------------------------------------
    # Fast path: seed from the predicted plane, then shrink the threshold.
    # ------------------------------------------------------------------

    if predicted is not None and predicted_offset is not None:
        seed = (
            np.abs(points @ predicted + float(predicted_offset))
            < float(seed_threshold_mm)
        )

        if int(np.count_nonzero(seed)) >= min_inliers:
            normal, offset, _ = fit_plane(points[seed])

            if plane_angle_deg(normal, predicted) <= normal_tolerance_deg:
                inliers = seed

                for threshold in (
                    seed_threshold_mm,
                    30.0,
                    15.0,
                    distance_threshold_mm,
                ):
                    residuals = points @ normal + offset
                    inliers = np.abs(residuals) < float(threshold)

                    if int(np.count_nonzero(inliers)) < min_inliers:
                        break

                    normal, offset, _ = fit_plane(points[inliers])

                residuals = points @ normal + offset
                inliers = np.abs(residuals) < float(distance_threshold_mm)

                if (
                    int(np.count_nonzero(inliers)) >= min_inliers
                    and plane_angle_deg(normal, predicted)
                    <= normal_tolerance_deg
                ):
                    normal, offset, _ = fit_plane(points[inliers])

                    residuals = points @ normal + offset

                    return normal, offset, residuals, inliers

    # ------------------------------------------------------------------
    # RANSAC fallback.
    # ------------------------------------------------------------------

    best_count = 0
    best_inliers: Optional[np.ndarray] = None

    for _ in range(int(max_iterations)):
        indices = rng.choice(len(points), size=3, replace=False)

        p0, p1, p2 = points[indices]

        normal = np.cross(p1 - p0, p2 - p0)

        length = float(np.linalg.norm(normal))

        if length < 1e-9:
            continue

        normal /= length
        offset = -float(np.dot(normal, p0))
        normal, offset = canonicalize_plane(normal, offset)

        if (
            predicted is not None
            and plane_angle_deg(normal, predicted)
            > normal_tolerance_deg
        ):
            continue

        residuals = np.abs(points @ normal + offset)
        count = int(np.count_nonzero(residuals < distance_threshold_mm))

        if count > best_count:
            best_count = count
            best_inliers = residuals < distance_threshold_mm

    if best_inliers is None or best_count < min_inliers:
        raise RuntimeError(
            "Could not find a dominant plane in the selected depth points"
        )

    normal, offset, _ = fit_plane(points[best_inliers])

    for threshold in (
        distance_threshold_mm * 2.0,
        distance_threshold_mm,
        distance_threshold_mm,
    ):
        residuals = points @ normal + offset
        inliers = np.abs(residuals) < float(threshold)

        if int(np.count_nonzero(inliers)) < min_inliers:
            break

        normal, offset, _ = fit_plane(points[inliers])

    residuals = points @ normal + offset
    inliers = np.abs(residuals) < float(distance_threshold_mm)

    if int(np.count_nonzero(inliers)) < min_inliers:
        raise RuntimeError("Plane refinement left too few inliers")

    normal, offset, _ = fit_plane(points[inliers])

    residuals = points @ normal + offset

    return normal, offset, residuals, inliers


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def depth_visualization(
    depth: np.ndarray,
    min_depth_mm: float = DEFAULT_MIN_DEPTH_MM,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
) -> np.ndarray:
    """Return a colour-mapped depth image."""

    depth = np.asarray(depth)

    valid = (depth >= float(min_depth_mm)) & (depth <= float(max_depth_mm))

    output = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    if not np.any(valid):
        return output

    near = float(min_depth_mm)
    far = float(max_depth_mm)

    normalized = np.zeros(depth.shape, dtype=np.uint8)

    normalized[valid] = np.clip(
        (depth[valid] - near) / max(far - near, 1.0) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    return cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)


def save_json(path: str | Path, data: object) -> None:
    """Write JSON with a stable, human-readable layout."""

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, sort_keys=False)


def load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
