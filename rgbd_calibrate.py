"""Reliable RGB-D extrinsic calibration for the Xtion Pro Live.

The workflow avoids manual polygon placement.  For every checkerboard pose:

1. The board is detected in the RGB image and its pose is solved with PnP.
2. The known printed-pattern rectangle is projected into the depth image.
3. The board plane is found automatically in the depth points using the RGB
   plane as a normal/offset guide, with a normal-constrained RANSAC fallback.
4. The RGB-D transform is refined with a multi-start point-to-plane
   optimisation.

Commands
--------
``python rgbd_calibrate.py capture``
    Capture observations interactively and save them.

``python rgbd_calibrate.py optimize``
    Optimise using the saved observations.

``python rgbd_calibrate.py all``
    Capture and optimise (default).

``python rgbd_calibrate.py validate``
    Re-evaluate the saved calibration and observations.

``python rgbd_calibrate.py selftest``
    Run a synthetic end-to-end test without a camera.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calib_common import (
    BOARD_COLUMNS,
    BOARD_ROWS,
    CALIBRATION_DIR,
    DEFAULT_MAX_DEPTH_MM,
    DEFAULT_MIN_DEPTH_MM,
    RGBD_OUTPUT_DIR,
    SQUARE_SIZE_MM,
    BoardPose,
    CameraIntrinsics,
    board_polygon_in_depth,
    create_object_points,
    depth_to_points,
    depth_visualization,
    detect_checkerboard,
    fit_plane,
    invert_transform,
    load_depth_intrinsics,
    load_rgb_intrinsics,
    pattern_outer_corners_mm,
    plane_angle_deg,
    polygon_mask,
    project_points,
    save_json,
    select_plane,
    solve_board_pose,
    transform_points,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OBSERVATIONS_PATH = RGBD_OUTPUT_DIR / "observations_rgbd.npz"
CALIBRATION_PATH = RGBD_OUTPUT_DIR / "rgbd_calibration.npz"
REPORT_PATH = RGBD_OUTPUT_DIR / "rgbd_calibration_report.txt"
CAPTURES_DIR = RGBD_OUTPUT_DIR / "captures"

TARGET_POSES = 20
MIN_POSES = 8

MAX_POINTS_PER_POSE = 2500

CAPTURE_MIN_DISTANCE_MM = 350.0
CAPTURE_MAX_DISTANCE_MM = 3000.0
CAPTURE_RECOMMENDED_MIN_MM = 500.0
CAPTURE_RECOMMENDED_MAX_MM = 1200.0

CAPTURE_MIN_INLIERS = 250
CAPTURE_MAX_PLANE_RMS_MM = 6.0
CAPTURE_MAX_NORMAL_ANGLE_DEG = 30.0
CAPTURE_MAX_REPROJECTION_RMS_PX = 1.0
CAPTURE_MIN_PATTERN_AREA_PX = 2000.0

PREVIEW_MIN_INLIERS = 60
PREVIEW_MAX_PLANE_RMS_MM = 12.0
PREVIEW_MAX_NORMAL_ANGLE_DEG = 40.0

OPTIMIZATION_LOSS_SCALE_MM = 5.0
OPTIMIZATION_MAX_NFEV = 2500


# ---------------------------------------------------------------------------
# Observation data
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """One RGB-D view of the static checkerboard."""

    points: np.ndarray
    rgb_normal: np.ndarray
    rgb_offset: float
    depth_normal: np.ndarray
    depth_offset: float
    rgb_rvec: np.ndarray
    rgb_tvec: np.ndarray
    reprojection_rms_px: float
    plane_rms_mm: float
    board_distance_mm: float
    rgb_image: str = ""
    depth_image: str = ""

    @property
    def num_points(self) -> int:
        return int(len(self.points))


@dataclass
class DepthPlaneDetection:
    """Result of locating the board plane in a depth image."""

    points: np.ndarray
    pixels: np.ndarray
    inliers: np.ndarray
    normal: np.ndarray
    offset: float
    residuals: np.ndarray
    plane_rms_mm: float
    normal_angle_deg: float
    polygon: np.ndarray
    prior_name: str
    prior_rotation: np.ndarray
    prior_translation: np.ndarray


# ---------------------------------------------------------------------------
# Observation persistence
# ---------------------------------------------------------------------------


def save_observations(
    observations: Sequence[Observation],
    path: Path = OBSERVATIONS_PATH,
) -> None:
    """Save observations as compact numeric arrays plus JSON metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lengths = np.array([o.num_points for o in observations], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)

    if observations:
        points = np.concatenate([o.points for o in observations], axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float64)

    def stack(attribute: str, shape: tuple[int, ...]) -> np.ndarray:
        values = [np.asarray(getattr(o, attribute)) for o in observations]

        if not values:
            return np.zeros((0,) + shape, dtype=np.float64)

        return np.stack(values).astype(np.float64)

    np.savez_compressed(
        path,
        points=points.astype(np.float64),
        point_offsets=offsets,
        rgb_normals=stack("rgb_normal", (3,)).reshape(-1, 3),
        rgb_offsets=np.array(
            [o.rgb_offset for o in observations], dtype=np.float64
        ),
        depth_normals=stack("depth_normal", (3,)).reshape(-1, 3),
        depth_offsets=np.array(
            [o.depth_offset for o in observations], dtype=np.float64
        ),
        rgb_rvecs=stack("rgb_rvec", (3,)).reshape(-1, 3),
        rgb_tvecs=stack("rgb_tvec", (3,)).reshape(-1, 3),
        reprojection_rms_px=np.array(
            [o.reprojection_rms_px for o in observations],
            dtype=np.float64,
        ),
        plane_rms_mm=np.array(
            [o.plane_rms_mm for o in observations], dtype=np.float64
        ),
        board_distance_mm=np.array(
            [o.board_distance_mm for o in observations],
            dtype=np.float64,
        ),
    )

    metadata = {
        "num_observations": len(observations),
        "observations": [
            {
                "index": index,
                "num_points": o.num_points,
                "rgb_image": o.rgb_image,
                "depth_image": o.depth_image,
                "reprojection_rms_px": float(o.reprojection_rms_px),
                "plane_rms_mm": float(o.plane_rms_mm),
                "board_distance_mm": float(o.board_distance_mm),
            }
            for index, o in enumerate(observations)
        ],
    }

    save_json(path.with_suffix(".json"), metadata)


def load_observations(
    path: Path = OBSERVATIONS_PATH,
) -> list[Observation]:
    """Load observations saved by :func:`save_observations`."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Observation file not found: {path}")

    metadata = {}

    metadata_path = path.with_suffix(".json")

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    image_metadata = metadata.get("observations", [])

    observations: list[Observation] = []

    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        offsets = np.asarray(data["point_offsets"], dtype=np.int64)

        for index in range(len(offsets) - 1):
            start = int(offsets[index])
            stop = int(offsets[index + 1])

            item = image_metadata[index] if index < len(image_metadata) else {}

            observations.append(
                Observation(
                    points=points[start:stop].copy(),
                    rgb_normal=np.asarray(
                        data["rgb_normals"][index], dtype=np.float64
                    ),
                    rgb_offset=float(data["rgb_offsets"][index]),
                    depth_normal=np.asarray(
                        data["depth_normals"][index], dtype=np.float64
                    ),
                    depth_offset=float(data["depth_offsets"][index]),
                    rgb_rvec=np.asarray(
                        data["rgb_rvecs"][index], dtype=np.float64
                    ),
                    rgb_tvec=np.asarray(
                        data["rgb_tvecs"][index], dtype=np.float64
                    ),
                    reprojection_rms_px=float(
                        data["reprojection_rms_px"][index]
                    ),
                    plane_rms_mm=float(data["plane_rms_mm"][index]),
                    board_distance_mm=float(
                        data["board_distance_mm"][index]
                    ),
                    rgb_image=str(item.get("rgb_image", "")),
                    depth_image=str(item.get("depth_image", "")),
                )
            )

    return observations


# ---------------------------------------------------------------------------
# Prior extrinsics
# ---------------------------------------------------------------------------


def load_prior_extrinsics() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load plausible RGB-D priors for region-of-interest projection.

    The priors are only used to place the depth region of interest and to
    seed plane selection.  The optimisation does not trust them.
    """

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

    files = [
        (
            "previous_calibration",
            RGBD_OUTPUT_DIR / "rgbd_calibration.npz",
            "rotation_depth_to_rgb",
            "translation_depth_to_rgb_mm",
        ),
        (
            "corner_calibration",
            RGBD_OUTPUT_DIR / "rgbd_corner_calibration.npz",
            "rotation_depth_to_rgb",
            "translation_depth_to_rgb_mm",
        ),
        (
            "plane_calibration",
            RGBD_OUTPUT_DIR / "rgbd_extrinsics.npz",
            "rotation",
            "translation",
        ),
        (
            "reprojection_calibration",
            RGBD_OUTPUT_DIR / "rgbd_extrinsics_reprojection.npz",
            "rotation",
            "translation",
        ),
    ]

    for name, path, rotation_key, translation_key in files:
        if not path.exists():
            continue

        try:
            with np.load(path) as data:
                if rotation_key not in data.files:
                    continue

                if translation_key not in data.files:
                    continue

                rotation = np.asarray(
                    data[rotation_key], dtype=np.float64
                ).reshape(3, 3)

                translation = np.asarray(
                    data[translation_key], dtype=np.float64
                ).reshape(3)

                if not np.all(np.isfinite(rotation)):
                    continue

                if not np.all(np.isfinite(translation)):
                    continue

                # A sane Xtion prior is a small rotation and a translation
                # of at most a few centimetres.
                if np.linalg.norm(translation) > 250.0:
                    continue

                candidates.append((name, rotation, translation))

        except Exception:
            continue

    # De-duplicate nearly identical priors.
    unique: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name, rotation, translation in candidates:
        duplicate = False

        for _, other_rotation, other_translation in unique:
            if (
                np.linalg.norm(rotation - other_rotation) < 1e-5
                and np.linalg.norm(translation - other_translation) < 0.1
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append((name, rotation, translation))

    identity = ("identity", np.eye(3), np.zeros(3))

    # Keep the previous calibration first (it is the best local prior),
    # followed by identity, then the older experimental calibrations.
    if unique and unique[0][0] == "previous_calibration":
        return [unique[0], identity] + unique[1:]

    return [identity] + unique


def prior_parameters(
    use_depth_scale: bool,
) -> Optional[np.ndarray]:
    """Return the most plausible prior as optimisation parameters."""

    for name, rotation, translation in load_prior_extrinsics():
        if name == "identity":
            continue

        try:
            rvec = Rotation.from_matrix(rotation).as_rotvec()
        except Exception:
            continue

        parameters = np.concatenate([rvec, translation])

        if use_depth_scale:
            parameters = np.concatenate([parameters, [0.0]])

        return parameters

    return None


# ---------------------------------------------------------------------------
# Depth plane detection
# ---------------------------------------------------------------------------


def detect_depth_plane(
    depth: np.ndarray,
    pose: BoardPose,
    depth_intrinsics: CameraIntrinsics,
    priors: Sequence[tuple[str, np.ndarray, np.ndarray]],
    fast: bool = False,
    extra_margin_mm: float = 10.0,
    max_points_per_pose: int = MAX_POINTS_PER_POSE,
    max_priors: Optional[int] = None,
) -> Optional[DepthPlaneDetection]:
    """Find the board plane in a depth image using RGB pose guidance."""

    if fast:
        stride = 2
        max_iterations = 300
        min_inliers = PREVIEW_MIN_INLIERS
    else:
        stride = 1
        max_iterations = 1200
        min_inliers = 40

    priors = list(priors)

    if max_priors is not None:
        priors = priors[: int(max_priors)]

    best: Optional[DepthPlaneDetection] = None
    best_quality = -np.inf

    for prior_name, prior_rotation, prior_translation in priors:
        polygon, _ = board_polygon_in_depth(
            pose,
            depth_intrinsics.camera_matrix,
            prior_rotation,
            prior_translation,
            extra_margin_mm=extra_margin_mm,
        )

        mask = polygon_mask(depth.shape, polygon)

        points, pixels = depth_to_points(
            depth,
            depth_intrinsics.camera_matrix,
            min_depth_mm=DEFAULT_MIN_DEPTH_MM,
            max_depth_mm=DEFAULT_MAX_DEPTH_MM,
            stride=stride,
            mask=mask,
        )

        if len(points) < min_inliers:
            continue

        predicted_normal = prior_rotation.T @ pose.normal

        predicted_offset = float(
            pose.offset + np.dot(pose.normal, prior_translation)
        )

        try:
            normal, offset, residuals, inliers = select_plane(
                points,
                predicted_normal=predicted_normal,
                predicted_offset=predicted_offset,
                normal_tolerance_deg=35.0,
                distance_threshold_mm=8.0,
                seed_threshold_mm=60.0,
                max_iterations=max_iterations,
                min_inliers=min_inliers,
                random_seed=0,
            )
        except Exception:
            continue

        inlier_count = int(np.count_nonzero(inliers))

        if inlier_count < min_inliers:
            continue

        plane_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))

        normal_angle = plane_angle_deg(normal, predicted_normal)

        # Prefer many inliers, a low plane RMS, and a normal close to the
        # RGB prediction.  The weights are deliberately soft because the
        # prior translation can be off by a few centimetres.
        quality = inlier_count / (
            1.0
            + (normal_angle / 12.0) ** 2
            + (plane_rms / 6.0) ** 2
        )

        if quality <= best_quality:
            continue

        inlier_points = points[inliers]
        inlier_pixels = pixels[inliers]

        if len(inlier_points) > max_points_per_pose:
            rng = np.random.default_rng(12345)
            selection = rng.choice(
                len(inlier_points),
                size=max_points_per_pose,
                replace=False,
            )
            inlier_points = inlier_points[selection]
            inlier_pixels = inlier_pixels[selection]

        best_quality = quality

        best = DepthPlaneDetection(
            points=inlier_points,
            pixels=inlier_pixels,
            inliers=inliers,
            normal=normal,
            offset=offset,
            residuals=residuals,
            plane_rms_mm=plane_rms,
            normal_angle_deg=normal_angle,
            polygon=polygon,
            prior_name=prior_name,
            prior_rotation=prior_rotation,
            prior_translation=prior_translation,
        )

    return best


def validate_detection(
    pose: BoardPose,
    detection: Optional[DepthPlaneDetection],
    capture: bool = True,
    rgb_intrinsics: Optional[CameraIntrinsics] = None,
) -> tuple[bool, list[str]]:
    """Check whether a detected board/depth plane is usable."""

    messages: list[str] = []

    if detection is None:
        return False, ["No depth plane found"]

    if pose.reprojection_rms_px > CAPTURE_MAX_REPROJECTION_RMS_PX:
        messages.append(
            f"RGB reprojection RMS too high: "
            f"{pose.reprojection_rms_px:.3f} px"
        )

    if detection.plane_rms_mm > CAPTURE_MAX_PLANE_RMS_MM:
        messages.append(
            f"Depth plane RMS too high: {detection.plane_rms_mm:.3f} mm"
        )

    if detection.normal_angle_deg > CAPTURE_MAX_NORMAL_ANGLE_DEG:
        messages.append(
            f"Depth/RGB normal angle too large: "
            f"{detection.normal_angle_deg:.2f} deg"
        )

    if len(detection.points) < CAPTURE_MIN_INLIERS:
        messages.append(
            f"Too few depth points: {len(detection.points)}"
        )

    if pose.distance_mm < CAPTURE_MIN_DISTANCE_MM:
        messages.append(
            f"Board too close: {pose.distance_mm:.0f} mm"
        )

    if pose.distance_mm > CAPTURE_MAX_DISTANCE_MM:
        messages.append(
            f"Board too far: {pose.distance_mm:.0f} mm"
        )

    # Area of the printed pattern in the RGB image.
    if rgb_intrinsics is None:
        rgb_intrinsics = load_rgb_intrinsics()

    pattern_rgb = pose.transform_to_rgb(pattern_outer_corners_mm())

    projected = project_points(
        pattern_rgb,
        rgb_intrinsics.camera_matrix,
        rgb_intrinsics.distortion,
    )

    area = abs(
        float(
            cv2.contourArea(
                np.round(projected).astype(np.float32)
            )
        )
    )

    if capture and area < CAPTURE_MIN_PATTERN_AREA_PX:
        messages.append(
            f"Board is too small in the RGB image: {area:.0f} px"
        )

    if capture and (
        pose.distance_mm < CAPTURE_RECOMMENDED_MIN_MM
        or pose.distance_mm > CAPTURE_RECOMMENDED_MAX_MM
    ):
        messages.append(
            "Recommended board distance is "
            f"{CAPTURE_RECOMMENDED_MIN_MM:.0f}-"
            f"{CAPTURE_RECOMMENDED_MAX_MM:.0f} mm "
            f"(current {pose.distance_mm:.0f} mm)"
        )

    # Warnings that do not necessarily reject the pose.
    hard_failures = [
        message
        for message in messages
        if not message.startswith("Recommended board distance")
    ]

    return len(hard_failures) == 0, messages


# ---------------------------------------------------------------------------
# Pose diversity guidance
# ---------------------------------------------------------------------------

TILT_BINS: tuple[tuple[str, float, float], ...] = (
    ("front", 0.0, 10.0),
    ("moderate", 10.0, 25.0),
    ("strong", 25.0, 45.0),
    ("extreme", 45.0, 181.0),
)

DISTANCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("near", 0.0, 700.0),
    ("mid", 700.0, 900.0),
    ("far", 900.0, 1.0e9),
)

DIVERSITY_MIN_STRONG_TILTS = 3
DIVERSITY_MIN_DIRECTIONS = 3
DIVERSITY_MIN_POSITION_CELLS = 3
DIVERSITY_MIN_DISTANCE_BINS = 2

SIMILAR_POSE_MAX_ANGLE_DEG = 8.0
SIMILAR_POSE_MAX_DISTANCE_MM = 50.0
SIMILAR_POSE_MAX_IMAGE_FRACTION = 0.10


def board_tilt_deg(normal: np.ndarray) -> float:
    """Angle between the board normal and the camera optical axis."""

    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)

    # The board normal is oriented towards the camera, so its Z component is
    # negative for a front-facing board.
    cosine = float(np.clip(abs(normal[2]), 0.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


def board_tilt_direction(normal: np.ndarray) -> str:
    """Coarse tilt direction in image axes (L/R/U/D), or ``front``."""

    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)

    if board_tilt_deg(normal) < 8.0:
        return "front"

    # Direction from the board towards the camera.  Positive X points right
    # in the image, positive Y points down.
    towards_camera = -normal

    if abs(towards_camera[0]) >= abs(towards_camera[1]):
        return "R" if towards_camera[0] > 0 else "L"

    return "D" if towards_camera[1] > 0 else "U"


def _bin_name(
    value: float,
    bins: Sequence[tuple[str, float, float]],
) -> str:
    for name, low, high in bins:
        if low <= value < high:
            return name

    return bins[-1][0]


def _project_board_center(
    rotation: np.ndarray,
    translation: np.ndarray,
    rgb_intrinsics: CameraIntrinsics,
) -> Optional[np.ndarray]:
    """Project the printed-pattern centre into the RGB image, if visible."""

    center = pattern_outer_corners_mm().mean(axis=0, keepdims=True)

    point = transform_points(
        center,
        np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        np.asarray(translation, dtype=np.float64).reshape(3),
    )

    projected = project_points(
        point,
        rgb_intrinsics.camera_matrix,
        rgb_intrinsics.distortion,
    )

    if projected.shape[0] < 1:
        return None

    uv = np.asarray(projected[0], dtype=np.float64)

    if not np.all(np.isfinite(uv)):
        return None

    if not (
        0.0 <= uv[0] <= rgb_intrinsics.width
        and 0.0 <= uv[1] <= rgb_intrinsics.height
    ):
        return None

    return uv


def _observation_board_center(
    observation: Observation,
    rgb_intrinsics: CameraIntrinsics,
) -> Optional[np.ndarray]:
    return _project_board_center(
        _rotation_matrix(observation.rgb_rvec),
        observation.rgb_tvec,
        rgb_intrinsics,
    )


@dataclass
class PoseDiversity:
    """Coverage summary of the captured board poses."""

    total: int
    tilt_counts: dict[str, int]
    direction_counts: dict[str, int]
    distance_counts: dict[str, int]
    position_cells: set[tuple[int, int]]

    @property
    def strong_tilts(self) -> int:
        return (
            self.tilt_counts.get("strong", 0)
            + self.tilt_counts.get("extreme", 0)
        )

    @property
    def used_directions(self) -> int:
        return sum(
            1 for key in ("L", "R", "U", "D")
            if self.direction_counts.get(key, 0) > 0
        )

    @property
    def used_distance_bins(self) -> int:
        return sum(1 for value in self.distance_counts.values() if value > 0)

    def summary_line(self) -> str:
        tilt = "/".join(
            str(self.tilt_counts.get(name, 0))
            for name, _, _ in TILT_BINS
        )
        directions = "/".join(
            str(self.direction_counts.get(key, 0))
            for key in ("L", "R", "U", "D")
        )
        distances = "/".join(
            str(self.distance_counts.get(name, 0))
            for name, _, _ in DISTANCE_BINS
        )

        return (
            f"Tilt F/M/S/X {tilt}  Dir L/R/U/D {directions}  "
            f"Pos {len(self.position_cells)}/9  Dist {distances}"
        )

    def issues(self, target_poses: int) -> list[str]:
        issues: list[str] = []

        if self.total < target_poses:
            issues.append(
                f"Only {self.total}/{target_poses} poses captured"
            )

        if self.strong_tilts < DIVERSITY_MIN_STRONG_TILTS:
            issues.append(
                f"Need at least {DIVERSITY_MIN_STRONG_TILTS} strongly "
                f"tilted poses (25-45 deg); have {self.strong_tilts}"
            )

        if self.used_directions < DIVERSITY_MIN_DIRECTIONS:
            issues.append(
                f"Tilt the board in at least {DIVERSITY_MIN_DIRECTIONS} "
                f"directions (left/right/up/down); have "
                f"{self.used_directions}"
            )

        if len(self.position_cells) < DIVERSITY_MIN_POSITION_CELLS:
            issues.append(
                f"Move the board to more image regions; covered "
                f"{len(self.position_cells)}/9 cells, want at least "
                f"{DIVERSITY_MIN_POSITION_CELLS}"
            )

        if self.used_distance_bins < DIVERSITY_MIN_DISTANCE_BINS:
            issues.append(
                "Vary the board distance; used "
                f"{self.used_distance_bins} distance bins, want at least "
                f"{DIVERSITY_MIN_DISTANCE_BINS}"
            )

        return issues


def pose_diversity(
    observations: Sequence[Observation],
    rgb_intrinsics: Optional[CameraIntrinsics] = None,
) -> PoseDiversity:
    """Summarise tilt, direction, distance, and image coverage."""

    if rgb_intrinsics is None:
        rgb_intrinsics = load_rgb_intrinsics()

    tilt_counts = {name: 0 for name, _, _ in TILT_BINS}
    direction_counts = {"front": 0, "L": 0, "R": 0, "U": 0, "D": 0}
    distance_counts = {name: 0 for name, _, _ in DISTANCE_BINS}
    position_cells: set[tuple[int, int]] = set()

    for observation in observations:
        tilt_counts[
            _bin_name(board_tilt_deg(observation.rgb_normal), TILT_BINS)
        ] += 1

        direction_counts[
            board_tilt_direction(observation.rgb_normal)
        ] += 1

        distance_counts[
            _bin_name(observation.board_distance_mm, DISTANCE_BINS)
        ] += 1

        center = _observation_board_center(observation, rgb_intrinsics)

        if center is None:
            continue

        column = int(3.0 * center[0] / max(rgb_intrinsics.width, 1))
        row = int(3.0 * center[1] / max(rgb_intrinsics.height, 1))

        position_cells.add(
            (min(2, max(0, column)), min(2, max(0, row)))
        )

    return PoseDiversity(
        total=len(observations),
        tilt_counts=tilt_counts,
        direction_counts=direction_counts,
        distance_counts=distance_counts,
        position_cells=position_cells,
    )


def find_similar_pose(
    observations: Sequence[Observation],
    rvec: np.ndarray,
    tvec: np.ndarray,
    normal: np.ndarray,
    distance_mm: float,
    rgb_intrinsics: CameraIntrinsics,
) -> Optional[int]:
    """Return the 1-based index of an almost identical captured pose."""

    if not observations:
        return None

    rotation = _rotation_matrix(rvec)

    center = _project_board_center(rotation, tvec, rgb_intrinsics)

    image_diagonal = float(
        np.hypot(rgb_intrinsics.width, rgb_intrinsics.height)
    )

    for index, observation in enumerate(observations, start=1):
        angle = plane_angle_deg(observation.rgb_normal, normal)

        if angle > SIMILAR_POSE_MAX_ANGLE_DEG:
            continue

        distance_delta = abs(observation.board_distance_mm - distance_mm)

        if distance_delta > SIMILAR_POSE_MAX_DISTANCE_MM:
            continue

        if center is None:
            return index

        other_center = _observation_board_center(
            observation, rgb_intrinsics
        )

        if other_center is None:
            continue

        if np.linalg.norm(other_center - center) <= (
            SIMILAR_POSE_MAX_IMAGE_FRACTION * image_diagonal
        ):
            return index

    return None


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _draw_board_preview(
    image: np.ndarray,
    pose: BoardPose,
    rgb_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    output = image.copy()

    cv2.drawChessboardCorners(
        output,
        (BOARD_COLUMNS, BOARD_ROWS),
        pose.corners,
        True,
    )

    pattern_rgb = pose.transform_to_rgb(pattern_outer_corners_mm())

    projected = project_points(
        pattern_rgb,
        rgb_intrinsics.camera_matrix,
        rgb_intrinsics.distortion,
    )

    cv2.polylines(
        output,
        [np.round(projected).astype(np.int32)],
        True,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def _draw_depth_preview(
    depth: np.ndarray,
    detection: Optional[DepthPlaneDetection],
) -> np.ndarray:
    output = depth_visualization(depth)

    if detection is None:
        return output

    cv2.polylines(
        output,
        [np.round(detection.polygon).astype(np.int32)],
        True,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    for u, v in detection.pixels[::3]:
        cv2.circle(
            output,
            (int(round(u)), int(round(v))),
            1,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

    return output


def _capture_one(
    rgb: np.ndarray,
    depth: np.ndarray,
    rgb_intrinsics: CameraIntrinsics,
    depth_intrinsics: CameraIntrinsics,
    priors: Sequence[tuple[str, np.ndarray, np.ndarray]],
    index: int,
) -> tuple[Optional[Observation], list[str]]:
    """Detect, validate, and build one observation from a frame pair."""

    found, corners, _ = detect_checkerboard(
        rgb,
        columns=BOARD_COLUMNS,
        rows=BOARD_ROWS,
        refine=True,
        fast=False,
    )

    if not found:
        return None, ["Checkerboard not detected"]

    pose = solve_board_pose(
        corners,
        rgb_intrinsics.camera_matrix,
        rgb_intrinsics.distortion,
    )

    if pose is None:
        return None, ["PnP failed"]

    detection = detect_depth_plane(
        depth,
        pose,
        depth_intrinsics,
        priors,
        fast=False,
    )

    valid, messages = validate_detection(
        pose,
        detection,
        capture=True,
        rgb_intrinsics=rgb_intrinsics,
    )

    if not valid or detection is None:
        return None, messages

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    rgb_path = CAPTURES_DIR / f"pose_{index:03d}_rgb.png"
    depth_path = CAPTURES_DIR / f"pose_{index:03d}_depth.npy"

    cv2.imwrite(str(rgb_path), rgb)
    np.save(depth_path, depth)

    observation = Observation(
        points=detection.points,
        rgb_normal=pose.normal,
        rgb_offset=pose.offset,
        depth_normal=detection.normal,
        depth_offset=detection.offset,
        rgb_rvec=pose.rvec,
        rgb_tvec=pose.tvec,
        reprojection_rms_px=pose.reprojection_rms_px,
        plane_rms_mm=detection.plane_rms_mm,
        board_distance_mm=pose.distance_mm,
        rgb_image=str(rgb_path),
        depth_image=str(depth_path),
    )

    return observation, messages


def capture_observations(
    target_poses: int = TARGET_POSES,
    min_poses: int = MIN_POSES,
) -> list[Observation]:
    """Interactively capture checkerboard observations."""

    from xtion import XtionCamera

    rgb_intrinsics = load_rgb_intrinsics()
    depth_intrinsics = load_depth_intrinsics()
    priors = load_prior_extrinsics()

    print("\n==============================================")
    print(" RGB-D EXTRINSIC CAPTURE")
    print("==============================================")
    print(f"RGB intrinsics:  {rgb_intrinsics.width}x{rgb_intrinsics.height}")
    print(
        f"Depth intrinsics: "
        f"{depth_intrinsics.width}x{depth_intrinsics.height}"
    )
    print(f"Board: {BOARD_COLUMNS}x{BOARD_ROWS} internal corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print()
    print("Hold the board still at roughly "
          f"{CAPTURE_RECOMMENDED_MIN_MM:.0f}-"
          f"{CAPTURE_RECOMMENDED_MAX_MM:.0f} mm from the camera.")
    print("Vary the tilt: front-facing, then roughly +/-30-45 degrees.")
    print("Move the board across the whole image and vary its distance.")
    print(
        "Overlay legend: Tilt F/M/S/X = front/moderate/strong/extreme, "
        "Dir = tilt direction,"
    )
    print(
        "Pos = covered image cells (3x3), "
        "Dist = near/mid/far (<700 / 700-900 / >900 mm)."
    )
    print()
    print("SPACE : capture current pose")
    print("U     : undo last pose")
    print("O     : save observations and optimise now")
    print("Q/ESC : finish capture")
    print("==============================================\n")

    camera = XtionCamera()

    observations: list[Observation] = []
    diversity_gate_warned = False

    try:
        camera.open()

        cv2.namedWindow("RGB capture", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth capture", cv2.WINDOW_NORMAL)

        while True:
            rgb, depth = camera.get_frames()

            preview = rgb.copy()
            detection = None
            pose = None

            found, corners, _ = detect_checkerboard(
                rgb,
                columns=BOARD_COLUMNS,
                rows=BOARD_ROWS,
                refine=False,
                fast=True,
            )

            if found:
                pose = solve_board_pose(
                    corners,
                    rgb_intrinsics.camera_matrix,
                    rgb_intrinsics.distortion,
                )

            if pose is not None:
                preview = _draw_board_preview(
                    preview, pose, rgb_intrinsics
                )

                detection = detect_depth_plane(
                    depth,
                    pose,
                    depth_intrinsics,
                    priors,
                    fast=True,
                    max_priors=2,
                )

                valid, messages = validate_detection(
                    pose,
                    detection,
                    capture=False,
                    rgb_intrinsics=rgb_intrinsics,
                )

                status = "READY - SPACE TO CAPTURE" if valid else "NOT READY"
                color = (0, 255, 0) if valid else (0, 0, 255)
            else:
                status = "CHECKERBOARD NOT FOUND"
                color = (0, 0, 255)
                messages = []

            cv2.putText(
                preview,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                preview,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            info_lines = [f"Captured: {len(observations)}/{target_poses}"]

            if pose is not None:
                info_lines.append(
                    f"Distance: {pose.distance_mm:.0f} mm"
                )
                info_lines.append(
                    f"RGB reprojection: {pose.reprojection_rms_px:.2f} px"
                )

            if detection is not None:
                info_lines.append(
                    f"Depth points: {len(detection.points)}"
                )
                info_lines.append(
                    f"Plane RMS: {detection.plane_rms_mm:.2f} mm"
                )
                info_lines.append(
                    f"Normal angle: {detection.normal_angle_deg:.2f} deg"
                )
                info_lines.append(f"Prior: {detection.prior_name}")

            if observations:
                info_lines.append(
                    pose_diversity(
                        observations, rgb_intrinsics
                    ).summary_line()
                )

            if pose is not None:
                similar = find_similar_pose(
                    observations,
                    pose.rvec,
                    pose.tvec,
                    pose.normal,
                    pose.distance_mm,
                    rgb_intrinsics,
                )

                if similar is not None:
                    info_lines.append(
                        f"Similar to pose {similar}: move or tilt the board"
                    )

            y = 80

            for line in info_lines:
                cv2.putText(
                    preview,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    preview,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                y += 26

            if messages:
                y += 6

                for message in messages[:4]:
                    cv2.putText(
                        preview,
                        message,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                    y += 22

            depth_preview = _draw_depth_preview(depth, detection)

            cv2.imshow("RGB capture", preview)
            cv2.imshow("Depth capture", depth_preview)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("u"):
                if observations:
                    removed = observations.pop()
                    print(
                        f"Removed pose {len(observations)} "
                        f"({removed.rgb_image})"
                    )
                continue

            if key == ord("o"):
                if len(observations) < 3:
                    print("Need at least 3 poses before optimising.")
                    continue

                save_observations(observations)
                print(f"Saved {len(observations)} observations.")
                break

            if key != 32:
                continue

            # Capture a fresh frame pair so RGB and depth are consistent.
            rgb_capture, depth_capture = camera.get_frames()

            observation, messages = _capture_one(
                rgb_capture,
                depth_capture,
                rgb_intrinsics,
                depth_intrinsics,
                priors,
                len(observations),
            )

            if observation is None:
                print("Capture rejected:")
                for message in messages:
                    print("  -", message)
                continue

            similar = find_similar_pose(
                observations,
                observation.rgb_rvec,
                observation.rgb_tvec,
                observation.rgb_normal,
                observation.board_distance_mm,
                rgb_intrinsics,
            )

            observations.append(observation)

            if similar is not None:
                print(
                    f"  warning: very similar to pose {similar}; "
                    "move or tilt the board for more information"
                )

            print(
                f"Accepted pose {len(observations)} | "
                f"distance {observation.board_distance_mm:.0f} mm | "
                f"points {observation.num_points} | "
                f"plane RMS {observation.plane_rms_mm:.2f} mm | "
                f"normal {observation.rgb_normal}"
            )

            for message in messages:
                print("  warning:", message)

            diversity = pose_diversity(observations, rgb_intrinsics)

            print(f"  coverage: {diversity.summary_line()}")

            if len(observations) >= target_poses:
                issues = diversity.issues(target_poses)

                if not issues:
                    print(
                        f"Reached target of {target_poses} poses "
                        "with good coverage."
                    )
                    break

                if not diversity_gate_warned:
                    print(
                        f"\nReached {target_poses} poses, but coverage is "
                        "still limited:"
                    )

                    for issue in issues:
                        print(f"  - {issue}")

                    print(
                        "Add more varied poses, or press Q/ESC to finish "
                        "anyway.\n"
                    )

                    diversity_gate_warned = True

        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\nCapture interrupted.")

    finally:
        camera.close()
        cv2.destroyAllWindows()

    if observations:
        diversity = pose_diversity(observations, rgb_intrinsics)

        print("\nCapture coverage:")
        print(f"  {diversity.summary_line()}")

        issues = diversity.issues(target_poses)

        if issues:
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("  Coverage looks good.")

        save_observations(observations)
        print(f"\nSaved {len(observations)} observations to {OBSERVATIONS_PATH}")

    if len(observations) < min_poses:
        print(
            f"\nOnly {len(observations)} poses captured. "
            f"At least {min_poses} are recommended."
        )

    return observations


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


def _rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(
        np.asarray(rvec, dtype=np.float64).reshape(3)
    )

    return rotation


def _scale_from_parameters(
    parameters: np.ndarray, use_depth_scale: bool
) -> float:
    if use_depth_scale:
        return float(np.exp(parameters[6]))

    return 1.0


def _residual_data(
    observations: Sequence[Observation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.concatenate([o.points for o in observations], axis=0)

    normals = np.concatenate(
        [
            np.tile(
                np.asarray(o.rgb_normal, dtype=np.float64).reshape(1, 3),
                (o.num_points, 1),
            )
            for o in observations
        ],
        axis=0,
    )

    offsets = np.concatenate(
        [
            np.full(o.num_points, float(o.rgb_offset), dtype=np.float64)
            for o in observations
        ]
    )

    return points, normals, offsets


def initial_guess_from_planes(
    observations: Sequence[Observation],
    use_depth_scale: bool,
) -> np.ndarray:
    """Closed-form initial guess from the per-pose plane parameters."""

    depth_normals = np.stack(
        [np.asarray(o.depth_normal, dtype=np.float64) for o in observations]
    )

    rgb_normals = np.stack(
        [np.asarray(o.rgb_normal, dtype=np.float64) for o in observations]
    )

    try:
        rotation = Rotation.align_vectors(
            rgb_normals, depth_normals
        )[0].as_matrix()
    except Exception:
        rotation = np.eye(3)

    depth_offsets = np.array(
        [o.depth_offset for o in observations], dtype=np.float64
    )

    rgb_offsets = np.array(
        [o.rgb_offset for o in observations], dtype=np.float64
    )

    if use_depth_scale:
        # n_r . t - s * d_d = -d_r
        matrix = np.column_stack([rgb_normals, -depth_offsets])
        solution, *_ = np.linalg.lstsq(
            matrix, -rgb_offsets, rcond=None
        )

        translation = solution[:3]
        scale = float(solution[3])

        if not np.isfinite(scale) or not (0.8 <= scale <= 1.2):
            scale = 1.0
            translation, *_ = np.linalg.lstsq(
                rgb_normals,
                depth_offsets - rgb_offsets,
                rcond=None,
            )
    else:
        # n_r . t = d_d - d_r
        translation, *_ = np.linalg.lstsq(
            rgb_normals,
            depth_offsets - rgb_offsets,
            rcond=None,
        )
        scale = 1.0

    rvec = Rotation.from_matrix(rotation).as_rotvec()

    parameters = np.concatenate([rvec, translation])

    if use_depth_scale:
        parameters = np.concatenate([parameters, [np.log(scale)]])

    return parameters


def build_initial_guesses(
    observations: Sequence[Observation],
    use_depth_scale: bool,
) -> list[tuple[str, np.ndarray]]:
    guesses: list[tuple[str, np.ndarray]] = []

    plane_guess = initial_guess_from_planes(
        observations, use_depth_scale
    )

    guesses.append(("plane", plane_guess))

    identity = np.zeros(7 if use_depth_scale else 6, dtype=np.float64)

    guesses.append(("identity", identity))

    prior = prior_parameters(use_depth_scale)

    if prior is not None and len(prior) == len(identity):
        guesses.append(("prior", prior))

    # Small rotation perturbations around the closed-form solution.
    for axis, angle_deg in ((0, 5.0), (1, -5.0), (2, 5.0)):
        perturbed = plane_guess.copy()

        perturbation = Rotation.from_rotvec(
            np.eye(3)[axis] * np.radians(angle_deg)
        ).as_matrix()

        base_rotation = _rotation_matrix(perturbed[:3])

        perturbed[:3] = Rotation.from_matrix(
            perturbation @ base_rotation
        ).as_rotvec()

        guesses.append((f"plane_rot_{axis}", perturbed))

    # Remove duplicates.
    unique: list[tuple[str, np.ndarray]] = []

    for name, parameters in guesses:
        duplicate = False

        for _, other in unique:
            if np.linalg.norm(parameters - other) < 1e-8:
                duplicate = True
                break

        if not duplicate:
            unique.append((name, parameters))

    return unique


def evaluate_solution(
    parameters: np.ndarray,
    observations: Sequence[Observation],
    use_depth_scale: bool,
) -> dict:
    """Compute point-to-plane and plane-parameter metrics per pose."""

    rotation = _rotation_matrix(parameters[:3])

    translation = np.asarray(parameters[3:6], dtype=np.float64)

    scale = _scale_from_parameters(parameters, use_depth_scale)

    per_pose = []

    all_residuals = []

    for index, observation in enumerate(observations):
        points = observation.points * scale

        points_rgb = points @ rotation.T + translation

        residuals = (
            points_rgb
            @ np.asarray(observation.rgb_normal, dtype=np.float64)
            + float(observation.rgb_offset)
        )

        absolute = np.abs(residuals)

        predicted_normal = rotation @ np.asarray(
            observation.depth_normal, dtype=np.float64
        )

        normal_angle = plane_angle_deg(
            predicted_normal, observation.rgb_normal
        )

        predicted_offset = (
            scale * float(observation.depth_offset)
            - float(np.dot(predicted_normal, translation))
        )

        offset_error = predicted_offset - float(observation.rgb_offset)

        per_pose.append(
            {
                "index": index,
                "num_points": int(len(residuals)),
                "rms_mm": float(np.sqrt(np.mean(residuals**2))),
                "median_abs_mm": float(np.median(absolute)),
                "p95_abs_mm": float(np.percentile(absolute, 95)),
                "max_abs_mm": float(np.max(absolute)),
                "normal_angle_deg": float(normal_angle),
                "offset_error_mm": float(offset_error),
            }
        )

        all_residuals.append(residuals)

    combined = np.concatenate(all_residuals) if all_residuals else np.empty(0)

    absolute = np.abs(combined)

    return {
        "rms_mm": float(np.sqrt(np.mean(combined**2))),
        "median_abs_mm": float(np.median(absolute)),
        "p95_abs_mm": float(np.percentile(absolute, 95)),
        "max_abs_mm": float(np.max(absolute)),
        "per_pose": per_pose,
    }


def optimize_extrinsics(
    observations: Sequence[Observation],
    use_depth_scale: bool = False,
    multistart: bool = True,
    initial_parameters: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """Optimise the RGB-D transform with point-to-plane residuals."""

    if len(observations) < 3:
        raise RuntimeError("At least three observations are required")

    points, normals, offsets = _residual_data(observations)

    parameter_count = 7 if use_depth_scale else 6

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        rotation = _rotation_matrix(parameters[:3])

        translation = parameters[3:6]

        scale = _scale_from_parameters(parameters, use_depth_scale)

        transformed = (points * scale) @ rotation.T + translation

        return np.einsum(
            "ij,ij->i", transformed, normals
        ) + offsets

    if multistart:
        guesses = build_initial_guesses(observations, use_depth_scale)
    else:
        if initial_parameters is None:
            initial_parameters = initial_guess_from_planes(
                observations, use_depth_scale
            )

        guesses = [("initial", np.asarray(initial_parameters, np.float64))]

    lower = np.array(
        [-np.inf, -np.inf, -np.inf, -500.0, -500.0, -500.0]
        + ([-0.7] if use_depth_scale else []),
        dtype=np.float64,
    )

    upper = np.array(
        [np.inf, np.inf, np.inf, 500.0, 500.0, 500.0]
        + ([0.7] if use_depth_scale else []),
        dtype=np.float64,
    )

    best: Optional[dict] = None

    for name, initial in guesses:
        initial = np.asarray(initial, dtype=np.float64).reshape(
            parameter_count
        )

        initial = np.clip(initial, lower, upper)

        try:
            result = least_squares(
                residual_function,
                initial,
                method="trf",
                loss="soft_l1",
                f_scale=OPTIMIZATION_LOSS_SCALE_MM,
                bounds=(lower, upper),
                x_scale="jac",
                max_nfev=OPTIMIZATION_MAX_NFEV,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
        except Exception as error:
            if verbose:
                print(f"  start {name}: failed ({error})")

            continue

        metrics = evaluate_solution(
            result.x, observations, use_depth_scale
        )

        score = float(result.cost)

        if verbose:
            print(
                f"  start {name:12s}: "
                f"cost {score:12.3f}  "
                f"RMS {metrics['rms_mm']:8.3f} mm  "
                f"p95 {metrics['p95_abs_mm']:8.3f} mm"
            )

        candidate = {
            "parameters": np.asarray(result.x, dtype=np.float64),
            "rotation": _rotation_matrix(result.x[:3]),
            "translation": np.asarray(
                result.x[3:6], dtype=np.float64
            ),
            "depth_scale": _scale_from_parameters(
                result.x, use_depth_scale
            ),
            "result": result,
            "metrics": metrics,
            "score": score,
            "initial_guess": name,
            "num_starts": len(guesses),
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("All optimisation starts failed")

    return best


def leave_one_out_validation(
    observations: Sequence[Observation],
    parameters: np.ndarray,
    use_depth_scale: bool,
    verbose: bool = True,
) -> list[float]:
    """Evaluate each pose with a transform fitted to the other poses."""

    held_out_rms: list[float] = []

    for index in range(len(observations)):
        training = (
            list(observations[:index]) + list(observations[index + 1 :])
        )

        held_out = observations[index]

        try:
            result = optimize_extrinsics(
                training,
                use_depth_scale=use_depth_scale,
                multistart=False,
                initial_parameters=parameters,
                verbose=False,
            )

            metrics = evaluate_solution(
                result["parameters"], [held_out], use_depth_scale
            )

            value = float(metrics["rms_mm"])

        except Exception:
            value = float("nan")

        held_out_rms.append(value)

        if verbose:
            print(
                f"  leave-one-out {index:02d}: "
                f"{value:8.3f} mm"
            )

    return held_out_rms


# ---------------------------------------------------------------------------
# Reporting and persistence
# ---------------------------------------------------------------------------


def print_metrics(
    metrics: dict,
    held_out_rms: Optional[Sequence[float]] = None,
) -> None:
    print("\n==============================================")
    print(" RGB-D CALIBRATION METRICS")
    print("==============================================")
    print(f"Overall point-to-plane RMS : {metrics['rms_mm']:.3f} mm")
    print(f"Median absolute residual   : {metrics['median_abs_mm']:.3f} mm")
    print(f"95th percentile residual   : {metrics['p95_abs_mm']:.3f} mm")
    print(f"Maximum absolute residual  : {metrics['max_abs_mm']:.3f} mm")
    print()
    print("Per pose:")
    print(
        "  idx  points    RMS      p95      max    normal   offset"
    )

    for pose in metrics["per_pose"]:
        print(
            f"  {pose['index']:02d}  "
            f"{pose['num_points']:6d}  "
            f"{pose['rms_mm']:7.3f}  "
            f"{pose['p95_abs_mm']:7.3f}  "
            f"{pose['max_abs_mm']:7.3f}  "
            f"{pose['normal_angle_deg']:7.2f}  "
            f"{pose['offset_error_mm']:7.2f}"
        )

    rms_values = [pose["rms_mm"] for pose in metrics["per_pose"]]

    if rms_values:
        median_rms = float(np.median(rms_values))

        threshold = max(5.0, 2.0 * median_rms)

        outliers = [
            pose
            for pose in metrics["per_pose"]
            if pose["rms_mm"] > threshold
            or pose["normal_angle_deg"] > 5.0
        ]

        if outliers:
            print("\nSuspicious poses (consider recapturing):")

            for pose in outliers:
                print(
                    f"  pose {pose['index']:02d}: "
                    f"RMS {pose['rms_mm']:.3f} mm, "
                    f"normal {pose['normal_angle_deg']:.2f} deg"
                )
        else:
            print("\nAll poses look consistent.")

    if held_out_rms is not None:
        finite = np.asarray(
            [value for value in held_out_rms if np.isfinite(value)],
            dtype=np.float64,
        )

        if len(finite):
            print("\nLeave-one-out held-out RMS:")
            print(f"  median : {np.median(finite):.3f} mm")
            print(f"  mean   : {np.mean(finite):.3f} mm")
            print(f"  max    : {np.max(finite):.3f} mm")

    print("==============================================\n")


def save_calibration(
    result: dict,
    observations: Sequence[Observation],
    held_out_rms: Optional[Sequence[float]] = None,
    use_depth_scale: bool = False,
    path: Path = CALIBRATION_PATH,
) -> None:
    """Save the calibration in NPZ, XML, JSON, and text formats."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]

    per_pose_rms = np.array(
        [pose["rms_mm"] for pose in metrics["per_pose"]],
        dtype=np.float64,
    )

    per_pose_p95 = np.array(
        [pose["p95_abs_mm"] for pose in metrics["per_pose"]],
        dtype=np.float64,
    )

    per_pose_max = np.array(
        [pose["max_abs_mm"] for pose in metrics["per_pose"]],
        dtype=np.float64,
    )

    per_pose_normal = np.array(
        [pose["normal_angle_deg"] for pose in metrics["per_pose"]],
        dtype=np.float64,
    )

    per_pose_offset = np.array(
        [pose["offset_error_mm"] for pose in metrics["per_pose"]],
        dtype=np.float64,
    )

    held_out = (
        np.asarray(held_out_rms, dtype=np.float64)
        if held_out_rms is not None
        else np.empty(0, dtype=np.float64)
    )

    np.savez_compressed(
        path,
        rotation_depth_to_rgb=result["rotation"],
        translation_depth_to_rgb_mm=result["translation"].reshape(3, 1),
        depth_scale=np.array(result["depth_scale"], dtype=np.float64),
        residual_rms_mm=np.array(metrics["rms_mm"], dtype=np.float64),
        residual_median_mm=np.array(
            metrics["median_abs_mm"], dtype=np.float64
        ),
        residual_p95_mm=np.array(
            metrics["p95_abs_mm"], dtype=np.float64
        ),
        residual_max_mm=np.array(
            metrics["max_abs_mm"], dtype=np.float64
        ),
        num_poses=np.array(len(observations), dtype=np.int64),
        per_pose_rms_mm=per_pose_rms,
        per_pose_p95_mm=per_pose_p95,
        per_pose_max_mm=per_pose_max,
        per_pose_normal_angle_deg=per_pose_normal,
        per_pose_offset_error_mm=per_pose_offset,
        held_out_rms_mm=held_out,
        use_depth_scale=np.array(bool(use_depth_scale), dtype=bool),
        initial_guess=np.array(result["initial_guess"]),
        num_starts=np.array(result["num_starts"], dtype=np.int64),
        board_columns=np.array(BOARD_COLUMNS, dtype=np.int64),
        board_rows=np.array(BOARD_ROWS, dtype=np.int64),
        square_size_mm=np.array(SQUARE_SIZE_MM, dtype=np.float64),
    )

    # OpenCV XML for tools that prefer it.
    xml_path = path.with_suffix(".xml")

    storage = cv2.FileStorage(
        str(xml_path), cv2.FILE_STORAGE_WRITE
    )

    storage.write("rotation_depth_to_rgb", result["rotation"])
    storage.write(
        "translation_depth_to_rgb_mm",
        result["translation"].reshape(3, 1),
    )
    storage.write("depth_scale", float(result["depth_scale"]))
    storage.write("residual_rms_mm", float(metrics["rms_mm"]))
    storage.write("num_poses", int(len(observations)))
    storage.write("use_depth_scale", int(bool(use_depth_scale)))
    storage.write("board_columns", int(BOARD_COLUMNS))
    storage.write("board_rows", int(BOARD_ROWS))
    storage.write("square_size_mm", float(SQUARE_SIZE_MM))
    storage.release()

    # JSON summary.
    summary = {
        "rotation_depth_to_rgb": result["rotation"].tolist(),
        "translation_depth_to_rgb_mm": result["translation"].tolist(),
        "depth_scale": float(result["depth_scale"]),
        "residual_rms_mm": float(metrics["rms_mm"]),
        "residual_median_mm": float(metrics["median_abs_mm"]),
        "residual_p95_mm": float(metrics["p95_abs_mm"]),
        "residual_max_mm": float(metrics["max_abs_mm"]),
        "num_poses": int(len(observations)),
        "use_depth_scale": bool(use_depth_scale),
        "initial_guess": result["initial_guess"],
        "num_starts": int(result["num_starts"]),
        "per_pose": metrics["per_pose"],
        "held_out_rms_mm": held_out.tolist(),
        "board": {
            "columns": int(BOARD_COLUMNS),
            "rows": int(BOARD_ROWS),
            "square_size_mm": float(SQUARE_SIZE_MM),
            "pattern_size_mm": [
                float((BOARD_COLUMNS + 1) * SQUARE_SIZE_MM),
                float((BOARD_ROWS + 1) * SQUARE_SIZE_MM),
            ],
        },
    }

    save_json(path.with_suffix(".json"), summary)

    # Human-readable report.
    lines = [
        "RGB-D CALIBRATION REPORT",
        "========================",
        "",
        f"Poses: {len(observations)}",
        f"Depth scale calibrated: {bool(use_depth_scale)}",
        f"Depth scale: {result['depth_scale']:.8f}",
        "",
        f"Point-to-plane RMS: {metrics['rms_mm']:.4f} mm",
        f"Median absolute residual: {metrics['median_abs_mm']:.4f} mm",
        f"95th percentile residual: {metrics['p95_abs_mm']:.4f} mm",
        f"Maximum absolute residual: {metrics['max_abs_mm']:.4f} mm",
        "",
        "Rotation depth -> RGB:",
        np.array2string(result["rotation"], precision=8),
        "",
        "Translation depth -> RGB (mm):",
        np.array2string(result["translation"], precision=6),
        "",
        "Per-pose metrics:",
        "  idx  points    RMS      p95      max    normal   offset",
    ]

    for pose in metrics["per_pose"]:
        lines.append(
            f"  {pose['index']:02d}  "
            f"{pose['num_points']:6d}  "
            f"{pose['rms_mm']:7.3f}  "
            f"{pose['p95_abs_mm']:7.3f}  "
            f"{pose['max_abs_mm']:7.3f}  "
            f"{pose['normal_angle_deg']:7.2f}  "
            f"{pose['offset_error_mm']:7.2f}"
        )

    if held_out.size:
        finite = held_out[np.isfinite(held_out)]

        if finite.size:
            lines.extend(
                [
                    "",
                    "Leave-one-out held-out RMS (mm):",
                    f"  median: {np.median(finite):.4f}",
                    f"  mean:   {np.mean(finite):.4f}",
                    f"  max:    {np.max(finite):.4f}",
                ]
            )

    report_path = path.parent / f"{path.stem}_report.txt"

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved calibration to:")
    print(f"  {path}")
    print(f"  {xml_path}")
    print(f"  {path.with_suffix('.json')}")
    print(f"  {report_path}")


# ---------------------------------------------------------------------------
# Validation command
# ---------------------------------------------------------------------------


def validate_saved_calibration(
    observations_path: Path = OBSERVATIONS_PATH,
    calibration_path: Path = CALIBRATION_PATH,
    run_leave_one_out: bool = True,
) -> None:
    observations = load_observations(observations_path)

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}"
        )

    with np.load(calibration_path) as data:
        rotation = np.asarray(
            data["rotation_depth_to_rgb"], dtype=np.float64
        ).reshape(3, 3)

        translation = np.asarray(
            data["translation_depth_to_rgb_mm"], dtype=np.float64
        ).reshape(3)

        depth_scale = float(data["depth_scale"]) if "depth_scale" in data.files else 1.0

        use_depth_scale = (
            bool(data["use_depth_scale"])
            if "use_depth_scale" in data.files
            else False
        )

    rvec = Rotation.from_matrix(rotation).as_rotvec()

    parameters = np.concatenate([rvec, translation])

    if use_depth_scale:
        parameters = np.concatenate(
            [parameters, [np.log(depth_scale)]]
        )

    metrics = evaluate_solution(
        parameters, observations, use_depth_scale
    )

    held_out = None

    if run_leave_one_out:
        print("\nRunning leave-one-out validation...")

        held_out = leave_one_out_validation(
            observations,
            parameters,
            use_depth_scale,
            verbose=True,
        )

    print_metrics(metrics, held_out)


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------


def _random_board_pose(
    rng: np.random.Generator,
    distance_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a board pose in RGB coordinates that faces the camera.

    The board normal is built from the camera-to-board direction with a
    random tilt of up to 35 degrees, so the board always faces the camera
    at a realistic angle instead of accidentally becoming edge-on.
    """

    translation = np.array(
        [
            rng.uniform(-0.25, 0.25) * float(distance_mm),
            rng.uniform(-0.20, 0.20) * float(distance_mm),
            float(distance_mm),
        ],
        dtype=np.float64,
    )

    view_direction = translation / np.linalg.norm(translation)

    # Random axis perpendicular to the view direction.
    axis = rng.normal(size=3)
    axis -= view_direction * float(np.dot(axis, view_direction))
    axis_length = float(np.linalg.norm(axis))

    if axis_length < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis /= axis_length

    tilt = np.radians(rng.uniform(0.0, 35.0))

    normal = Rotation.from_rotvec(axis * tilt).apply(view_direction)

    # Build a right-handed frame whose third column is the board normal.
    up = np.array([0.0, 1.0, 0.0])

    if abs(float(np.dot(up, normal))) > 0.95:
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, normal)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation = np.column_stack([x_axis, y_axis, normal])

    # Random in-plane rotation around the board normal.
    yaw = np.radians(rng.uniform(-25.0, 25.0))

    rotation = rotation @ Rotation.from_euler(
        "z", yaw
    ).as_matrix()

    return rotation, translation


def _synthetic_board_surface_points(
    rotation: np.ndarray,
    translation: np.ndarray,
    columns: int = 24,
    rows: int = 16,
) -> np.ndarray:
    corners = pattern_outer_corners_mm()

    min_x = float(corners[:, 0].min())
    max_x = float(corners[:, 0].max())
    min_y = float(corners[:, 1].min())
    max_y = float(corners[:, 1].max())

    xs = np.linspace(min_x, max_x, columns)
    ys = np.linspace(min_y, max_y, rows)

    grid_x, grid_y = np.meshgrid(xs, ys)

    board_points = np.column_stack(
        [
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.zeros(grid_x.size, dtype=np.float64),
        ]
    )

    return transform_points(board_points, rotation, translation)


def run_selftest(
    num_poses: int = 20,
    use_depth_scale: bool = True,
    seed: int = 42,
) -> bool:
    """Validate the full pipeline on synthetic data."""

    print("\n==============================================")
    print(" RGB-D CALIBRATION SELF-TEST")
    print("==============================================")

    rng = np.random.default_rng(seed)

    try:
        rgb_intrinsics = load_rgb_intrinsics()
        depth_intrinsics = load_depth_intrinsics()
    except FileNotFoundError:
        rgb_intrinsics = CameraIntrinsics(
            np.array(
                [
                    [1060.5, 0.0, 663.9],
                    [0.0, 1060.7, 519.3],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array([0.057, -0.262, 0.003, 0.003, 0.241]),
            1280,
            1024,
        )

        depth_intrinsics = CameraIntrinsics(
            np.array(
                [
                    [569.36, 0.0, 319.5],
                    [0.0, 568.98, 239.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.zeros(5),
            640,
            480,
        )

    # Ground truth: a plausible Xtion RGB-D transform.
    true_rotation = Rotation.from_euler(
        "xyz", [0.4, -1.2, 0.3], degrees=True
    ).as_matrix()

    true_translation = np.array([-27.0, 29.0, -15.0], dtype=np.float64)

    true_scale = 1.003 if use_depth_scale else 1.0

    object_points = create_object_points()

    observations: list[Observation] = []

    print(
        f"Generating {num_poses} synthetic poses with "
        f"1 mm depth noise and 0.3 px RGB noise..."
    )

    for index in range(num_poses):
        distance = rng.uniform(600.0, 1100.0)

        board_rotation, board_translation = _random_board_pose(
            rng, distance
        )

        # RGB corner measurements.
        board_corners_rgb = transform_points(
            object_points.astype(np.float64),
            board_rotation,
            board_translation,
        )

        image_corners = project_points(
            board_corners_rgb,
            rgb_intrinsics.camera_matrix,
            rgb_intrinsics.distortion,
        )

        image_corners += rng.normal(
            scale=0.3, size=image_corners.shape
        )

        pose = solve_board_pose(
            image_corners.reshape(-1, 1, 2).astype(np.float32),
            rgb_intrinsics.camera_matrix,
            rgb_intrinsics.distortion,
        )

        if pose is None:
            print(f"  pose {index}: PnP failed")
            return False

        # Synthetic depth points on the board surface.
        surface_rgb = _synthetic_board_surface_points(
            board_rotation, board_translation
        )

        # Add a few points outside the board to exercise the plane selector.
        outlier_count = max(20, int(0.15 * len(surface_rgb)))

        outliers = np.column_stack(
            [
                rng.uniform(-250.0, 250.0, outlier_count),
                rng.uniform(-200.0, 200.0, outlier_count),
                rng.uniform(500.0, 1200.0, outlier_count),
            ]
        )

        surface_rgb = np.vstack([surface_rgb, outliers])

        rotation_inv, translation_inv = invert_transform(
            true_rotation, true_translation
        )

        surface_depth = (
            surface_rgb @ rotation_inv.T + translation_inv
        ) / true_scale

        surface_depth += rng.normal(
            scale=1.0, size=surface_depth.shape
        )

        predicted_normal = pose.normal

        predicted_offset = pose.offset

        normal, offset, residuals, inliers = select_plane(
            surface_depth,
            predicted_normal=predicted_normal,
            predicted_offset=predicted_offset,
            normal_tolerance_deg=35.0,
            distance_threshold_mm=8.0,
            seed_threshold_mm=60.0,
            max_iterations=1200,
            min_inliers=50,
        )

        inlier_points = surface_depth[inliers]

        if len(inlier_points) > MAX_POINTS_PER_POSE:
            selection = rng.choice(
                len(inlier_points),
                size=MAX_POINTS_PER_POSE,
                replace=False,
            )
            inlier_points = inlier_points[selection]

        plane_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))

        observations.append(
            Observation(
                points=inlier_points,
                rgb_normal=pose.normal,
                rgb_offset=pose.offset,
                depth_normal=normal,
                depth_offset=offset,
                rgb_rvec=pose.rvec,
                rgb_tvec=pose.tvec,
                reprojection_rms_px=pose.reprojection_rms_px,
                plane_rms_mm=plane_rms,
                board_distance_mm=pose.distance_mm,
                rgb_image=f"synthetic_{index:03d}",
                depth_image=f"synthetic_{index:03d}",
            )
        )

    print("Optimising synthetic observations...")

    result = optimize_extrinsics(
        observations,
        use_depth_scale=use_depth_scale,
        multistart=True,
        verbose=True,
    )

    rotation_error_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    (
                        np.trace(result["rotation"] @ true_rotation.T)
                        - 1.0
                    )
                    / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
    )

    translation_error_mm = float(
        np.linalg.norm(result["translation"] - true_translation)
    )

    scale_error = float(abs(result["depth_scale"] - true_scale))

    metrics = result["metrics"]

    print("\nSelf-test result:")
    print(f"  rotation error    : {rotation_error_deg:.5f} deg")
    print(f"  translation error : {translation_error_mm:.5f} mm")
    print(f"  depth scale error : {scale_error:.8f}")
    print(f"  residual RMS      : {metrics['rms_mm']:.5f} mm")
    print(f"  residual p95      : {metrics['p95_abs_mm']:.5f} mm")

    passed = (
        rotation_error_deg < 0.25
        and translation_error_mm < 2.0
        and scale_error < 0.01
        and metrics["rms_mm"] < 4.0
    )

    print("\nSELF-TEST:", "PASS" if passed else "FAIL")
    print("==============================================\n")

    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _optimise_and_save(
    observations: Sequence[Observation],
    use_depth_scale: bool,
    run_leave_one_out: bool,
) -> dict:
    print("\nOptimising RGB-D extrinsics...")
    print(f"Poses: {len(observations)}")
    print(f"Depth scale calibrated: {use_depth_scale}")

    result = optimize_extrinsics(
        observations,
        use_depth_scale=use_depth_scale,
        multistart=True,
        verbose=True,
    )

    held_out = None

    if run_leave_one_out:
        print("\nRunning leave-one-out validation...")

        held_out = leave_one_out_validation(
            observations,
            result["parameters"],
            use_depth_scale,
            verbose=True,
        )

    print_metrics(result["metrics"], held_out)

    print("Rotation depth -> RGB:")
    print(result["rotation"])

    print("\nTranslation depth -> RGB (mm):")
    print(result["translation"])

    print(f"\nDepth scale: {result['depth_scale']:.8f}")

    save_calibration(
        result,
        observations,
        held_out_rms=held_out,
        use_depth_scale=use_depth_scale,
    )

    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reliable RGB-D extrinsic calibration"
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "capture",
            "optimize",
            "all",
            "validate",
            "selftest",
        ],
        help="Action to perform (default: all)",
    )

    parser.add_argument(
        "--depth-scale",
        action="store_true",
        help="Also estimate a uniform depth scale",
    )

    parser.add_argument(
        "--no-leave-one-out",
        action="store_true",
        help="Skip leave-one-out validation",
    )

    parser.add_argument(
        "--observations",
        type=Path,
        default=OBSERVATIONS_PATH,
        help="Observation file path",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_POSES,
        help="Target number of capture poses",
    )

    parser.add_argument(
        "--selftest-poses",
        type=int,
        default=30,
        help="Number of synthetic poses for selftest",
    )

    args = parser.parse_args(argv)

    if args.command == "selftest":
        passed = run_selftest(
            num_poses=args.selftest_poses,
            use_depth_scale=True,
        )

        return 0 if passed else 1

    if args.command == "validate":
        try:
            validate_saved_calibration(
                observations_path=args.observations,
                run_leave_one_out=not args.no_leave_one_out,
            )
        except FileNotFoundError as error:
            print(f"\nValidation impossible: {error}")
            print(
                "Capture RGB-D observations first with:\n"
                "  python rgbd_calibrate.py capture\n"
                "then optimise with:\n"
                "  python rgbd_calibrate.py optimize"
            )
            return 1

        return 0

    if args.command == "optimize":
        try:
            observations = load_observations(args.observations)
        except FileNotFoundError as error:
            print(f"\nCannot optimise: {error}")
            print(
                "Capture RGB-D observations first with:\n"
                "  python rgbd_calibrate.py capture"
            )
            return 1

        if len(observations) < MIN_POSES:
            print(
                f"Only {len(observations)} observations available. "
                f"At least {MIN_POSES} are recommended."
            )

        if len(observations) < 3:
            print("Not enough observations to optimise.")
            return 1

        _optimise_and_save(
            observations,
            use_depth_scale=args.depth_scale,
            run_leave_one_out=not args.no_leave_one_out,
        )

        return 0

    # capture or all
    observations = capture_observations(
        target_poses=args.target,
        min_poses=MIN_POSES,
    )

    if args.command == "capture":
        return 0

    if len(observations) < 3:
        print("Not enough observations to optimise.")
        return 1

    _optimise_and_save(
        observations,
        use_depth_scale=args.depth_scale,
        run_leave_one_out=not args.no_leave_one_out,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
