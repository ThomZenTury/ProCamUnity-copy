import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = PROJECT_ROOT / "calibration_data" / "projector"

CORRESPONDENCE_FILE = (
    DATASET / "projector_correspondences.npz"
)

OUTPUT = PROJECT_ROOT / "coordinate_system_diagnostics"

OUTPUT.mkdir(parents=True, exist_ok=True)

WIDTH = 1920
HEIGHT = 1080

# Candidate Model B
K_B = np.array([
    [934.4554, 0, 938.4490],
    [0, 934.4554, -149.0770],
    [0, 0, 1]
])


# ============================================================
# LOAD CORRESPONDENCES
# ============================================================

def load_correspondences():

    if not CORRESPONDENCE_FILE.exists():
        raise FileNotFoundError(CORRESPONDENCE_FILE)

    archive = np.load(CORRESPONDENCE_FILE)

    print("\nAVAILABLE NPZ ARRAYS")

    for key in archive.files:
        print(key, archive[key].shape, archive[key].dtype)

    required = [
        "object_points",
        "projector_points",
        "valid",
        "pose_names"
    ]

    missing = [
        key for key in required
        if key not in archive.files
    ]

    if missing:
        raise RuntimeError(
            f"Missing correspondence arrays: {missing}"
        )

    return {
        key: archive[key].copy()
        for key in required
    }


# ============================================================
# COORDINATE STATISTICS
# ============================================================

def coordinate_statistics(points):

    x = points[:, 0]
    y = points[:, 1]

    return {

        "count": int(len(points)),

        "x_min": float(x.min()),
        "x_max": float(x.max()),

        "y_min": float(y.min()),
        "y_max": float(y.max()),

        "x_mean": float(x.mean()),
        "y_mean": float(y.mean()),

        "x_std": float(x.std()),
        "y_std": float(y.std()),

        "inside_projector": int(
            np.sum(
                (x >= 0) &
                (x < WIDTH) &
                (y >= 0) &
                (y < HEIGHT)
            )
        )

    }


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def calculate_coverage(points):

    grid_columns = 12
    grid_rows = 8

    occupancy = np.zeros(
        (grid_rows, grid_columns),
        dtype=np.int32
    )

    x = points[:, 0]
    y = points[:, 1]

    gx = np.clip(
        (x / WIDTH * grid_columns).astype(int),
        0,
        grid_columns - 1
    )

    gy = np.clip(
        (y / HEIGHT * grid_rows).astype(int),
        0,
        grid_rows - 1
    )

    np.add.at(
        occupancy,
        (gy, gx),
        1
    )

    occupied = int(np.count_nonzero(occupancy))

    return {

        "grid": occupancy.tolist(),

        "occupied_cells": occupied,

        "total_cells": grid_rows * grid_columns,

        "coverage_percentage": (
            100.0 * occupied /
            (grid_rows * grid_columns)
        )

    }


# ============================================================
# CONVEX HULL COVERAGE
# ============================================================

def calculate_hull_coverage(points):

    points = points.astype(np.float32)

    if len(points) < 3:
        return 0.0

    hull = cv2.convexHull(points)

    area = cv2.contourArea(hull)

    return float(
        100.0 * area / (WIDTH * HEIGHT)
    )


# ============================================================
# INSPECT DECODED CAPTURES
# ============================================================

def inspect_decoded_captures():

    results = {}

    folders = sorted(
        DATASET.glob("pose_*")
    )

    for folder in folders:

        if not folder.is_dir():
            continue

        path = folder / "decoded_projector.npz"

        if not path.exists():
            continue

        with np.load(path) as data:

            px = data["projector_x"]
            py = data["projector_y"]
            valid = data["valid"].astype(bool)

        if not valid.any():
            continue

        x = px[valid]
        y = py[valid]

        results[folder.name] = {

            "camera_image_shape": list(px.shape),

            "valid_pixels": int(valid.sum()),

            "x_min": float(x.min()),
            "x_max": float(x.max()),

            "y_min": float(y.min()),
            "y_max": float(y.max()),

            "x_mean": float(x.mean()),
            "y_mean": float(y.mean()),

            "out_of_bounds": int(
                np.sum(
                    (x < 0) |
                    (x >= WIDTH) |
                    (y < 0) |
                    (y >= HEIGHT)
                )
            )

        }

    return results


# ============================================================
# CREATE COVERAGE VISUALIZATION
# ============================================================

def create_coverage_plot(points):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 7)
    )

    # --------------------------------------------------------
    # Projector coordinates
    # --------------------------------------------------------

    ax = axes[0]

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=8,
        alpha=0.5
    )

    ax.scatter(
        K_B[0, 2],
        K_B[1, 2],
        color="red",
        marker="x",
        s=150,
        linewidths=3,
        label="Model B principal point"
    )

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)

    ax.set_aspect("equal")

    ax.set_title(
        "Projector correspondence coverage"
    )

    ax.set_xlabel("Projector X")
    ax.set_ylabel("Projector Y")

    ax.legend()

    ax.grid(alpha=0.2)

    # --------------------------------------------------------
    # Density map
    # --------------------------------------------------------

    ax = axes[1]

    histogram = ax.hist2d(
        points[:, 0],
        points[:, 1],
        bins=[48, 27],
        range=[
            [0, WIDTH],
            [0, HEIGHT]
        ],
        cmap="viridis"
    )

    ax.scatter(
        K_B[0, 2],
        K_B[1, 2],
        color="red",
        marker="x",
        s=150,
        linewidths=3
    )

    ax.set_title(
        "Projector correspondence density"
    )

    ax.set_xlabel("Projector X")
    ax.set_ylabel("Projector Y")

    ax.invert_yaxis()

    fig.colorbar(
        histogram[3],
        ax=ax,
        label="Correspondence count"
    )

    plt.tight_layout()

    output = OUTPUT / "projector_coordinate_coverage.png"

    plt.savefig(
        output,
        dpi=180
    )

    plt.close()

    print("Saved:", output)


# ============================================================
# CREATE PER-POSE VISUALIZATION
# ============================================================

def create_pose_plot(points, names):

    unique_names = sorted(
        set(names)
    )

    fig, axes = plt.subplots(
        3, 5,
        figsize=(19, 12)
    )

    axes = axes.flatten()

    for index, name in enumerate(unique_names):

        ax = axes[index]

        mask = np.array([
            n == name for n in names
        ])

        pose_points = points[mask]

        ax.scatter(
            pose_points[:, 0],
            pose_points[:, 1],
            s=12
        )

        ax.set_xlim(0, WIDTH)
        ax.set_ylim(HEIGHT, 0)

        ax.set_aspect("equal")

        ax.set_title(
            f"{name} ({len(pose_points)} points)"
        )

        ax.grid(alpha=0.2)

    for index in range(len(unique_names), len(axes)):
        axes[index].axis("off")

    plt.tight_layout()

    output = OUTPUT / "coverage_per_pose.png"

    plt.savefig(
        output,
        dpi=180
    )

    plt.close()

    print("Saved:", output)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPROJECTOR COORDINATE SYSTEM DIAGNOSTICS")

    print("=" * 65)

    print("Dataset:", DATASET)

    data = load_correspondences()

    object_points = data["object_points"]
    projector_points = data["projector_points"]
    valid = data["valid"].astype(bool)

    pose_names = [
        str(name) for name in data["pose_names"]
    ]

    all_points = []
    all_names = []

    per_pose = {}

    # --------------------------------------------------------
    # PROCESS EACH POSE
    # --------------------------------------------------------

    for i, name in enumerate(pose_names):

        mask = valid[i]

        obj = object_points[i][mask]
        img = projector_points[i][mask]

        finite = (
            np.isfinite(obj).all(axis=1)
            &
            np.isfinite(img).all(axis=1)
        )

        img = img[finite]

        if len(img) == 0:
            continue

        stats = coordinate_statistics(img)

        stats["hull_coverage_percentage"] = (
            calculate_hull_coverage(img)
        )

        per_pose[name] = stats

        all_points.append(img)

        all_names.extend(
            [name] * len(img)
        )

        print(
            f"{name:12s} "
            f"X={stats['x_min']:.1f}..{stats['x_max']:.1f} "
            f"Y={stats['y_min']:.1f}..{stats['y_max']:.1f} "
            f"N={len(img)}"
        )

    points = np.concatenate(
        all_points,
        axis=0
    )

    # --------------------------------------------------------
    # GLOBAL STATISTICS
    # --------------------------------------------------------

    print("\nGLOBAL COORDINATE STATISTICS")

    global_stats = coordinate_statistics(points)

    for key, value in global_stats.items():
        print(f"{key}: {value}")

    coverage = calculate_coverage(points)

    hull_coverage = calculate_hull_coverage(points)

    print("\nSPATIAL COVERAGE")

    print(
        "Occupied grid cells:",
        coverage["occupied_cells"],
        "/",
        coverage["total_cells"]
    )

    print(
        "Grid coverage:",
        f"{coverage['coverage_percentage']:.2f}%"
    )

    print(
        "Convex hull coverage:",
        f"{hull_coverage:.2f}%"
    )

    # --------------------------------------------------------
    # PRINCIPAL POINT
    # --------------------------------------------------------

    print("\nMODEL B PRINCIPAL POINT")

    cx = K_B[0, 2]
    cy = K_B[1, 2]

    print("cx:", cx)
    print("cy:", cy)

    print(
        "Distance from image centre:",
        np.sqrt(
            (cx - WIDTH / 2) ** 2 +
            (cy - HEIGHT / 2) ** 2
        )
    )

    # --------------------------------------------------------
    # DECODED CAPTURE INSPECTION
    # --------------------------------------------------------

    decoded = inspect_decoded_captures()

    print("\nDECODED CAPTURES")

    for name, stats in decoded.items():

        print(
            f"{name}: "
            f"X={stats['x_min']:.0f}..{stats['x_max']:.0f}, "
            f"Y={stats['y_min']:.0f}..{stats['y_max']:.0f}"
        )

    # --------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------

    create_coverage_plot(points)

    create_pose_plot(
        points,
        all_names
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report = {

        "projector_resolution": [
            WIDTH,
            HEIGHT
        ],

        "global_statistics": global_stats,

        "grid_coverage": coverage,

        "convex_hull_coverage_percentage": hull_coverage,

        "model_B": {

            "camera_matrix": K_B.tolist(),

            "principal_point": [
                float(cx),
                float(cy)
            ],

            "principal_point_inside_image": bool(
                0 <= cx < WIDTH and
                0 <= cy < HEIGHT
            )

        },

        "per_pose": per_pose,

        "decoded_captures": decoded

    }

    report_path = (
        OUTPUT /
        "coordinate_system_report.json"
    )

    with open(report_path, "w") as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print("\n")
    print("=" * 65)

    print("DIAGNOSTICS COMPLETE")

    print("=" * 65)

    print("Output:", OUTPUT)

    print("JSON:", report_path)

    print("=" * 65)


if __name__ == "__main__":
    main()
