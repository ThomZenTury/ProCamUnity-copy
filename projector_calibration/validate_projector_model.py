import sys
import json
import csv

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAPTURE_ROOT = (
    PROJECT_ROOT /
    "calibration_data" /
    "projector"
)

INPUT_NPZ = CAPTURE_ROOT / "projector_correspondences.npz"

OUTPUT_DIR = CAPTURE_ROOT / "intrinsic_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 1920
HEIGHT = 1080

IMAGE_SIZE = (WIDTH, HEIGHT)

MINIMUM_CORNERS = 12

# Model D: k1 only
MODEL_D_FLAGS = (
    cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
)

# Model C: k1, k2, k3
MODEL_C_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST

MODELS = {
    "D_simple": MODEL_D_FLAGS,
    "C_radial_only": MODEL_C_FLAGS,
}


# ============================================================
# LOAD CORRESPONDENCES
# ============================================================

def load_data():

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
# CALIBRATION FUNCTION
# ============================================================

def calibrate(poses, flags):

    object_points = [
        p["object"] for p in poses
    ]

    image_points = [
        p["image"] for p in poses
    ]

    initial_matrix = cv2.initCameraMatrix2D(
        object_points,
        image_points,
        IMAGE_SIZE,
        0
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_MAX_ITER,
        300,
        1e-10
    )

    result = cv2.calibrateCamera(
        object_points,
        image_points,
        IMAGE_SIZE,
        initial_matrix,
        np.zeros((5, 1), dtype=np.float64),
        flags=(
            flags
            | cv2.CALIB_USE_INTRINSIC_GUESS
        ),
        criteria=criteria
    )

    rms, K, D, rvecs, tvecs = result

    return {

        "rms": float(rms),

        "K": K,

        "D": D,

        "rvecs": rvecs,

        "tvecs": tvecs

    }


# ============================================================
# REPROJECTION RESIDUALS
# ============================================================

def calculate_residuals(
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

    observed = pose["image"].reshape(-1, 2)

    projected = projected.reshape(-1, 2)

    residual = projected - observed

    magnitude = np.linalg.norm(
        residual,
        axis=1
    )

    return residual, magnitude


# ============================================================
# PROJECTOR IMAGE COVERAGE
# ============================================================

def plot_coverage(poses, K):

    all_points = np.concatenate([
        p["image"].reshape(-1, 2)
        for p in poses
    ])

    x = all_points[:, 0]

    y = all_points[:, 1]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 8)
    )

    # --------------------------------------------------------
    # LEFT: CORRESPONDENCE LOCATIONS
    # --------------------------------------------------------

    ax = axes[0]

    ax.scatter(
        x,
        y,
        s=12,
        alpha=0.55,
        color="royalblue"
    )

    ax.scatter(
        K[0, 2],
        K[1, 2],
        marker="x",
        s=180,
        linewidths=3,
        color="red",
        label="Estimated principal point"
    )

    ax.scatter(
        WIDTH / 2,
        HEIGHT / 2,
        marker="+",
        s=180,
        linewidths=3,
        color="limegreen",
        label="Image centre"
    )

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)

    ax.set_aspect("equal")

    ax.set_title("Projector correspondence coverage")

    ax.set_xlabel("Projector X (pixels)")
    ax.set_ylabel("Projector Y (pixels)")

    ax.legend()

    ax.grid(alpha=0.2)

    # --------------------------------------------------------
    # RIGHT: SPATIAL DENSITY
    # --------------------------------------------------------

    ax = axes[1]

    density = ax.hist2d(
        x,
        y,
        bins=(48, 27),
        range=[
            [0, WIDTH],
            [0, HEIGHT]
        ],
        cmap="viridis"
    )

    fig.colorbar(
        density[3],
        ax=ax,
        label="Number of observations"
    )

    ax.scatter(
        K[0, 2],
        K[1, 2],
        marker="x",
        s=180,
        linewidths=3,
        color="red"
    )

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)

    ax.set_aspect("equal")

    ax.set_title("Correspondence spatial density")

    ax.set_xlabel("Projector X (pixels)")
    ax.set_ylabel("Projector Y (pixels)")

    fig.tight_layout()

    output = OUTPUT_DIR / "projector_coverage.png"

    fig.savefig(output, dpi=160)

    plt.close(fig)

    print("Saved:", output)

    # Coverage statistics

    print("\nPROJECTOR COVERAGE STATISTICS")

    print("-" * 50)

    print(
        f"X range: {x.min():.1f} - {x.max():.1f}"
    )

    print(
        f"Y range: {y.min():.1f} - {y.max():.1f}"
    )

    print(
        f"X coverage: "
        f"{100 * (x.max() - x.min()) / WIDTH:.1f}%"
    )

    print(
        f"Y coverage: "
        f"{100 * (y.max() - y.min()) / HEIGHT:.1f}%"
    )

    print(
        "Correspondence count:",
        len(all_points)
    )

    return all_points


# ============================================================
# LEAVE-ONE-POSE-OUT PARAMETER STABILITY
# ============================================================

def analyse_stability(poses, model_name, flags):

    print("\nPARAMETER STABILITY:", model_name)

    print("-" * 65)

    records = []

    for i in range(len(poses)):

        training = [
            p for j, p in enumerate(poses)
            if i != j
        ]

        try:

            result = calibrate(
                training,
                flags
            )

            K = result["K"]

            D = result["D"].flatten()

            record = {

                "excluded_pose": poses[i]["name"],

                "rms": result["rms"],

                "fx": float(K[0, 0]),

                "fy": float(K[1, 1]),

                "cx": float(K[0, 2]),

                "cy": float(K[1, 2]),

                "k1": float(D[0]),

                "k2": float(D[1]),

                "p1": float(D[2]),

                "p2": float(D[3]),

                "k3": float(D[4])

            }

            records.append(record)

            print(
                f"{poses[i]['name']:12s} "
                f"fx={record['fx']:9.2f} "
                f"fy={record['fy']:9.2f} "
                f"cx={record['cx']:8.2f} "
                f"cy={record['cy']:8.2f} "
                f"k1={record['k1']:8.4f}"
            )

        except cv2.error as e:

            print(
                f"{poses[i]['name']}: FAILED: {e}"
            )

    return records


# ============================================================
# STABILITY SUMMARY
# ============================================================

def stability_summary(records):

    parameters = [
        "fx",
        "fy",
        "cx",
        "cy",
        "k1"
    ]

    summary = {}

    print("\nINTRINSIC STABILITY SUMMARY")

    print("-" * 65)

    for parameter in parameters:

        values = np.array([
            r[parameter] for r in records
        ])

        stats = {

            "mean": float(np.mean(values)),

            "std": float(np.std(values)),

            "minimum": float(np.min(values)),

            "maximum": float(np.max(values)),

            "range": float(
                np.max(values) - np.min(values)
            )

        }

        summary[parameter] = stats

        print(
            f"{parameter:5s}: "
            f"mean={stats['mean']:10.3f} "
            f"std={stats['std']:9.3f} "
            f"range={stats['range']:9.3f}"
        )

    return summary


# ============================================================
# STABILITY PLOTS
# ============================================================

def plot_stability(records, model_name):

    parameters = [
        "fx",
        "fy",
        "cx",
        "cy",
        "k1"
    ]

    fig, axes = plt.subplots(
        5, 1,
        figsize=(13, 15)
    )

    names = [
        r["excluded_pose"] for r in records
    ]

    for ax, parameter in zip(
        axes,
        parameters
    ):

        values = [
            r[parameter] for r in records
        ]

        ax.plot(
            range(len(values)),
            values,
            "o-",
            color="royalblue"
        )

        ax.set_ylabel(parameter)

        ax.set_xticks(range(len(names)))

        ax.set_xticklabels(
            names,
            rotation=45
        )

        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Intrinsic stability: {model_name}"
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR /
        f"stability_{model_name}.png"
    )

    fig.savefig(output, dpi=160)

    plt.close(fig)

    print("Saved:", output)


# ============================================================
# RESIDUAL PLOTS
# ============================================================

def plot_residuals(poses, result):

    for i, pose in enumerate(poses):

        residual, magnitude = calculate_residuals(
            pose,
            result["K"],
            result["D"],
            result["rvecs"][i],
            result["tvecs"][i]
        )

        points = pose["image"].reshape(-1, 2)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6)
        )

        # ----------------------------------------------------
        # RESIDUAL MAGNITUDE
        # ----------------------------------------------------

        scatter = axes[0].scatter(
            points[:, 0],
            points[:, 1],
            c=magnitude,
            cmap="inferno",
            s=70
        )

        fig.colorbar(
            scatter,
            ax=axes[0],
            label="Residual (pixels)"
        )

        axes[0].set_title(
            f"{pose['name']} - residual magnitude"
        )

        # ----------------------------------------------------
        # RESIDUAL VECTORS
        # ----------------------------------------------------

        axes[1].quiver(
            points[:, 0],
            points[:, 1],
            residual[:, 0],
            -residual[:, 1],
            angles="xy",
            scale_units="xy",
            scale=0.02,
            color="red"
        )

        axes[1].scatter(
            points[:, 0],
            points[:, 1],
            s=15,
            color="black"
        )

        axes[1].set_title(
            f"{pose['name']} - residual vectors"
        )

        for ax in axes:

            ax.set_xlim(0, WIDTH)

            ax.set_ylim(HEIGHT, 0)

            ax.set_aspect("equal")

            ax.set_xlabel("Projector X")

            ax.set_ylabel("Projector Y")

            ax.grid(alpha=0.2)

        fig.tight_layout()

        output = (
            OUTPUT_DIR /
            f"residual_{pose['name']}.png"
        )

        fig.savefig(output, dpi=160)

        plt.close(fig)

    print("Saved residual plots for all poses.")


# ============================================================
# WRITE CSV
# ============================================================

def save_csv(all_records):

    output = OUTPUT_DIR / "stability_report.csv"

    with open(
        output,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_records[0].keys())
        )

        writer.writeheader()

        writer.writerows(all_records)

    print("Saved:", output)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nPROJECTOR MODEL VALIDATION")

    print("=" * 65)

    poses = load_data()

    print("Usable poses:", len(poses))

    print(
        "Total correspondences:",
        sum(len(p["object"]) for p in poses)
    )

    # --------------------------------------------------------
    # FULL DATASET CALIBRATION
    # --------------------------------------------------------

    full_results = {}

    stability_results = {}

    all_records = []

    for model_name, flags in MODELS.items():

        print("\n")
        print("=" * 65)

        print("MODEL:", model_name)

        print("=" * 65)

        result = calibrate(
            poses,
            flags
        )

        full_results[model_name] = result

        K = result["K"]

        D = result["D"].flatten()

        print("\nFull dataset calibration:")

        print("RMS:", result["rms"])

        print("Camera matrix:")

        print(K)

        print("Distortion:")

        print(D)

        print(
            "Principal point:",
            K[0, 2],
            K[1, 2]
        )

        print(
            "Principal point inside image:",
            (
                0 <= K[0, 2] < WIDTH
                and
                0 <= K[1, 2] < HEIGHT
            )
        )

        # ----------------------------------------------------
        # STABILITY
        # ----------------------------------------------------

        records = analyse_stability(
            poses,
            model_name,
            flags
        )

        stability_results[model_name] = (
            stability_summary(records)
        )

        all_records.extend([
            {
                "model": model_name,
                **record
            }
            for record in records
        ])

        plot_stability(
            records,
            model_name
        )

        # ----------------------------------------------------
        # RESIDUALS
        # ----------------------------------------------------

        if model_name == "D_simple":

            plot_residuals(
                poses,
                result
            )

            all_points = plot_coverage(
                poses,
                K
            )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    save_csv(all_records)

    report = {

        "projector_resolution": [
            WIDTH,
            HEIGHT
        ],

        "number_of_poses": len(poses),

        "models": {

            name: {

                "rms": result["rms"],

                "camera_matrix": result["K"].tolist(),

                "distortion": (
                    result["D"].flatten().tolist()
                ),

                "stability": stability_results[name]

            }

            for name, result in full_results.items()

        }

    }

    json_path = (
        OUTPUT_DIR /
        "validation_report.json"
    )

    with open(json_path, "w") as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print("\n")
    print("=" * 65)

    print("PROJECTOR VALIDATION COMPLETE")

    print("=" * 65)

    print("Output directory:")

    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.iterdir()):

        print(" ", path.name)

    print("=" * 65)


if __name__ == "__main__":

    main()
