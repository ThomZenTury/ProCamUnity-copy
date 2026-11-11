import cv2
import numpy as np
import openni
from openni import openni2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pathlib import Path

from xtion import XtionCamera


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_WIDTH = 8
BOARD_HEIGHT = 5
SQUARE_SIZE = 29.0
WHITE_BORDER = 20.0

NUM_POSES_TARGET = 15

OUTPUT_DIR = Path("calibration_data/rgbd_extrinsics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RGB intrinsics
K_RGB = np.array([
    [1060.53706, 0.0, 663.877816],
    [0.0, 1060.68969, 519.283264],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

D_RGB = np.array([
    0.05706026,
    -0.26159217,
    0.00266085,
    0.00262938,
    0.24140464
], dtype=np.float64)

# Depth intrinsics
K_DEPTH = np.array([
    [569.35884766, 0.0, 319.5],
    [0.0, 568.98424650, 239.5],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# OpenCV checkerboard object points
OBJECT_POINTS = np.zeros(
    (BOARD_WIDTH * BOARD_HEIGHT, 3),
    dtype=np.float32
)

OBJECT_POINTS[:, :2] = np.mgrid[
    0:BOARD_WIDTH,
    0:BOARD_HEIGHT
].T.reshape(-1, 2)

OBJECT_POINTS *= SQUARE_SIZE

# Outer checkerboard corners.
#
# 8 x 5 internal corners
# 29 mm squares
# 20 mm white border
#
# Board dimensions:
# Width:  9 * 29 + 40 = 301 mm
# Height: 6 * 29 + 40 = 214 mm

OUTER_CORNERS_OBJECT = np.array([
    [-49.0, -49.0, 0.0],   # A
    [252.0, -49.0, 0.0],   # B
    [252.0, 165.0, 0.0],   # C
    [-49.0, 165.0, 0.0]    # D
], dtype=np.float64)

CORNER_NAMES = ["A", "B", "C", "D"]


# ============================================================
# RGB CHECKERBOARD DETECTION
# ============================================================

# def detect_checkerboard(rgb):

#     gray = cv2.cvtColor(
#         rgb,
#         cv2.COLOR_BGR2GRAY
#     )

#     flags = (
#         cv2.CALIB_CB_ADAPTIVE_THRESH |
#         cv2.CALIB_CB_NORMALIZE_IMAGE
#     )

#     found, corners = cv2.findChessboardCorners(
#         gray,
#         (BOARD_WIDTH, BOARD_HEIGHT),
#         flags
#     )

#     if not found:
#         return None

#     corners = cv2.cornerSubPix(
#         gray,
#         corners,
#         (11, 11),
#         (-1, -1),
#         (
#             cv2.TERM_CRITERIA_EPS |
#             cv2.TERM_CRITERIA_MAX_ITER,
#             40,
#             0.001
#         )
#     )

#     success, rvec, tvec = cv2.solvePnP(
#         OBJECT_POINTS,
#         corners,
#         K_RGB,
#         D_RGB
#     )

#     if not success:
#         return None

#     projected, _ = cv2.projectPoints(
#         OUTER_CORNERS_OBJECT,
#         rvec,
#         tvec,
#         K_RGB,
#         D_RGB
#     )

#     return {
#         "corners": corners,
#         "rvec": rvec,
#         "tvec": tvec,
#         "outer_rgb": projected.reshape(4, 2)
#     }

def detect_checkerboard(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ) if image.ndim == 3 else image.copy()

    pattern_size = (BOARD_WIDTH, BOARD_HEIGHT)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags
    )

    if not found:

        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size
        )

    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001
    )

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
    )

    calibration = np.load(
        "calibration_data/rgb_intrinsics.npz"
    )

    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["distortion"]

    success, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    if tvec[2, 0] <= 0:
        return None

    R, _ = cv2.Rodrigues(rvec)

    # Check that the checkerboard is in front of
    # the camera.
    camera_points = (
        R @ OBJECT_POINTS.T + tvec
    )

    if np.any(camera_points[2, :] <= 0):
        return None

    outer_rgb, _ = cv2.projectPoints(
        OUTER_CORNERS_OBJECT,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    return {
        "corners": corners,
        "rvec": rvec,
        "tvec": tvec,
        "outer_rgb": outer_rgb.reshape(4, 2)
    }


# ============================================================
# RGB DISPLAY
# ============================================================

def display_rgb(rgb, pose, count):

    display = rgb.copy()

    cv2.drawChessboardCorners(
        display,
        (BOARD_WIDTH, BOARD_HEIGHT),
        pose["corners"],
        True
    )

    for i, point in enumerate(pose["outer_rgb"]):

        x, y = np.round(point).astype(int)

        cv2.circle(
            display,
            (x, y),
            10,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            display,
            CORNER_NAMES[i],
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )

    cv2.putText(
        display,
        f"Pose {count + 1}/{NUM_POSES_TARGET}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "RGB - Checkerboard",
        display
    )


# ============================================================
# DEPTH DISPLAY
# ============================================================

def depth_visualization(depth):

    valid = depth > 0

    output = np.zeros(
        depth.shape,
        dtype=np.uint8
    )

    if not np.any(valid):
        return cv2.cvtColor(
            output,
            cv2.COLOR_GRAY2BGR
        )

    near = np.percentile(depth[valid], 2)
    far = np.percentile(depth[valid], 98)

    output[valid] = np.clip(
        (depth[valid] - near) /
        max(far - near, 1) * 255,
        0,
        255
    ).astype(np.uint8)

    return cv2.applyColorMap(
        255 - output,
        cv2.COLORMAP_JET
    )


def project_checkerboard_to_depth(pose):
    """
    Project the RGB-detected checkerboard onto the depth image.

    Uses the existing RGB-D extrinsic calibration.
    """

    calibration = np.load(
        "calibration_data/rgbd_extrinsics/rgbd_corner_calibration.npz"
    )

    R = calibration["rotation_depth_to_rgb"]

    T = calibration[
        "translation_depth_to_rgb_mm"
    ].reshape(3, 1)

    # Checkerboard pose in RGB camera coordinates.
    R_board, _ = cv2.Rodrigues(pose["rvec"])

    board_rgb = (
        R_board @ OUTER_CORNERS_OBJECT.T
        + pose["tvec"]
    )

    # Transform RGB coordinates into depth coordinates.
    board_depth = R.T @ (board_rgb - T)

    # Project into depth image.
    projected = []

    for point in board_depth.T:

        X, Y, Z = point

        if Z <= 0:
            return None

        u = K_DEPTH[0, 0] * X / Z + K_DEPTH[0, 2]
        v = K_DEPTH[1, 1] * Y / Z + K_DEPTH[1, 2]

        projected.append([u, v])

    return np.array(projected, dtype=np.float32)


def display_depth_with_checkerboard(depth, pose):
    display = depth_visualization(depth)

    if pose is not None:

        corners = project_checkerboard_to_depth(pose)

        if corners is not None:

            corners = np.round(corners).astype(int)

            # Draw checkerboard outline.
            cv2.polylines(
                display,
                [corners],
                True,
                (0, 255, 0),
                3
            )

            # Label the four corners.
            for i, point in enumerate(corners):

                x, y = point

                cv2.circle(
                    display,
                    (x, y),
                    8,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    display,
                    CORNER_NAMES[i],
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

    # cv2.imshow(
    #     "Depth Preview",
    #     display
    # )
    # display_depth_with_checkerboard(
    #     depth,
    #     pose
    # )
    cv2.imshow(
        "Depth Preview",
        display
    )


# ============================================================
# DEPTH CORNER SELECTOR
# ============================================================

class DepthCornerSelector:

    def __init__(self, depth):

        self.depth = depth.copy()

        self.image = depth_visualization(depth)

        self.display = self.image.copy()

        self.points = []

        self.window = "Depth - Select corners A B C D"

        cv2.namedWindow(
            self.window,
            cv2.WINDOW_NORMAL
        )

        cv2.setMouseCallback(
            self.window,
            self.mouse_callback
        )

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(self.points) < 4:

                self.points.append((x, y))

                self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:

            if self.points:

                self.points.pop()

                self.update_display()

    def update_display(self):

        self.display = self.image.copy()

        for i, point in enumerate(self.points):

            cv2.circle(
                self.display,
                point,
                8,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                self.display,
                CORNER_NAMES[i],
                (point[0] + 12, point[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        instructions = [
            "Click corners A, B, C, D in RGB label order",
            "Right click: undo",
            "ENTER: accept",
            "R: reset",
            "ESC: cancel"
        ]

        for i, line in enumerate(instructions):

            cv2.putText(
                self.display,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    def select(self):

        self.update_display()

        while True:

            cv2.imshow(
                self.window,
                self.display
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (10, 13):

                if len(self.points) == 4:
                    break

            elif key == ord("r"):

                self.points = []

                self.update_display()

            elif key == 27:

                cv2.destroyWindow(self.window)

                return None

        result = np.array(
            self.points,
            dtype=np.int32
        )

        cv2.destroyWindow(self.window)

        return result


# ============================================================
# DEPTH PIXEL -> 3D POINT
# ============================================================

def depth_pixel_to_world(camera, x, y, depth):

    # Use OpenNI's own coordinate conversion.
    #
    # Returns:
    # X: right
    # Y: up
    # Z: forward

    world = openni2.convert_depth_to_world(
        camera.depth_stream,
        float(x),
        float(y),
        float(depth)
    )

    # Convert OpenNI coordinates to OpenCV camera
    # coordinates: X right, Y down, Z forward.

    return np.array([
        world[0],
        -world[1],
        world[2]
    ], dtype=np.float64)


# def sample_depth_corner(camera, depth, point, radius=3):

def sample_depth_corner(camera, depth, point, radius=2):

    x, y = point

    h, w = depth.shape

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    patch = depth[y0:y1, x0:x1]

    valid = patch[
        (patch > 200) &
        (patch < 8000)
    ]

    if len(valid) < 5:

        raise RuntimeError(
            f"Insufficient depth around corner ({x},{y})"
        )

    median_depth = float(
        np.median(valid)
    )

    return depth_pixel_to_world(
        camera,
        x,
        y,
        median_depth
    )


# ============================================================
# KABSCH INITIALIZATION
# ============================================================

def estimate_rigid_transform(source, target):

    source_center = np.mean(
        source,
        axis=0
    )

    target_center = np.mean(
        target,
        axis=0
    )

    source_centered = source - source_center
    target_centered = target - target_center

    H = source_centered.T @ target_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:

        Vt[-1, :] *= -1

        R = Vt.T @ U.T

    T = target_center - R @ source_center

    return R, T


# ============================================================
# CAPTURE
# ============================================================

def capture_observations():

    observations = []

    camera = XtionCamera()

    try:

        camera.open()

        print("\nRGB-D CORNER CALIBRATION")
        print("========================")
        print("SPACE: Capture pose")
        print("Q: Finish")
        print()

        while len(observations) < NUM_POSES_TARGET:

            rgb, depth = camera.get_frames()

            pose = detect_checkerboard(rgb)

            if pose is not None:

                display_rgb(
                    rgb,
                    pose,
                    len(observations)
                )

            else:

                display = rgb.copy()

                cv2.putText(
                    display,
                    "CHECKERBOARD NOT DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "RGB - Checkerboard",
                    display
                )

            # cv2.imshow(
            #     "Depth Preview",
            #     depth_visualization(depth)
            # )

            display_depth_with_checkerboard(
                depth,
                pose
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key != 32:
                continue

            if pose is None:

                print("Checkerboard not detected.")

                continue

            print(
                f"\nCapturing pose "
                f"{len(observations) + 1}"
            )

            selector = DepthCornerSelector(
                depth
            )

            selected = selector.select()

            if selected is None:

                print("Cancelled.")

                continue

            try:

                depth_points = []

                for point in selected:

                    depth_points.append(
                        sample_depth_corner(
                            camera,
                            depth,
                            point
                        )
                    )

                depth_points = np.asarray(
                    depth_points,
                    dtype=np.float64
                )

                rgb_points = pose["outer_rgb"].astype(
                    np.float64
                )

                # # Reject points behind the RGB camera.
                # if np.any(pose["tvec"][:, 0] <= 0):
                #     print("Invalid RGB pose.")
                #     continue

                # Reject poses behind the RGB camera.
                if pose["tvec"][2, 0] <= 0:
                    print("Invalid RGB pose: checkerboard behind camera.")
                    print("Translation:", pose["tvec"].flatten())
                    continue

                observations.append({
                    "depth_points": depth_points,
                    "rgb_points": rgb_points,
                    "rgb_rvec": pose["rvec"].copy(),
                    "rgb_tvec": pose["tvec"].copy()
                })

                print(
                    f"Accepted pose "
                    f"{len(observations)}"
                )

                print("Depth corners (mm):")
                print(depth_points)

                print("RGB corners (px):")
                print(rgb_points)

            except Exception as error:

                print(
                    "Capture failed:",
                    error
                )

        cv2.destroyAllWindows()

    finally:

        camera.close()

    return observations


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_extrinsics(observations):

    if len(observations) < 6:

        raise RuntimeError(
            "At least 6 poses are required."
        )

    source = np.concatenate([
        obs["depth_points"]
        for obs in observations
    ])

    target = []

    for obs in observations:

        # Convert RGB image points to 3D points using
        # the known checkerboard geometry and solvePnP.
        #
        # The RGB-side points are the projected outer
        # checkerboard surface corners.

        rvec = obs["rgb_rvec"]
        tvec = obs["rgb_tvec"]

        points = cv2.Rodrigues(rvec)[0] @ (
            OUTER_CORNERS_OBJECT.T
        ) + tvec

        target.extend(points.T)

    target = np.asarray(
        target,
        dtype=np.float64
    )

    R_initial, T_initial = estimate_rigid_transform(
        source,
        target
    )

    initial_rotvec = Rotation.from_matrix(
        R_initial
    ).as_rotvec()

    initial = np.concatenate([
        initial_rotvec,
        T_initial
    ])

    def residual_function(parameters):

        R = Rotation.from_rotvec(
            parameters[:3]
        ).as_matrix()

        T = parameters[3:6]

        transformed = (
            source @ R.T + T
        )

        residuals = transformed - target

        return residuals.ravel()

    result = least_squares(
        residual_function,
        initial,
        method="trf",
        loss="soft_l1",
        f_scale=10.0,
        max_nfev=5000,
        verbose=1
    )

    R = Rotation.from_rotvec(
        result.x[:3]
    ).as_matrix()

    T = result.x[3:6]

    residuals = residual_function(
        result.x
    ).reshape(-1, 3)

    distances = np.linalg.norm(
        residuals,
        axis=1
    )
    print("\n========== PER-POSE ERRORS ==========")

    for i, obs in enumerate(observations):

        start = i * 4
        end = start + 4

        pose_errors = distances[start:end]

        print(f"\nPose {i + 1}:")

        for j, error in enumerate(pose_errors):

            print(
                f"  Corner {CORNER_NAMES[j]}: "
                f"{error:.2f} mm"
            )

        print(
            f"  Mean: {np.mean(pose_errors):.2f} mm"
        )

        print(
            f"  Maximum: {np.max(pose_errors):.2f} mm"
        )

    print("\n====================================")

    rms = np.sqrt(
        np.mean(distances ** 2)
    )

    print("\n========== RGB-D CORNER RESULTS ==========")

    print("\nRotation:")
    print(R)

    print("\nTranslation (mm):")
    print(T)

    print("\nOptimization successful:")
    print(result.success)

    print("\nCost:")
    print(result.cost)

    print("\n3D point residual RMS:")
    print(rms)

    print("\nMean point error:")
    print(np.mean(distances))

    print("\nMaximum point error:")
    print(np.max(distances))

    print("==========================================")

    return R, T, result, rms, distances


# ============================================================
# SAVE
# ============================================================

def save_results(
    R,
    T,
    result,
    rms,
    distances,
    observations
):

    np.savez(
        OUTPUT_DIR / "rgbd_corner_calibration.npz",

        rotation_depth_to_rgb=R,

        translation_depth_to_rgb_mm=T.reshape(3, 1),

        optimization_cost=result.cost,

        residual_rms_mm=rms,

        num_poses=len(observations),

        point_errors_mm=distances
    )

    fs = cv2.FileStorage(
        str(
            OUTPUT_DIR /
            "rgbd_corner_calibration.xml"
        ),
        cv2.FILE_STORAGE_WRITE
    )

    fs.write(
        "rotation_depth_to_rgb",
        R
    )

    fs.write(
        "translation_depth_to_rgb_mm",
        T.reshape(3, 1)
    )

    fs.write(
        "num_poses",
        len(observations)
    )

    fs.write(
        "residual_rms_mm",
        float(rms)
    )

    fs.release()

    with open(
        OUTPUT_DIR / "calibration_report.txt",
        "w"
    ) as f:

        f.write("RGB-D CORNER CALIBRATION\n")
        f.write("========================\n\n")

        f.write(f"Poses: {len(observations)}\n")

        f.write(f"RMS: {rms:.4f} mm\n")

        f.write(
            f"Mean error: {np.mean(distances):.4f} mm\n"
        )

        f.write(
            f"Maximum error: {np.max(distances):.4f} mm\n\n"
        )

        f.write("Rotation:\n")
        f.write(np.array2string(R))

        f.write("\n\nTranslation:\n")
        f.write(np.array2string(T))

    print("\nResults saved to:")
    print(OUTPUT_DIR.resolve())


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print(" RGB-D CORNER CALIBRATION")
    print("========================================")

    print("RGB: 1280 x 1024")
    print("Depth: 640 x 480")
    print("Checkerboard: 8 x 5")
    print("Square size: 29 mm")
    print("Border: 20 mm")

    print("\nOuter surface dimensions:")
    print("Width: 301 mm")
    print("Height: 214 mm")

    observations = capture_observations()

    if len(observations) < 6:

        print(
            "\nNot enough observations. "
            "At least 6 are required."
        )

        return

    R, T, result, rms, distances = optimize_extrinsics(
        observations
    )

    save_results(
        R,
        T,
        result,
        rms,
        distances,
        observations
    )


if __name__ == "__main__":

    main()
