import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# --------------------------------------------------
# XTION CONFIGURATION
# --------------------------------------------------

XTION = {

    "rgb_width": 1280,
    "rgb_height": 1024,
    "rgb_fps": 30,

    "depth_width": 640,
    "depth_height": 480,
    "depth_fps": 30,

    "depth_scale": 0.001,

    "rotate_preview": True,

    "openni_library": (
        "/usr/lib/x86_64-linux-gnu"
    )
}


# --------------------------------------------------
# PROJECTOR CONFIGURATION
# --------------------------------------------------

PROJECTOR = {

    "width": 1920,
    "height": 1080,

    # Set these to the projector's desktop position.
    "screen_x": 1920,
    "screen_y": 0,

    "fullscreen": True
}


# --------------------------------------------------
# CHESSBOARD CONFIGURATION
# --------------------------------------------------

CHESSBOARD = {

    "columns": 8,
    "rows": 5,

    "square_size_mm": 29.0,

    "minimum_images": 15
}


# --------------------------------------------------
# CALIBRATION CONFIGURATION
# --------------------------------------------------

CALIBRATION = {

    "minimum_views": 8,

    "graycode_threshold": 20,

    "black_threshold": 25,

    "pattern_delay": 0.5,

    "depth_min_mm": 300,

    "depth_max_mm": 5000
}


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

DATA_DIR = os.path.join(BASE_DIR, "data")

CAMERA_DIR = os.path.join(
    DATA_DIR, "camera"
)

CAMERA_IMAGES_DIR = os.path.join(
    CAMERA_DIR, "images"
)

PROCAM_DIR = os.path.join(
    DATA_DIR, "procam"
)

PATTERN_DIR = os.path.join(
    BASE_DIR, "patterns"
)

EXPORT_DIR = os.path.join(
    BASE_DIR, "exports"
)


for directory in [
    CAMERA_IMAGES_DIR,
    PROCAM_DIR,
    PATTERN_DIR,
    EXPORT_DIR
]:

    os.makedirs(directory, exist_ok=True)
