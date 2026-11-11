import sys
import json
import time

from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPixmap, QPainter
from PyQt6.QtCore import Qt, QTimer

import signal


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from xtion import XtionCamera

from calib_common import (
    BOARD_COLUMNS,
    BOARD_ROWS,
    detect_checkerboard,
    solve_board_pose,
    load_rgb_intrinsics,
)


PATTERN_DIR = ROOT / "patterns"

OUTPUT_DIR = (
    ROOT /
    "calibration_data" /
    "projector"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# PROJECTOR CONFIGURATION
# ------------------------------------------------------------

PROJECTOR_X = 3840
PROJECTOR_Y = 233

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

PATTERN_DELAY_MS = 300

MAX_BOARD_MOVEMENT_PX = 2.0

TARGET_POSES = 15


# ------------------------------------------------------------
# PROJECTOR DISPLAY
# ------------------------------------------------------------

class ProjectorWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.pixmap = QPixmap()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            |
            Qt.WindowType.WindowStaysOnTopHint
        )

        self.setGeometry(
            PROJECTOR_X,
            PROJECTOR_Y,
            PROJECTOR_WIDTH,
            PROJECTOR_HEIGHT
        )

        self.setCursor(
            Qt.CursorShape.BlankCursor
        )

        self.showFullScreen()

    def display_pattern(self, filename):

        self.pixmap = QPixmap(str(filename))

        if self.pixmap.isNull():

            raise RuntimeError(
                f"Cannot load pattern: {filename}"
            )

        self.update()

        QApplication.processEvents()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.fillRect(
            self.rect(),
            Qt.GlobalColor.black
        )

        if not self.pixmap.isNull():

            painter.drawPixmap(
                self.rect(),
                self.pixmap
            )

        painter.end()


# ------------------------------------------------------------
# CAPTURE APPLICATION
# ------------------------------------------------------------

class ProjectorCapture:

    def __init__(self):

        self.closed = False

        self.camera = XtionCamera()

        self.camera.open()

        self.rgb_intrinsics = load_rgb_intrinsics()

        self.patterns = sorted(
            PATTERN_DIR.glob("*.png")
        )

        if len(self.patterns) != 46:

            raise RuntimeError(
                f"Expected 46 patterns, found {len(self.patterns)}"
            )

        self.display = ProjectorWindow()

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.preview
        )

        self.timer.start(30)

        self.state = "preview"

        self.latest_rgb = None

        self.latest_corners = None

        self.latest_pose = None

        self.pattern_index = 0

        self.capture_dir = None

        self.capture_metadata = None

        self.accepted_poses = 0

        self.pose_number = self.next_pose_number()

        # Start with black projection.

        self.display.display_pattern(
            self.patterns[1]
        )

        cv2.namedWindow(
            "Xtion Preview",
            cv2.WINDOW_NORMAL
        )

        cv2.moveWindow(
            "Xtion Preview",
            100,
            100
        )

        print("\nPROJECTOR CALIBRATION CAPTURE")
        print("==============================")

        print("Projector: DP-1")
        print("Resolution: 1920 x 1080")

        print("\nControls:")
        print("C: Capture current checkerboard pose")
        print("Q: Quit")
        print()

    # --------------------------------------------------------
    # NEXT POSE NUMBER
    # --------------------------------------------------------

    def next_pose_number(self):

        existing = list(
            OUTPUT_DIR.glob("pose_*")
        )

        numbers = []

        for folder in existing:

            try:

                numbers.append(
                    int(folder.name.split("_")[1])
                )

            except Exception:

                pass

        return max(numbers, default=0) + 1

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    def preview(self):

        if self.state != "preview":

            return

        rgb, depth = self.camera.get_frames()

        found, corners, gray = detect_checkerboard(
            rgb,
            columns=BOARD_COLUMNS,
            rows=BOARD_ROWS,
            refine=True,
            fast=True
        )

        preview = rgb.copy()

        self.latest_rgb = rgb.copy()

        self.latest_corners = None

        self.latest_pose = None

        if found:

            pose = solve_board_pose(
                corners,
                self.rgb_intrinsics.camera_matrix,
                self.rgb_intrinsics.distortion
            )

            if pose is not None:

                self.latest_corners = corners.copy()

                self.latest_pose = pose

                cv2.drawChessboardCorners(
                    preview,
                    (BOARD_COLUMNS, BOARD_ROWS),
                    corners,
                    True
                )

                status = "READY - PRESS C"

                color = (0, 255, 0)

                cv2.putText(
                    preview,
                    f"Distance: {pose.distance_mm:.0f} mm",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    preview,
                    f"Reprojection: {pose.reprojection_rms_px:.2f} px",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            else:

                status = "PNP FAILED"

                color = (0, 0, 255)

        else:

            status = "CHECKERBOARD NOT FOUND"

            color = (0, 0, 255)

        cv2.putText(
            preview,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            preview,
            f"Accepted poses: {self.accepted_poses}/{TARGET_POSES}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Xtion Preview",
            preview
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            self.start_capture()

        elif key == ord("q"):

            self.close()

    # --------------------------------------------------------
    # START CAPTURE
    # --------------------------------------------------------

    def start_capture(self):

        if self.latest_pose is None:

            print("No valid checkerboard detected.")

            return

        self.timer.stop()

        self.state = "capturing"

        self.capture_dir = (
            OUTPUT_DIR /
            f"pose_{self.pose_number:03d}"
        )

        self.capture_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        self.start_corners = (
            self.latest_corners.copy()
        )

        self.start_pose = self.latest_pose

        cv2.imwrite(
            str(self.capture_dir / "rgb_start.png"),
            self.latest_rgb
        )

        self.capture_metadata = {

            "pose_number": self.pose_number,

            "resolution": [
                PROJECTOR_WIDTH,
                PROJECTOR_HEIGHT
            ],

            "rvec": self.start_pose.rvec.tolist(),

            "tvec_mm": self.start_pose.tvec.tolist(),

            "reprojection_rms_px":
                self.start_pose.reprojection_rms_px,

            "corners_start":
                self.start_corners.reshape(-1, 2).tolist(),

            "pattern_files": [
                p.name for p in self.patterns
            ]

        }

        self.pattern_index = 0

        print(
            f"\nStarting pose {self.pose_number}"
        )

        self.show_next_pattern()

    # --------------------------------------------------------
    # SHOW AND CAPTURE PATTERNS
    # --------------------------------------------------------

    def show_next_pattern(self):

        if self.pattern_index >= len(self.patterns):

            self.display.display_pattern(
                self.patterns[0]
            )

            QTimer.singleShot(
                PATTERN_DELAY_MS,
                self.finish_capture
            )

            return

        pattern = self.patterns[
            self.pattern_index
        ]

        self.display.display_pattern(
            pattern
        )

        print(
            f"Pattern {self.pattern_index + 1}/46: "
            f"{pattern.name}"
        )

        QTimer.singleShot(
            PATTERN_DELAY_MS,
            self.capture_pattern
        )

    def capture_pattern(self):

        pattern = self.patterns[
            self.pattern_index
        ]

        # Discard two frames to allow the camera
        # to receive the new projected pattern.

        for _ in range(2):

            rgb, depth = self.camera.get_frames()

        output_path = (
            self.capture_dir /
            pattern.name
        )

        if not cv2.imwrite(
            str(output_path),
            rgb
        ):

            raise RuntimeError(
                f"Could not save {output_path}"
            )

        self.pattern_index += 1

        self.show_next_pattern()

    # --------------------------------------------------------
    # VERIFY BOARD MOVEMENT
    # --------------------------------------------------------

    def finish_capture(self):

        # Capture a fresh image under black projection.

        for _ in range(3):

            rgb, depth = self.camera.get_frames()

        cv2.imwrite(
            str(self.capture_dir / "rgb_end.png"),
            rgb
        )

        found, corners, gray = detect_checkerboard(
            rgb,
            columns=BOARD_COLUMNS,
            rows=BOARD_ROWS,
            refine=True,
            fast=False
        )

        movement = None

        accepted = False

        if found:

            displacement = (
                corners.reshape(-1, 2)
                -
                self.start_corners.reshape(-1, 2)
            )

            movement = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            displacement ** 2,
                            axis=1
                        )
                    )
                )
            )

            accepted = (
                movement <= MAX_BOARD_MOVEMENT_PX
            )

        self.capture_metadata.update({

            "accepted": accepted,

            "board_movement_rms_px": movement,

            "movement_threshold_px":
                MAX_BOARD_MOVEMENT_PX,

            "corners_end":
                corners.reshape(-1, 2).tolist()
                if found else None

        })

        with open(
            self.capture_dir / "metadata.json",
            "w"
        ) as f:

            json.dump(
                self.capture_metadata,
                f,
                indent=4
            )

        if accepted:

            self.accepted_poses += 1

            print(
                f"\nPOSE ACCEPTED: {self.pose_number}"
            )

            print(
                f"Board movement: {movement:.3f} px"
            )

        else:

            print(
                f"\nPOSE REJECTED: {self.pose_number}"
            )

            if movement is None:

                print(
                    "Checkerboard could not be detected after capture."
                )

            else:

                print(
                    f"Board movement: {movement:.3f} px"
                )

        self.pose_number += 1

        print(
            f"Accepted poses: "
            f"{self.accepted_poses}/{TARGET_POSES}"
        )

        print(
            "\nMove the checkerboard to another position."
        )

        # Display white while positioning the next checkerboard pose.

        self.display.display_pattern(
            self.patterns[1]
        )

        self.state = "preview"

        self.timer.start(30)

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    # def close(self):

    #     self.timer.stop()

    #     self.camera.close()

    #     cv2.destroyAllWindows()

    #     QApplication.quit()

    def close(self):

        if self.closed:
            return

        self.closed = True

        self.timer.stop()

        try:
            self.camera.close()

        except Exception as error:

            print(
                "Camera shutdown warning:",
                error
            )

        self.display.close()

        cv2.destroyAllWindows()

        QApplication.quit()




# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

# def main():

#     app = QApplication(sys.argv)

#     capture = ProjectorCapture()

#     app.aboutToQuit.connect(
#         capture.close
#     )

#     sys.exit(
#         app.exec()
#     )

def main():

    app = QApplication(sys.argv)

    capture = ProjectorCapture()

    app.aboutToQuit.connect(
        capture.close
    )

    # Handle Ctrl+C from terminal.
    signal.signal(
        signal.SIGINT,
        lambda signum, frame: app.quit()
    )

    # Keep Python's signal handler responsive
    # while the Qt event loop is running.
    signal_timer = QTimer()

    signal_timer.timeout.connect(
        lambda: None
    )

    signal_timer.start(200)

    sys.exit(
        app.exec()
    )



if __name__ == "__main__":

    main()
