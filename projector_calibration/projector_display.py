import sys

from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QWidget
)

from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QColor
)

from PyQt6.QtCore import (
    Qt,
    QRect
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECTOR_X = 3840
PROJECTOR_Y = 233

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080

PATTERN_DIR = Path("patterns")


# ============================================================
# DISPLAY
# ============================================================

class ProjectorDisplay(QWidget):

    def __init__(self):

        super().__init__()

        self.files = sorted(
            PATTERN_DIR.glob("*.png")
        )

        self.index = 0

        if not self.files:

            raise RuntimeError(
                "No patterns found."
            )

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
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

        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.fillRect(
            self.rect(),
            QColor(0, 0, 0)
        )

        if self.index < len(self.files):

            pixmap = QPixmap(
                str(self.files[self.index])
            )

            painter.drawPixmap(
                QRect(
                    0,
                    0,
                    PROJECTOR_WIDTH,
                    PROJECTOR_HEIGHT
                ),
                pixmap
            )

        painter.end()

    def keyPressEvent(self, event):

        key = event.key()

        if key == Qt.Key.Key_Escape:

            self.close()

        elif key == Qt.Key.Key_Right:

            self.index = min(
                self.index + 1,
                len(self.files) - 1
            )

            self.update()

        elif key == Qt.Key.Key_Left:

            self.index = max(
                self.index - 1,
                0
            )

            self.update()

        elif key == Qt.Key.Key_Space:

            self.index = min(
                self.index + 1,
                len(self.files) - 1
            )

            self.update()

        elif key == Qt.Key.Key_Home:

            self.index = 0

            self.update()

        elif key == Qt.Key.Key_End:

            self.index = len(self.files) - 1

            self.update()


# ============================================================
# MAIN
# ============================================================

def main():

    app = QApplication(sys.argv)

    print("\nAVAILABLE DISPLAYS")
    print("==================")

    for i, screen in enumerate(
        app.screens()
    ):

        print(
            i,
            screen.name(),
            screen.geometry()
        )

    print("\nPROJECTOR CONTROLS")
    print("==================")

    print("RIGHT / SPACE: Next pattern")
    print("LEFT: Previous pattern")
    print("HOME: First pattern")
    print("END: Last pattern")
    print("ESC: Exit")

    window = ProjectorDisplay()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":

    main()
