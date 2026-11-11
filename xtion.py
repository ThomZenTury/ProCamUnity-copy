import os
import platform
import cv2
import numpy as np

from openni import openni2
from openni import _openni2 as openni_api

from config import XTION


class XtionCamera:

    def __init__(self):

        self.device = None
        self.color_stream = None
        self.depth_stream = None

        self.initialized = False

    def open(self):

        lib_dir = XTION["openni_library"]

        driver_dir = os.path.join(
            lib_dir,
            "OpenNI2",
            "Drivers"
        )

        os.environ["OPENNI2_REDIST"] = lib_dir
        os.environ["OPENNI2_DRIVERS_PATH"] = driver_dir

        openni2.initialize(lib_dir)

        self.device = openni2.Device.open_any()

        self.color_stream = (
            self.device.create_color_stream()
        )

        self.depth_stream = (
            self.device.create_depth_stream()
        )

        self.configure_streams()

        self.initialized = True

        print("Xtion initialized successfully.")

    # def configure_streams(self):

    #     rgb_modes = (
    #         self.color_stream
    #         .get_sensor_info()
    #         .videoModes
    #     )

    #     depth_modes = (
    #         self.depth_stream
    #         .get_sensor_info()
    #         .videoModes
    #     )

    #     rgb_mode = None

    #     for mode in rgb_modes:

    #         if (
    #             mode.resolutionX == XTION["rgb_width"]
    #             and mode.resolutionY == XTION["rgb_height"]
    #             and mode.fps == XTION["rgb_fps"]
    #             and mode.pixelFormat ==
    #                 openni_api.ONI_PIXEL_FORMAT_RGB888
    #         ):

    #             rgb_mode = mode
    #             break

    #     depth_mode = None

    #     for mode in depth_modes:

    #         if (
    #             mode.resolutionX == XTION["depth_width"]
    #             and mode.resolutionY == XTION["depth_height"]
    #             and mode.fps == XTION["depth_fps"]
    #             and mode.pixelFormat ==
    #                 openni_api.ONI_PIXEL_FORMAT_DEPTH_1_MM
    #         ):

    #             depth_mode = mode
    #             break

    #     if rgb_mode is None:
    #         raise RuntimeError(
    #             "Requested RGB mode is unavailable."
    #         )

    #     if depth_mode is None:
    #         raise RuntimeError(
    #             "Requested depth mode is unavailable."
    #         )

    #     self.color_stream.set_video_mode(rgb_mode)

    #     self.depth_stream.set_video_mode(depth_mode)

    #     self.color_stream.start()
    #     self.depth_stream.start()

    #     print(
    #         "RGB:",
    #         rgb_mode.resolutionX,
    #         rgb_mode.resolutionY,
    #         rgb_mode.fps
    #     )

    #     print(
    #         "Depth:",
    #         depth_mode.resolutionX,
    #         depth_mode.resolutionY,
    #         depth_mode.fps
    #     )

    def configure_streams(self):

        rgb_modes = (
            self.color_stream
            .get_sensor_info()
            .videoModes
        )

        depth_modes = (
            self.depth_stream
            .get_sensor_info()
            .videoModes
        )

        pixel_formats = openni_api.OniPixelFormat

        print("\nAvailable pixel formats:")

        for name in dir(pixel_formats):

            if not name.startswith("_"):

                print(name, getattr(pixel_formats, name))

        rgb_mode = None
        depth_mode = None

        for mode in rgb_modes:

            if (
                mode.resolutionX == 1280
                and mode.resolutionY == 1024
                and mode.fps == 30
                and "RGB888" in str(mode.pixelFormat)
            ):

                rgb_mode = mode
                break

        for mode in depth_modes:

            if (
                mode.resolutionX == 640
                and mode.resolutionY == 480
                and mode.fps == 30
                and "DEPTH_1_MM" in str(mode.pixelFormat)
            ):

                depth_mode = mode
                break

        if rgb_mode is None:
            raise RuntimeError("RGB mode not found")

        if depth_mode is None:
            raise RuntimeError("Depth mode not found")

        self.color_stream.set_video_mode(rgb_mode)

        self.depth_stream.set_video_mode(depth_mode)

        self.color_stream.start()
        self.depth_stream.start()

        print("\nXtion configured:")
        print("RGB: 1280x1024 @ 30 FPS")
        print("Depth: 640x480 @ 30 FPS")

    def get_frames(self):

        if not self.initialized:
            raise RuntimeError("Xtion is not initialized.")

        color_frame = self.color_stream.read_frame()

        rgb = np.frombuffer(
            color_frame.get_buffer_as_uint8(),
            dtype=np.uint8
        ).reshape(
            color_frame.height,
            color_frame.width,
            3
        )

        # OpenNI returns RGB888.
        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

        depth_frame = self.depth_stream.read_frame()

        depth = np.frombuffer(
            depth_frame.get_buffer_as_uint16(),
            dtype=np.uint16
        ).reshape(
            depth_frame.height,
            depth_frame.width
        )

        return bgr.copy(), depth.copy()

    def close(self):

        if self.color_stream:
            self.color_stream.stop()

        if self.depth_stream:
            self.depth_stream.stop()

        if self.initialized:
            openni2.unload()

        self.initialized = False

        print("Xtion disconnected.")

    def __enter__(self):

        self.open()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()
