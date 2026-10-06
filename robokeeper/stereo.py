"""Packed stereo frames and latest-only capture for the Camarray CSI HAT."""

import threading
import time

import cv2
import numpy as np


def split_stereo(frame, eye_size=(1280, 800), swap=False):
    """Split a side-by-side frame without resizing either sensor's image."""
    width, height = eye_size
    if frame.dtype != np.uint8 or frame.ndim not in (2, 3):
        raise ValueError("Expected an 8-bit grayscale or BGR stereo frame")
    if frame.shape[:2] != (height, width * 2):
        raise ValueError(
            f"Expected packed stereo {width * 2}x{height}, got "
            f"{frame.shape[1]}x{frame.shape[0]}. Check the camera mode; "
            "do not split a single-camera image or a scaled stereo stream."
        )
    views = (frame[:, :width], frame[:, width:])
    return views[::-1] if swap else views


def pair_diagnostics(left, right):
    """Raw correspondences for inspection, without asserting stereo identity."""
    measured = all(r.state == "confirmed" and r.observed and r.center is not None
                   for r in (left, right))
    return {
        "both_confirmed_observed": measured,
        "raw_disparity_px": float(left.center[0] - right.center[0]) if measured else None,
        "raw_vertical_offset_px": float(left.center[1] - right.center[1]) if measured else None,
    }


class LatestPicameraStereo:
    """One libcamera request contains both views; no software pairing queues.

    Track timestamps use SensorTimestamp when supplied, otherwise monotonic
    receive time. Receive time is separate for latency diagnostics, so clocks
    are never subtracted from each other. Requests are released after copying.
    """

    def __init__(self, width=2560, height=800, fps=60, camera_num=0,
                 tuning_file=None, exposure_us=None, gain=1.0):
        from picamera2 import Picamera2

        # The constructor must discover cameras after installing the tuning
        # override. Calling global_camera_info first initializes the IPA with
        # default tuning and prevents the requested file taking effect.
        tuning_args = ({"tuning": Picamera2.load_tuning_file(tuning_file)}
                       if tuning_file else {})
        try:
            self.camera = Picamera2(camera_num, **tuning_args)
        except RuntimeError as exc:
            raise RuntimeError("Cannot initialize Camarray camera. Check "
                               "rpicam-hello --list-cameras and the Arducam driver/overlay. "
                               f"{exc}") from exc
        self.condition = threading.Condition()
        self.sequence = 0
        self.latest = None
        self.error = None
        self.running = False
        self.timestamp_source = None
        try:
            # Request the OV9281 native 8-bit mode directly. Some Camarray drivers
            # advertise a bogus 640x200 mode with zero frame duration, which makes
            # Picamera2.sensor_modes fail while probing every unrelated mode.
            controls = {"FrameRate": fps}
            if exposure_us is not None:
                controls.update(AeEnable=False, ExposureTime=exposure_us, AnalogueGain=gain)
            config = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "YUV420"},
                raw={"size": (width, height), "format": "R8"},
                sensor={"output_size": (width, height), "bit_depth": 8},
                controls=controls, buffer_count=4, queue=False,
            )
            self.camera.configure(config)
            configured = self.camera.camera_configuration()
            actual = configured["main"]
            if (tuple(actual["size"]) != (width, height)
                    or tuple(configured["raw"]["size"]) != (width, height)):
                raise RuntimeError(f"Unexpected configured stream: {actual}")
            duration_min = self.camera.camera_controls.get("FrameDurationLimits", (0,))[0]
            if duration_min > 0 and fps > 1e6 / duration_min + .1:
                raise RuntimeError(f"Requested {fps} fps exceeds this mode's limit "
                                   f"{1e6 / duration_min:.1f}")
            self.height = height
            self.width = width
            self.camera.start()
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
        except BaseException:
            self.camera.close()
            raise

    def _read_loop(self):
        try:
            while self.running:
                request = self.camera.capture_request()
                try:
                    metadata = request.get_metadata()
                    # YUV420's first height rows are full-resolution monochrome luma.
                    frame = request.make_array("main")[:self.height, :self.width].copy()
                    received = time.monotonic()
                finally:
                    request.release()
                sensor_ns = metadata.get("SensorTimestamp")
                source = "sensor" if sensor_ns is not None else "receive_monotonic"
                if self.timestamp_source is not None and source != self.timestamp_source:
                    raise RuntimeError("Camera timestamp source changed during capture")
                self.timestamp_source = source
                timestamp = sensor_ns / 1e9 if sensor_ns is not None else received
                with self.condition:
                    self.sequence += 1
                    self.latest = (frame, timestamp, received, {
                        "timestamp_source": source,
                        "sensor_timestamp_ns": sensor_ns,
                        "exposure_us": metadata.get("ExposureTime"),
                        "analogue_gain": metadata.get("AnalogueGain"),
                        "frame_duration_us": metadata.get("FrameDuration"),
                    })
                    self.condition.notify_all()
        except Exception as exc:
            self.error = str(exc)
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()

    def read(self, previous_sequence):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous_sequence or not self.running, timeout=2)
            if self.sequence == previous_sequence:
                return None
            return self.sequence, *self.latest

    def close(self):
        self.running = False
        try:
            self.camera.stop()
            self.thread.join(timeout=2)
        finally:
            self.camera.close()


class PreviewBuffer:
    """Compatible with the existing browser MJPEG handler; no camera ownership."""

    def __init__(self):
        self.condition = threading.Condition()
        self.sequence = 0
        self.frame = None
        self.error = None
        self.running = True

    def publish(self, image):
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.condition:
                self.sequence += 1
                self.frame = jpeg.tobytes()
                self.condition.notify_all()

    def next_frame(self, previous, timeout=5):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence != previous or not self.running,
                                    timeout=timeout)
            return self.sequence, self.frame, self.error

    def close(self):
        with self.condition:
            self.running = False
            self.error = "Stereo test stopped"
            self.condition.notify_all()
