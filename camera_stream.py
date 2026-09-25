#!/usr/bin/env python3
"""Serve a USB UVC camera as a simple browser-viewable MJPEG stream."""

import argparse
import logging
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class Camera:
    def __init__(self, device, width, height, fps, quality, fourcc):
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open {device}. Check the device path and camera permissions.")

        if fourcc != "auto":
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        logging.info(
            "Camera mode: %dx%d, %.1f fps requested, format %s",
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.capture.get(cv2.CAP_PROP_FPS),
            actual_fourcc,
        )

        self.quality = quality
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        failures = 0
        while self.running:
            ok, image = self.capture.read()
            if not ok:
                failures += 1
                if failures >= 10:
                    with self.condition:
                        self.error = "Camera stopped returning frames"
                        self.condition.notify_all()
                    logging.error(self.error)
                    break
                time.sleep(0.1)
                continue
            failures = 0
            ok, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            if not ok:
                continue
            with self.condition:
                self.frame = encoded.tobytes()
                self.sequence += 1
                self.condition.notify_all()

    def next_frame(self, previous, timeout=5):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous or self.error or not self.running,
                timeout=timeout,
            )
            return self.sequence, self.frame, self.error

    def close(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)
        self.capture.release()


class Handler(BaseHTTPRequestHandler):
    camera = None

    def do_GET(self):
        if self.path == "/":
            body = (
                b"<!doctype html><html><head><meta name='viewport' "
                b"content='width=device-width, initial-scale=1'><title>Pi camera</title>"
                b"<style>body{background:#111;color:#eee;font:16px sans-serif;"
                b"text-align:center;margin:2rem}img{max-width:100%;height:auto}</style>"
                b"</head><body><h1>Pi camera</h1><img src='/stream.mjpg' "
                b"alt='Live camera stream'></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/snapshot.jpg":
            _, frame, error = self.camera.next_frame(0)
            if frame is None or error:
                self.send_error(503, error or "No frame available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
        elif self.path == "/healthz":
            _, frame, error = self.camera.next_frame(0, timeout=0)
            healthy = frame is not None and error is None
            body = b"ok\n" if healthy else b"camera unavailable\n"
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = 0
            try:
                while self.camera.running:
                    sequence, frame, error = self.camera.next_frame(sequence)
                    if error:
                        break
                    if frame is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode("ascii")
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0", help="V4L2 video device")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", choices=("MJPG", "YUYV", "auto"), default="MJPG")
    parser.add_argument("--quality", type=int, choices=range(1, 101), metavar="1-100", default=80)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    camera = Camera(args.device, args.width, args.height, args.fps, args.quality, args.fourcc)
    Handler.camera = camera
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logging.info("Open http://<pi-ip>:%d/ in a browser (Ctrl+C to stop)", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        camera.close()


if __name__ == "__main__":
    main()
