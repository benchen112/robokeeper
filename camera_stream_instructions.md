# Raspberry Pi USB camera live view

This starter streams an Arducam OV9281 **USB UVC** camera to a browser as MJPEG. It uses V4L2 through OpenCV; `rpicam-vid` and CSI camera setup are not needed for this USB model.

## On the Raspberry Pi

Install the packages:

```bash
sudo apt update
sudo apt install -y python3-opencv v4l-utils
```

Copy `camera_stream.py` from this project to the Pi, then find the video device and supported modes:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

The camera may expose multiple `/dev/video*` entries. Pick the entry that reports **Video Capture** and has the image formats in `--list-formats-ext`. A stable path under `/dev/v4l/by-id/` is useful when other cameras are connected.

Start with a modest mode:

```bash
python3 camera_stream.py --device /dev/video0 --width 640 --height 480 --fps 30
```

Open `http://<pi-ip>:8000/` on a computer or phone on the same network. Find the Pi's IP with `hostname -I`. Press Ctrl+C in the Pi terminal to stop. `/snapshot.jpg` returns one JPEG and `/healthz` reports whether frames are arriving.

The camera's advertised 100 fps is a capture capability, not a guarantee for browser MJPEG streaming. To try a faster mode, choose an exact resolution, pixel format, and frame rate shown by `v4l2-ctl --list-formats-ext`, then pass them through `--width`, `--height`, `--fps`, and `--fourcc` (`MJPG`, `YUYV`, or `auto`). The startup log shows the mode OpenCV actually selected. MJPG usually uses less USB bandwidth; YUYV can be useful if MJPG is absent. Browser frame rate may be lower because the Pi encodes JPEGs and sends them over the network.

If opening the camera fails, confirm that no other program has it open and that your user can access `/dev/video*` (`groups` should include `video`; logging out and back in applies a new group membership). If the page loads but stays blank, check the Pi terminal for capture errors and try a mode listed by `v4l2-ctl`.

The server listens on all network interfaces by default and has no authentication. Use `--host 127.0.0.1` if you only want local access or plan to forward the port over SSH.
