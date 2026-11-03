
# Sunrise / Sunset Hand Control

Control a stylized city skyline's time of day with nothing but your hand. Pinch your thumb and index finger together to raise the sun; spread them apart to make it set. The scene shifts through midnight, dawn, sunrise, morning, and high noon, with twinkling stars, drifting clouds, lit windows, a landmark tower, and traffic light trails.

Hand tracking is done in real time with [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), and everything is rendered with OpenCV.

## How it works

The script tracks the distance between your thumb tip (landmark 4) and index fingertip (landmark 8) from your webcam feed. That distance is mapped to a `sun_level` between 0 and 1:

| Gesture | Result |
|---|---|
| Fingers touching | Sun fully up (high noon) |
| Fingers fully spread | Sun fully down (midnight) |

The value is smoothed so the transitions feel gradual rather than jumpy.

## Requirements

- Python 3.9 - 3.12 (MediaPipe wheels are not available for every Python version)
- A webcam
- A desktop environment (the app opens an OpenCV window)


## Usage

```bash
python3 sun_hands.py
```

The hand landmark model (`hand_landmarker.task`, ~8 MB) is included in the repo. If it's missing, the script downloads it automatically on first run and saves it next to the script.

### Controls

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `f` | Toggle fullscreen |
| `c` | Toggle horizontal camera flip (mirror mode) |

## Configuration

Tunable constants are near the top of `sun_hands.py`:

| Constant | Default | Description |
|---|---|---|
| `DIST_MIN` | `0.02` | Pinch distance that counts as "fingers touching" (sun fully up) |
| `DIST_MAX` | `0.28` | Pinch distance that counts as "fully spread" (sun fully down) |
| `SMOOTHING` | `0.15` | 0-1; lower values respond more slowly but more smoothly |

Distances are normalized to image size, so if the effect feels too sensitive or too sluggish with your camera or hand size, adjust `DIST_MIN` / `DIST_MAX`. The on-screen HUD shows your live pinch distance to help with tuning.

## Troubleshooting

- **"could not open webcam"**: Another app may be using the camera, or your camera isn't at index 0. Change the index in `cv2.VideoCapture(0)`. On macOS, make sure your terminal has camera permission under System Settings > Privacy & Security > Camera.
- **Model download fails**: Download `hand_landmarker.task` manually from the [MediaPipe model page](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models) and place it next to `sun_hands.py`.
- **`ImportError` on `HandLandmarksConnections`**: Your MediaPipe version is too old. Run `pip install --upgrade mediapipe`.
- **Choppy performance**: Close other camera/video apps, or lower the capture resolution in `main()`.

## Project structure

```
.
├── sun_hands.py            # main application
├── hand_landmarker.task    # MediaPipe hand tracking model
├── requirements.txt
└── README.md
```

## Credits

- Hand tracking: [MediaPipe](https://github.com/google-ai-edge/mediapipe) by Google
- Rendering: [OpenCV](https://opencv.org/)
