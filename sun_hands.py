
import os
import math
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
THUMB_TIP = 4
INDEX_TIP = 8


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


def create_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return HandLandmarker.create_from_options(options)



DIST_MIN = 0.02   # fingers touching -> sun fully up (noon)
DIST_MAX = 0.28   # fingers fully spread -> sun fully down (midnight)

SMOOTHING = 0.15  # 0-1, lower = smoother/slower response to hand movement


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


# Sky color model
# ----------------------------------------------------------------------------
#   0.00        -> deep night
#   0.00-0.25   -> pre-dawn / dawn glow rising
#   0.25-0.50   -> sunrise (orange/pink horizon, blue above)
#   0.50-0.75   -> daytime brightening
#   0.75-1.00   -> full daylight

# (sun_level, top_color_RGB, horizon_color_RGB) keyframes
KEYFRAMES = [
    (0.00, (25, 15, 55), (60, 25, 90)),       # midnight - deep violet (Tehran dusk)
    (0.15, (45, 25, 80), (120, 45, 110)),     # pre-dawn - magenta/violet haze
    (0.30, (60, 70, 130), (255, 130, 100)),   # sunrise - orange horizon burst
    (0.45, (90, 140, 210), (255, 180, 120)),  # early morning - warm gold fading up
    (0.65, (135, 190, 235), (255, 235, 200)), # late morning - soft blue/white
    (1.00, (135, 206, 250), (255, 255, 240)), # noon - bright sky blue
]


def sky_colors(sun_level):
    sun_level = clamp(sun_level)
    for i in range(len(KEYFRAMES) - 1):
        t0, top0, hor0 = KEYFRAMES[i]
        t1, top1, hor1 = KEYFRAMES[i + 1]
        if t0 <= sun_level <= t1:
            t = 0 if t1 == t0 else (sun_level - t0) / (t1 - t0)
            top = lerp_color(top0, top1, t)
            hor = lerp_color(hor0, hor1, t)
            return (top[2], top[1], top[0]), (hor[2], hor[1], hor[0])
    top, hor = KEYFRAMES[-1][1], KEYFRAMES[-1][2]
    return (top[2], top[1], top[0]), (hor[2], hor[1], hor[0])


def draw_sky_gradient(canvas, top_color, horizon_color, horizon_y):
    h, w = canvas.shape[:2]
    horizon_y = int(clamp(horizon_y, 0, h - 1))
    if horizon_y > 0:
        grad = np.linspace(0, 1, horizon_y).reshape(-1, 1)
        top = np.array(top_color, dtype=np.float32)
        hor = np.array(horizon_color, dtype=np.float32)
        sky_col = (top * (1 - grad) + hor * grad).astype(np.uint8)
        canvas[0:horizon_y, :] = sky_col.reshape(-1, 1, 3)
    if horizon_y < h:
        ground_top = tuple(int(c * 0.35) for c in horizon_color)
        ground_bottom = (15, 25, 20)
        grad2 = np.linspace(0, 1, h - horizon_y).reshape(-1, 1)
        g0 = np.array(ground_top, dtype=np.float32)
        g1 = np.array(ground_bottom, dtype=np.float32)
        ground_col = (g0 * (1 - grad2) + g1 * grad2).astype(np.uint8)
        canvas[horizon_y:h, :] = ground_col.reshape(-1, 1, 3)


def draw_sun(canvas, sun_level, t_anim):
    h, w = canvas.shape[:2]
    horizon_y = int(h * 0.62)

    cx = int(lerp(w * 0.12, w * 0.88, sun_level))
    peak_y = h * 0.12
    base_y = horizon_y + h * 0.18
    ease = sun_level ** 0.7
    cy = int(base_y - (base_y - peak_y) * ease)

    radius = int(lerp(38, 55, 1 - abs(sun_level - 0.5) * 0.6))

    if sun_level < 0.35:
        t = sun_level / 0.35
        sun_col = lerp_color((80, 30, 200), (60, 140, 255), t)   # BGR: red -> orange
    else:
        t = clamp((sun_level - 0.35) / 0.65)
        sun_col = lerp_color((60, 140, 255), (240, 255, 255), t)  # orange -> warm white

    glow_layer = canvas.copy()
    max_glow = int(radius * (2.6 if sun_level < 0.4 else 3.4))
    for r in range(max_glow, radius, -6):
        alpha = 0.05 * (1 - (r - radius) / (max_glow - radius))
        cv2.circle(glow_layer, (cx, cy), r, sun_col, -1)
        canvas[:] = cv2.addWeighted(glow_layer, alpha, canvas, 1 - alpha, 0)
        glow_layer = canvas.copy()

    cv2.circle(canvas, (cx, cy), radius, sun_col, -1)
    cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)

    return cx, cy, radius, horizon_y, sun_col


def draw_stars(canvas, sun_level, seed_points, t_anim):
    if sun_level > 0.35:
        return
    h, w = canvas.shape[:2]
    alpha_mult = clamp(1 - sun_level / 0.35)
    overlay = canvas.copy()
    for (sx, sy, brightness, twinkle_phase) in seed_points:
        x = int(sx * w)
        y = int(sy * h * 0.6)
        tw = 0.5 + 0.5 * math.sin(t_anim * 2 + twinkle_phase)
        b = int(brightness * tw)
        cv2.circle(overlay, (x, y), 1, (b, b, b), -1)
    canvas[:] = cv2.addWeighted(overlay, alpha_mult, canvas, 1 - alpha_mult, 0)


def draw_clouds(canvas, sun_level, horizon_y, t_anim):
    h, w = canvas.shape[:2]
    opacity = clamp(0.15 + 0.35 * sun_level)
    tint_day = np.array([245, 245, 250], dtype=np.float32)
    tint_dusk = np.array([160, 150, 200], dtype=np.float32)
    tint = tint_day * sun_level + tint_dusk * (1 - sun_level)

    overlay = canvas.copy()
    cloud_rows = [0.20, 0.30, 0.40]
    for i, row in enumerate(cloud_rows):
        drift = (t_anim * (10 + i * 4)) % (w + 300) - 150
        for j in range(3):
            cx = int(drift + j * (w / 2.2) + i * 60) % (w + 300) - 150
            cy = int(h * row)
            for k in range(4):
                rx = 40 + k * 14
                ry = 18
                ox = k * 22 - 33
                cv2.ellipse(overlay, (cx + ox, cy), (rx, ry), 0, 0, 360,
                            tuple(int(c) for c in tint), -1, cv2.LINE_AA)
    canvas[:] = cv2.addWeighted(overlay, opacity, canvas, 1 - opacity, 0)


def generate_city(width, horizon_y, canvas_h, seed=7):
    """Precompute a skyline of buildings (x, w, top_y) plus window grids.
    Called once at startup so the skyline doesn't reshuffle every frame."""
    rng = np.random.default_rng(seed)
    buildings = []
    x = -20
    layer_defs = [

        (0.10, 0.26, 65, 120, 0.55),
        (0.18, 0.38, 60, 115, 0.75),
        (0.26, 0.55, 55, 110, 1.00),
    ]
    for min_hf, max_hf, min_w, max_w, shade in layer_defs:
        x = -30
        layer = []
        while x < width + 30:
            bw = int(rng.uniform(min_w, max_w))
            bh = int(canvas_h * rng.uniform(min_hf, max_hf))
            top_y = horizon_y - bh
            has_antenna = rng.random() < 0.25
            win_w, win_h, gap = 6, 8, 6
            cols = max(1, (bw - gap) // (win_w + gap))
            rows = max(1, (bh - gap) // (win_h + gap))
            lit_mask = rng.random((rows, cols)) < rng.uniform(0.25, 0.6)
            tint = tuple(int(rng.uniform(-10, 10)) for _ in range(3))
            layer.append({
                "x": x, "w": bw, "top": top_y, "h": bh,
                "antenna": has_antenna,
                "cols": cols, "rows": rows,
                "win_w": win_w, "win_h": win_h, "gap": gap,
                "lit_mask": lit_mask,
                "shade": shade,
                "tint": tint,
            })

            x += bw + int(rng.uniform(30, 90))
        buildings.append(layer)
    return buildings


def generate_landmark(width, horizon_y, canvas_h, seed=99):
    """A tall tower with an observation pod and spire, like Tehran's Milad
    Tower in the reference photo -- a single distinctive silhouette rising
    well above the rest of the skyline."""
    rng = np.random.default_rng(seed)
    return {
        "x": int(width * 0.74),
        "height": int(canvas_h * 0.58),
        "pod_frac": 0.72,   
    }


def draw_landmark_tower(canvas, horizon_y, landmark, sun_level, t_anim):
    x = landmark["x"]
    top_y = horizon_y - landmark["height"]
    pod_y = horizon_y - int(landmark["height"] * landmark["pod_frac"])

    tower_col = tuple(int(c) for c in lerp_color((30, 26, 34), (95, 88, 92), sun_level))

    cv2.line(canvas, (x, horizon_y), (x, pod_y), tower_col, 4, cv2.LINE_AA)

    pod_rx, pod_ry = 11, 16
    cv2.ellipse(canvas, (x, pod_y), (pod_rx, pod_ry), 0, 0, 360, tower_col, -1, cv2.LINE_AA)

    if sun_level < 0.5:
        ring_alpha = clamp((0.5 - sun_level) / 0.5)
        pulse = 0.8 + 0.2 * math.sin(t_anim * 1.2)
        ring_col = tuple(int(c * pulse) for c in (150, 220, 90))  
        overlay = canvas.copy()
        cv2.ellipse(overlay, (x, pod_y + 3), (pod_rx + 4, 5), 0, 0, 360, ring_col, 2, cv2.LINE_AA)
        canvas[:] = cv2.addWeighted(overlay, ring_alpha, canvas, 1 - ring_alpha, 0)


    cv2.line(canvas, (x, pod_y - pod_ry), (x, top_y), tower_col, 2, cv2.LINE_AA)

    blink = 0.5 + 0.5 * math.sin(t_anim * 2.5)
    if blink > 0.55:
        cv2.circle(canvas, (x, top_y), 2, (60, 60, 255), -1, cv2.LINE_AA)


def draw_traffic(canvas, horizon_y, t_anim, sun_level):
    
    h, w = canvas.shape[:2]
    lane_y = horizon_y + 5
    visibility = clamp(1.0 - sun_level * 1.3)
    if visibility <= 0.05:
        return

    overlay = canvas.copy()
    streak_len = 46
    
    for i in range(6):
        speed = 220
        offset = (t_anim * speed + i * 130) % (w + streak_len * 2) - streak_len
        x0 = int(offset)
        cv2.line(overlay, (x0, lane_y - 2), (x0 + streak_len, lane_y - 2),
                  (170, 220, 255), 2, cv2.LINE_AA)
    for i in range(6):
        speed = 180
        offset = (w + streak_len) - ((t_anim * speed + i * 150) % (w + streak_len * 2))
        x0 = int(offset)
        cv2.line(overlay, (x0, lane_y + 2), (x0 - streak_len, lane_y + 2),
                  (70, 70, 255), 2, cv2.LINE_AA)

    canvas[:] = cv2.addWeighted(overlay, visibility, canvas, 1 - visibility, 0)

def draw_city(canvas, horizon_y, sun_level, buildings, t_anim):
    h, w = canvas.shape[:2]
    # Night -> day silhouette tint (buildings darker/bluer at night, warmer at day)
    base_dark = lerp_color((18, 14, 22), (55, 48, 55), sun_level)  # BGR-ish neutral
    window_on_color = (60, 210, 255)   # warm lit-window glow (BGR)
    window_day_color = (150, 160, 165) # dim daytime glass reflection (BGR)

    # windows are visibly lit mainly at night/dawn/dusk, fade out by full day
    lit_strength = clamp(1.0 - (sun_level - 0.35) / 0.35) if sun_level > 0.35 else 1.0
    lit_strength = clamp(lit_strength)

    for layer in buildings:
        for b in layer:
            col = tuple(int(clamp(c * b["shade"] + b["tint"][i], 0, 255))
                         for i, c in enumerate(base_dark))
            x0, y0 = b["x"], b["top"]
            x1, y1 = b["x"] + b["w"], horizon_y
            cv2.rectangle(canvas, (x0, y0), (x1, y1), col, -1)

            if b["antenna"]:
                ax = x0 + b["w"] // 2
                cv2.line(canvas, (ax, y0), (ax, y0 - 18), col, 2, cv2.LINE_AA)
                blink = 0.5 + 0.5 * math.sin(t_anim * 3 + x0)
                if sun_level < 0.4 and blink > 0.6:
                    cv2.circle(canvas, (ax, y0 - 18), 2, (50, 50, 255), -1, cv2.LINE_AA)

            # windows
            ww, wh, gap = b["win_w"], b["win_h"], b["gap"]
            rows, cols = b["rows"], b["cols"]
            pad_x = (b["w"] - (cols * ww + (cols - 1) * gap)) // 2
            for r in range(rows):
                wy = y0 + gap + r * (wh + gap)
                if wy + wh > y1:
                    break
                for c in range(cols):
                    wx = x0 + pad_x + c * (ww + gap)
                    is_lit = b["lit_mask"][r, c]
                    if is_lit and lit_strength > 0.05:
                        flicker = 0.85 + 0.15 * math.sin(t_anim * 1.5 + r * 3 + c * 7)
                        wc = tuple(int(cc * flicker) for cc in window_on_color)
                        alpha = lit_strength
                        blended = tuple(int(col[i] * (1 - alpha) + wc[i] * alpha) for i in range(3))
                        cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), blended, -1)
                    elif sun_level > 0.5:
                        alpha = clamp((sun_level - 0.5) / 0.5) * 0.5
                        blended = tuple(int(col[i] * (1 - alpha) + window_day_color[i] * alpha) for i in range(3))
                        cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), blended, -1)

    # thin ground/street line
    street = lerp_color((25, 25, 28), (70, 65, 60), sun_level)
    cv2.rectangle(canvas, (0, horizon_y), (w, horizon_y + 3), (street[2], street[1], street[0]), -1)
    draw_traffic(canvas, horizon_y, t_anim, sun_level)


def draw_hud(canvas, sun_level, dist, phase_label):
    h, w = canvas.shape[:2]
    bar_w = int(w * 0.6)
    bar_x = int(w * 0.2)
    bar_y = h - 40
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (255, 255, 255), 1, cv2.LINE_AA)
    fill_w = int(bar_w * sun_level)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + 14), (0, 215, 255), -1)
    cv2.putText(canvas, phase_label, (bar_x, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"pinch dist: {dist:.3f}", (bar_x, bar_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def phase_label_for(sun_level):
    if sun_level < 0.08:
        return "Midnight"
    if sun_level < 0.28:
        return "Dawn approaching"
    if sun_level < 0.45:
        return "Sunrise"
    if sun_level < 0.7:
        return "Morning"
    if sun_level < 0.92:
        return "Daylight"
    return "High Noon"


def draw_hand_overlay(frame, landmarks_px):
    for conn in HAND_CONNECTIONS:
        a, b = conn.start, conn.end
        cv2.line(frame, landmarks_px[a], landmarks_px[b], (0, 220, 0), 2, cv2.LINE_AA)
    for p in landmarks_px:
        cv2.circle(frame, p, 3, (0, 140, 255), -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (index 0). Try changing the index in cv2.VideoCapture(...).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    landmarker = create_landmarker()

    rng = np.random.default_rng(42)
    stars = [(rng.random(), rng.random(), int(rng.integers(120, 255)), rng.random() * math.tau)
              for _ in range(120)]

    buildings = None  # generated lazily once we know the actual frame size
    landmark = None

    sun_level = 0.5
    flip = True
    fullscreen = False
    win_name = "Sunrise / Sunset Hand Control"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    t0 = time.time()
    last_dist = DIST_MAX

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        if flip:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        if buildings is None:
            horizon_y_init = int(h * 0.62)
            buildings = generate_city(w, horizon_y_init, h)
            landmark = generate_landmark(w, horizon_y_init, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        dist_norm = last_dist
        hand_found = False

        if result.hand_landmarks:
            hand_found = True
            lm = result.hand_landmarks[0]
            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]
            dx = thumb.x - index.x
            dy = thumb.y - index.y
            dist_norm = math.hypot(dx, dy)
            last_dist = dist_norm

            landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm]
            draw_hand_overlay(frame, landmarks_px)

        target = 1.0 - clamp((dist_norm - DIST_MIN) / (DIST_MAX - DIST_MIN))
        sun_level = lerp(sun_level, target, SMOOTHING)

        t_anim = time.time() - t0

        scene = np.zeros_like(frame)
        top_c, hor_c = sky_colors(sun_level)
        horizon_y = int(h * 0.62)
        draw_sky_gradient(scene, top_c, hor_c, horizon_y)
        draw_stars(scene, sun_level, stars, t_anim)
        cx, cy, r, horizon_y, sun_col = draw_sun(scene, sun_level, t_anim)
        draw_clouds(scene, sun_level, horizon_y, t_anim)
        draw_city(scene, horizon_y, sun_level, buildings, t_anim)
        draw_landmark_tower(scene, horizon_y, landmark, sun_level, t_anim)

        if 0.2 < sun_level < 0.5:
            warm_amt = 1 - abs(sun_level - 0.35) / 0.15
            overlay = np.full_like(scene, (40, 90, 255), dtype=np.uint8)
            scene = cv2.addWeighted(overlay, 0.06 * clamp(warm_amt), scene, 1, 0)

        # Top-left corner: the sun's path runs from hidden-bottom-left up to
        # top-right as sun_level increases, so top-left never overlaps it.
        preview = cv2.resize(frame, (int(w * 0.28), int(h * 0.28)))
        ph, pw = preview.shape[:2]
        pad = 12
        scene[pad:pad + ph, pad: pad + pw] = preview
        cv2.rectangle(scene, (pad, pad), (pad + pw, pad + ph), (255, 255, 255), 1)

        if not hand_found:
            cv2.putText(scene, "Show your hand: pinch thumb+index to raise the sun",
                        (pad, pad + ph + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        draw_hud(scene, sun_level, dist_norm, phase_label_for(sun_level))

        cv2.imshow(win_name, scene)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win_name, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        elif key == ord('c'):
            flip = not flip

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
