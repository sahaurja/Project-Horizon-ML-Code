"""YOLO26 nano detection + YOLO26 nano depth on your webcam.

Detects objects, estimates how far away each one is, and speaks the top
distinct objects with their distances, e.g. "person, 1.4 meters, and cup,
0.8 meters".

Requires ultralytics >= 8.4.104 (pip install -U ultralytics).

Usage:
    python webcam_depth.py                     # defaults
    python webcam_depth.py --depth-every 5     # run depth less often (faster)
    python webcam_depth.py --show-depth --mute

Press q in the video window to quit.
"""

import argparse
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth


def speech_command(text):
    """Return the system text-to-speech command for this platform."""
    if sys.platform == "darwin":
        return ["say", text]
    if sys.platform == "win32":
        return ["powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                f".Speak('{text}')"]
    for tool in ("espeak-ng", "espeak", "spd-say"):
        if shutil.which(tool):
            return [tool, text]
    raise SystemExit("No text-to-speech tool found; install espeak or use --mute")


def estimate_depth(depth_model, frame):
    """Return a depth map in meters, the same height and width as the frame."""
    result = depth_model(frame, verbose=False)[0]
    depth = result.depth.data.cpu().numpy()
    if depth.shape != frame.shape[:2]:
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
    return depth


def box_distance(depth_map, box):
    """Median depth over the central half of a box (avoids background edges)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    patch = depth_map[max(y1 + h // 4, 0):max(y2 - h // 4, 1),
                      max(x1 + w // 4, 0):max(x2 - w // 4, 1)]
    patch = patch[patch > 0]  # ignore invalid pixels
    if patch.size == 0:
        return None
    return float(np.median(patch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence")
    parser.add_argument("--top", type=int, default=3,
                        help="how many objects to speak")
    parser.add_argument("--repeat", type=float, default=5.0,
                        help="seconds before repeating the same object")
    parser.add_argument("--depth-model", default="yolo26n-depth.pt",
                        help="depth checkpoint (n/s/m/l/x, or a calibrated one)")
    parser.add_argument("--depth-every", type=int, default=3,
                        help="run depth every N frames (reuses the last map between)")
    parser.add_argument("--show-depth", action="store_true",
                        help="also show a colorized depth window")
    parser.add_argument("--mute", action="store_true", help="disable speech")
    args = parser.parse_args()

    detector = YOLO("yolo26n.pt")
    depth_model = YOLO(args.depth_model)  # downloads automatically on first run

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}")

    speaker = None
    last_spoken = None   # set of labels most recently spoken
    last_time = 0.0
    depth_map = None
    frame_idx = 0
    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector(frame, conf=args.conf, verbose=False)[0]
        boxes = result.boxes

        # Depth is the slower model, so refresh it only every N frames,
        # and only when there is something detected to measure.
        if len(boxes) and (depth_map is None or frame_idx % args.depth_every == 0):
            depth_map = estimate_depth(depth_model, frame)

        xyxy_all = boxes.xyxy.cpu().numpy()
        distances = [box_distance(depth_map, b) if depth_map is not None else None
                     for b in xyxy_all]

        found = []
        for c, p, d in zip(boxes.cls, boxes.conf, distances):
            dist = f", {d:.1f} m" if d is not None else ""
            found.append(f"{result.names[int(c)]} ({float(p):.2f}{dist})")
        if found:
            print(", ".join(found))

        # Speak the top distinct objects by confidence, with their distances,
        # without blocking the video. Skip while still speaking, so
        # announcements never pile up.
        if not args.mute and len(boxes):
            labels, idxs = [], []
            for i in boxes.conf.argsort(descending=True).tolist():
                name = result.names[int(boxes.cls[i])]
                if name not in labels:
                    labels.append(name)
                    idxs.append(i)
                if len(labels) == args.top:
                    break
            # Compare as a set, so a mere reordering is not re-announced.
            now = time.perf_counter()
            idle = speaker is None or speaker.poll() is not None
            due = set(labels) != last_spoken or now - last_time >= args.repeat
            if idle and due:
                parts = []
                for label, i in zip(labels, idxs):
                    d = distances[i]
                    parts.append(f"{label}, {d:.1f} meters" if d is not None else label)
                if len(parts) > 1:
                    text = ", ".join(parts[:-1]) + ", and " + parts[-1]
                else:
                    text = parts[0]
                speaker = subprocess.Popen(speech_command(text))
                last_spoken, last_time = set(labels), now

        # Draw boxes, distances, and frame rate.
        annotated = result.plot()
        for b, d in zip(xyxy_all, distances):
            if d is not None:
                cv2.putText(annotated, f"{d:.1f} m", (int(b[0]) + 4, int(b[3]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        now = time.perf_counter()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(annotated, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO26 + depth (q to quit)", annotated)

        if args.show_depth and depth_map is not None:
            cv2.imshow("Depth", colorize_depth(depth_map, cmap="inferno"))

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if speaker is not None and speaker.poll() is None:
        speaker.terminate()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()