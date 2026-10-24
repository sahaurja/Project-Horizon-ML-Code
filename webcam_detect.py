"""Run YOLO26 nano locally on your webcam and identify objects.

The three most confident objects in view are spoken aloud, e.g. "person, cup,
and laptop". They are announced when the set changes, and repeated every
--repeat seconds while it stays the same.

Usage:
    python webcam_detect.py            # default webcam
    python webcam_detect.py --camera 1 # a different camera
    python webcam_detect.py --mute     # no speech

Press q in the video window to quit.

"""

import argparse
import shutil
import subprocess
import sys
import time

import cv2
from ultralytics import YOLO


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence")
    parser.add_argument("--top", type=int, default=3,
                        help="how many objects to speak")
    parser.add_argument("--repeat", type=float, default=5.0,
                        help="seconds before repeating the same object")
    parser.add_argument("--mute", action="store_true", help="disable speech")
    args = parser.parse_args()

    # USe nano version
    model = YOLO("yolo26n.pt")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}")

    speaker = None       # the running speech process, if any
    last_spoken = None   # set of labels most recently spoken
    last_time = 0.0      # when it was spoken

    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=args.conf, verbose=False)[0]

        # Print the objects found in this frame, e.g. "person (0.91), cup (0.62)"
        found = [f"{result.names[int(c)]} ({float(p):.2f})"
                 for c, p in zip(result.boxes.cls, result.boxes.conf)]
        if found:
            print(", ".join(found))

        # Speak the top distinct objects by confidence, without blocking the
        # video. Skip while still speaking, so announcements never pile up.
        if not args.mute and len(result.boxes):
            labels = []
            for i in result.boxes.conf.argsort(descending=True).tolist():
                name = result.names[int(result.boxes.cls[i])]
                if name not in labels:
                    labels.append(name)
                if len(labels) == args.top:
                    break
            # Compare as a set, so a mere reordering is not re-announced.
            now = time.perf_counter()
            idle = speaker is None or speaker.poll() is not None
            due = set(labels) != last_spoken or now - last_time >= args.repeat
            if idle and due:
                if len(labels) > 1:
                    text = ", ".join(labels[:-1]) + ", and " + labels[-1]
                else:
                    text = labels[0]
                speaker = subprocess.Popen(speech_command(text))
                last_spoken, last_time = set(labels), now

        # Draw boxes and labels, plus the frame rate.
        annotated = result.plot()
        now = time.perf_counter()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(annotated, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO26 nano (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if speaker is not None and speaker.poll() is None:
        speaker.terminate()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
