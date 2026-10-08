import cv2
import numpy as np


def detect_and_track_objects(source=0):
    # Initialize video capture (0 for webcam, or pass 'video.mp4')
    cap = cv2.VideoCapture(source)

    # 1. Background Subtractor for Detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )

    # 2. Tracking variables
    tracker = None
    tracking = False
    bbox = None

    print("[INFO] Press 's' to select a custom object to track.")
    print("[INFO] Press 'r' to reset detection/tracking.")
    print("[INFO] Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # --- MODE 1: Active Tracking ---
        if tracking:
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                # Draw green bounding box for tracked object
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    "Status: Tracking",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Status: Tracking Lost",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # --- MODE 2: Automated Object Detection ---
        else:
            # Apply Background Subtraction
            fg_mask = bg_subtractor.apply(frame)

            # Threshold & remove noise
            _, thresh = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            # Find object contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                # Filter out small noisy movements
                if cv2.contourArea(cnt) < 1500:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                # Draw red bounding box for detected moving objects
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    "Detected Motion",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Status: Detecting Motion (Press 's' to track)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        cv2.imshow("Object Detection & Tracking", frame)

        key = cv2.waitKey(30) & 0xFF
        # Press 'q' to quit
        if key == ord("q"):
            break
        # Press 's' to manually select a region of interest (ROI) to track
        elif key == ord("s"):
            bbox = cv2.selectROI(
                "Object Detection & Tracking", frame, fromCenter=False
            )
            if bbox[2] > 0 and bbox[3] > 0:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
                tracking = True
        # Press 'r' to reset tracker back to detection mode
        elif key == ord("r"):
            tracking = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_track_objects(0)