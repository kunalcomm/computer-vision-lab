import cv2
import numpy as np
SOURCE = 0            # 0 = webcam, or "videos/surveillance.mp4"
MIN_AREA = 800        # ignore small blobs (noise)
cap = cv2.VideoCapture(SOURCE)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("Press 's' to save a screenshot, 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mask = bg_subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Moving Object", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        count += 1
    cv2.putText(frame, f"Objects: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, mask_bgr))
    cv2.imshow("Detection (left) | Foreground Mask (right)", combined)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        cv2.imwrite("surveillance_output.jpg", combined)
        print("Saved surveillance_output.jpg")
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()