import cv2
import numpy as np


def compute_dense_optical_flow(video_path=0):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()

    if not ret:
        print("Failed to open camera/video.")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Set saturation to maximum

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute Farneback Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Convert Cartesian flow vectors (dx, dy) to Polar coordinates (magnitude, angle)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Map angle to HSV Hue (0-180 in OpenCV) and magnitude to Value (0-255)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow("Original Frame", frame2)
        cv2.imshow("Dense Optical Flow (HSV Direction)", rgb_flow)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        prvs = next_gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Pass 0 to use live webcam, or "video.mp4" for a video file
    compute_dense_optical_flow(0)