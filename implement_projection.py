import numpy as np
import cv2

# Define intrinsic matrix (K)
K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0,   0,   1]], dtype=np.float32)

# Rotation vector and Translation vector (Extrinsic)
rvec = np.array([0.1, 0.2, 0.0], dtype=np.float32)
tvec = np.array([10.0, 5.0, 50.0], dtype=np.float32)

# 3D points in world coordinates
object_points = np.array([
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [10.0, 10.0, 10.0]
], dtype=np.float32)

# Project 3D points to 2D pixel coordinates
image_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, distCoeffs=None)

print("Projected 2D Pixel Coordinates:\n", image_points.reshape(-1, 2))