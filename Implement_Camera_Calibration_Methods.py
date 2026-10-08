import numpy as np
import cv2
import glob
import os

# ---------------------------------------------------------
# Step 1: Ensure image directory and test files exist
# ---------------------------------------------------------
os.makedirs('calibration_images', exist_ok=True)

# Search for existing images in the folder
images = glob.glob('calibration_images/*.jpg') + \
         glob.glob('calibration_images/*.png') + \
         glob.glob('calibration_images/*.jpeg')

# If no images exist, generate synthetic checkerboard images to run the demo
if len(images) == 0:
    print("No images found in 'calibration_images/'. Generating synthetic test images...")
    square_size = 50
    board = np.zeros((7 * square_size, 10 * square_size), dtype=np.uint8)
    for i in range(7):
        for j in range(10):
            if (i + j) % 2 == 0:
                board[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255

    for idx in range(3):
        # Apply minor shifts to simulate different camera angles
        M = np.float32([[1, 0, idx * 5], [0, 1, idx * 3]])
        transformed = cv2.warpAffine(board, M, (board.shape[1] + 20, board.shape[0] + 20))
        path = f'calibration_images/test_{idx}.jpg'
        cv2.imwrite(path, transformed)
        images.append(path)

# ---------------------------------------------------------
# Step 2: Set up Calibration Parameters
# ---------------------------------------------------------
# Grid of inner corners (cols, rows)
CHECKERBOARD = (6, 9)

# Termination criteria for sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points in world coordinates (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

image_shape = None

# ---------------------------------------------------------
# Step 3: Find Corners in Images
# ---------------------------------------------------------
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_shape = gray.shape[::-1]  # (width, height)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If corners found, refine and store
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
    else:
        # Try transposed dimensions if standard direction fails
        alt_board = (CHECKERBOARD[1], CHECKERBOARD[0])
        ret_alt, corners_alt = cv2.findChessboardCorners(gray, alt_board, None)
        if ret_alt:
            alt_objp = np.zeros((alt_board[0] * alt_board[1], 3), np.float32)
            alt_objp[:, :2] = np.mgrid[0:alt_board[0], 0:alt_board[1]].T.reshape(-1, 2)
            objpoints.append(alt_objp)
            corners2 = cv2.cornerSubPix(gray, corners_alt, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

# ---------------------------------------------------------
# Step 4: Perform Camera Calibration
# ---------------------------------------------------------
if len(objpoints) > 0 and image_shape is not None:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    print("--- Calibration Successful ---")
    print("Camera Matrix (Intrinsic parameters):\n", mtx)
    print("\nDistortion Coefficients:\n", dist)
else:
    print("Error: Could not detect corners in any image.")
    print("Please verify that inner corner counts match CHECKERBOARD = (6, 9).")