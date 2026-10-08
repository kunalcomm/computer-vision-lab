import cv2
import numpy as np
import os

left_path = 'stereo_left.png'
right_path = 'stereo_right.png'

# ---------------------------------------------------------
# Step 1: Generate synthetic stereo pair if missing
# ---------------------------------------------------------
if not os.path.exists(left_path) or not os.path.exists(right_path):
    print("Stereo images not found. Generating synthetic test images...")
    
    # Create base background (left image)
    img_left = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(img_left, (50, 50), (350, 250), 100, -1)
    
    # Draw an object in the foreground (rectangle)
    cv2.rectangle(img_left, (150, 100), (250, 200), 255, -1)
    cv2.putText(img_left, 'Object', (160, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    # Right image: Shift the foreground object left to simulate parallax (disparity)
    img_right = img_left.copy()
    cv2.rectangle(img_right, (150, 100), (250, 200), 100, -1) # Clear old spot
    cv2.rectangle(img_right, (130, 100), (230, 200), 255, -1) # Shifted 20px left
    cv2.putText(img_right, 'Object', (140, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    cv2.imwrite(left_path, img_left)
    cv2.imwrite(right_path, img_right)

# ---------------------------------------------------------
# Step 2: Load images in Grayscale
# ---------------------------------------------------------
img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

# ---------------------------------------------------------
# Step 3: Compute Disparity Map
# ---------------------------------------------------------
# numDisparities must be positive and divisible by 16
# blockSize (SADWindowSize) must be an odd integer between 5 and 255
stereo = cv2.StereoBM_create(numDisparities=16*2, blockSize=15)

disparity = stereo.compute(img_left, img_right)

# Normalize disparity map to range 0-255 for visualization
disparity_normalized = cv2.normalize(
    disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

# ---------------------------------------------------------
# Step 4: Calculate Depth Map
# ---------------------------------------------------------
focal_length = 800.0  # in pixels
baseline = 0.1        # in meters (10 cm)

# StereoBM scales raw disparity by 16; scale it back
disparity_float = disparity.astype(np.float32) / 16.0
disparity_float[disparity_float <= 0] = 0.1  # Avoid division by zero

# Depth Z = (focal_length * baseline) / disparity
depth_map = (focal_length * baseline) / disparity_float

# Save output
cv2.imwrite('disparity_map.png', disparity_normalized)

print("--- Depth Map Generation Successful ---")
print("Output saved as 'disparity_map.png'")