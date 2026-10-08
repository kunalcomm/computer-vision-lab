# 24BAI10443
import cv2
import numpy as np
import os

input_path = 'input.jpg'

# ---------------------------------------------------------
# Step 1: Generate synthetic input image if missing
# ---------------------------------------------------------
if not os.path.exists(input_path):
    print("'input.jpg' not found. Generating a synthetic test image...")
    # Create a 400x400 image with geometric shapes
    img_synth = np.zeros((400, 400), dtype=np.uint8) + 50
    cv2.rectangle(img_synth, (50, 50), (200, 200), 200, -1)
    cv2.circle(img_synth, (280, 280), 70, 255, -1)
    cv2.putText(img_synth, 'OPENCV', (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    
    cv2.imwrite(input_path, img_synth)

# ---------------------------------------------------------
# Step 2: Load Image in Grayscale
# ---------------------------------------------------------
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# Verify image loaded properly
if img is None:
    raise FileNotFoundError(f"Could not load image at {input_path}")

# ---------------------------------------------------------
# Step 3: Preprocessing & Edge Detection
# ---------------------------------------------------------
# Preprocessing: Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

# Edge Detection: Canny
canny_edges = cv2.Canny(blurred, 100, 200)

# Edge Detection: Sobel (X and Y gradients)
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobelx, sobely)

# Normalize Sobel magnitude to 8-bit image for saving
sobel_normalized = cv2.normalize(
    sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

# ---------------------------------------------------------
# Step 4: Save Outputs
# ---------------------------------------------------------
cv2.imwrite('canny_edges.jpg', canny_edges)
cv2.imwrite('sobel_edges.jpg', sobel_normalized)

print("--- Edge Detection Successful ---")
print("Saved 'canny_edges.jpg' and 'sobel_edges.jpg'")