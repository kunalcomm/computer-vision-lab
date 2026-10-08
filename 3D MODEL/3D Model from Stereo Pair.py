import cv2
import numpy as np
import os

# images are read from the same folder as this file
folder = os.path.dirname(os.path.abspath(__file__))

# read left and right images
left = cv2.imread(os.path.join(folder, "left.jpg"))
right = cv2.imread(os.path.join(folder, "right.jpg"))

if left is None or right is None:
    print("Images not found")
    exit()

# convert to gray
grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# disparity map
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5)
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# convert to 3D points
h, w = grayL.shape
f = 0.8 * w
Q = np.float32([[1, 0, 0, -w / 2],
                [0, -1, 0, h / 2],
                [0, 0, 0, -f],
                [0, 0, 1, 0]])
points = cv2.reprojectImageTo3D(disparity, Q)

# save 3D model as ply file
colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
mask = disparity > 1
points = points[mask]
colors = colors[mask]

with open(os.path.join(folder, "model.ply"), "w") as file:
    file.write("ply\nformat ascii 1.0\n")
    file.write("element vertex " + str(len(points)) + "\n")
    file.write("property float x\nproperty float y\nproperty float z\n")
    file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    file.write("end_header\n")
    for p, c in zip(points, colors):
        file.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

# show disparity
disp_img = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(os.path.join(folder, "disparity.png"), disp_img)
cv2.imshow("Disparity Map", disp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done. Saved disparity.png and model.ply")