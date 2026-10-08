import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
path = "images/single.jpg"
if os.path.exists(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
else:
    from skimage import data
    img = data.astronaut()
    print("images/single.jpg not found, using sample image")
scale = 400 / max(img.shape[:2])
img = cv2.resize(img, None, fx=scale, fy=scale)
h, w = img.shape[:2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
with torch.no_grad():
    pred = midas(transform(img).to(device))
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h, w), mode="bicubic",
        align_corners=False).squeeze().cpu().numpy()
inv = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)   # 0 far ... 1 near
Z = 1.0 / (inv + 0.2)                      # larger Z = farther away
f = max(h, w)                              # assumed focal length (pixels)
cx, cy = w / 2, h / 2
u, v = np.meshgrid(np.arange(w), np.arange(h))
X = (u - cx) * Z / f
Y = (v - cy) * Z / f
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img); ax1.set_title("Input Image"); ax1.axis("off")
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(inv, cmap="inferno"); ax2.set_title("Estimated Depth Map"); ax2.axis("off")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
s = 3                                      # subsample for speed
ax3.scatter(X[::s, ::s].ravel(), Z[::s, ::s].ravel(), -Y[::s, ::s].ravel(),
            c=img[::s, ::s].reshape(-1, 3) / 255.0, s=2)
ax3.set_title("3D Point Cloud")
ax3.set_xlabel("X"); ax3.set_ylabel("Z (depth)"); ax3.set_zlabel("Y")
ax3.view_init(elev=15, azim=-70)
plt.tight_layout()
plt.savefig("single_image_3d_output.png", dpi=150)
plt.show()