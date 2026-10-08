# Computer Vision Lab Experiments

A collection of 15 computer vision experiments implemented in Python with OpenCV, NumPy, Matplotlib and PyTorch.
Each experiment has its own script, and the full lab record (aim, theory, algorithm, code, output, observation and result) is in `LAB_RECORD_CV.docx`.

**Author:** Kunal  |  **Institution:** VIT Bhopal University

---

## Experiments

| No. | Experiment | Main technique |
|-----|-----------|----------------|
| 1 | Image preprocessing and edge detection | Smoothing/filtering, Sobel and Canny edge detection |
| 2 | Camera calibration methods | Chessboard corners, `cv2.calibrateCamera`, intrinsic matrix and distortion |
| 3 | Projection | Perspective projection of 3D points onto the image plane |
| 4 | Depth map from a stereo pair | Block-matching disparity and depth map |
| 5 | Construct 3D model from stereo pair | Disparity map, depth `Z = fB/d`, point cloud |
| 6 | Segmentation methods | Global / Otsu / adaptive thresholding, K-means |
| 7 | Construct 3D model from defocus images | Laplacian-variance focus measure, depth from defocus |
| 8 | 3D model construction from multiple images | SIFT, ratio-test matching, fundamental matrix, triangulation |
| 9 | Optical flow | Shi-Tomasi corners + Lucas-Kanade tracking |
| 10 | Object detection and tracking in video | YOLO (Ultralytics) + ByteTrack |
| 11 | Face detection and recognition | Haar cascade detection, LBPH recognition |
| 12 | Object detection from dynamic background (surveillance) | MOG2 background subtraction, morphology, contours |
| 13 | Content-based video retrieval | Shot detection, key frames, HSV histogram matching |
| 14 | Construct 3D model from a single image | MiDaS monocular depth + back-projection to point cloud |

---

## Repository structure

```
CV/
├── README.md
├── requirements.txt
├── .gitignore
├── LAB_RECORD_CV.docx        # full lab record with code and output screenshots
├── <one folder or script per experiment>
├── images/                   # sample input images
└── videos/                   # sample input videos (small files only)
```

## Setup

Python 3.9+ is recommended.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Important:** use OpenCV **4.x** (`opencv-contrib-python`). OpenCV 5.0 moved `CascadeClassifier`, which breaks Experiment 11.
Install only one OpenCV package; having several installed together causes import errors.

## Running an experiment

Run each script from the repository root so relative paths such as `images/...` and `videos/...` resolve correctly:

```bash
python "Face Detection and Recognition.py"
```

Notes:
- Experiments 11 and 12 use the **webcam** by default. Press `q` to quit and `s` to save a screenshot.
- Experiment 13 generates its own demo videos on the first run if none are present.
- Experiment 14 downloads the MiDaS model (about 100 MB) on first run and needs an internet connection. When prompted to trust the `rwightman/gen-efficientnet-pytorch` repository, answer `y`.
- Experiment 10 needs a YOLO weights file such as `yolov8n.pt` (downloaded automatically by Ultralytics).

## Requirements

See `requirements.txt`. Main libraries: OpenCV (contrib), NumPy, Matplotlib, scikit-image, PyTorch, timm, Ultralytics.

## Results

Output screenshots for every experiment are included in the lab record (`LAB_RECORD_CV.docx`).

## License

For educational use.
