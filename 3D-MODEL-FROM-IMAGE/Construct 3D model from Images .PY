import cv2
import numpy as np


def extract_and_match_features(img1, img2):
    """Detect SIFT keypoints and match them using FLANN-based matcher."""
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test to filter good matches
    good_pts1, good_pts2 = [], []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_pts1.append(kp1[m.queryIdx].pt)
            good_pts2.append(kp2[m.trainIdx].pt)

    return np.float32(good_pts1), np.float32(good_pts2)


def reconstruct_3d_points(pts1, pts2, K):
    """Estimate camera pose and triangulate 3D point cloud."""
    # Find Essential Matrix using RANSAC
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    # Select inlier points
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    # Recover relative rotation (R) and translation (t)
    _, R, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, K)

    # Projection matrix for Camera 1: [I | 0]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Projection matrix for Camera 2: [R | t]
    P2 = K @ np.hstack((R, t))

    # Triangulate 3D points from 2D correspondences
    pts4D = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)

    # Convert homogeneous coordinates (X, Y, Z, W) to 3D (X, Y, Z)
    pts3D = pts4D[:3] / pts4D[3]

    return pts3D.T


def export_to_ply(filename, points):
    """Save 3D point cloud as a PLY file for Blender / MeshLab."""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        for pt in points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
    print(f"3D Point Cloud saved successfully as '{filename}'.")


# --- Pipeline Execution ---
if __name__ == "__main__":
    # Load stereo or overlapping image pair
    img1 = cv2.imread("view1.jpg")
    img2 = cv2.imread("view2.jpg")

    if img1 is None or img2 is None:
        print("Please provide valid image paths 'view1.jpg' and 'view2.jpg'.")
    else:
        # Define Camera Intrinsic Matrix K (focal length f, optical center cx, cy)
        h, w = img1.shape[:2]
        focal_length = w * 1.2  # Approximate focal length if EXIF unavailable
        K = np.array(
            [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]]
        )

        # 1. Feature Matching
        pts1, pts2 = extract_and_match_features(img1, img2)

        # 2. Triangulate 3D Points
        points3d = reconstruct_3d_points(pts1, pts2, K)

        # 3. Export to PLY
        export_to_ply("reconstructed_model.ply", points3d)