import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- 1. SYNTHETIC IMAGE GENERATOR ----------------
def make_texture(seed, size=512):
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size, 3), np.uint8)
    for scale in (8, 32, 128):
        small = rng.integers(0, 255, (size // scale, size // scale, 3), np.uint8)
        tex = cv2.addWeighted(tex, 0.6, cv2.resize(small, (size, size),
                              interpolation=cv2.INTER_CUBIC), 0.7, 0)
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = tuple(int(v) for v in rng.integers(20, size - 20, 2))
        if rng.random() < 0.5:
            cv2.circle(tex, p, int(rng.integers(5, 25)), c, -1)
        else:
            q = (p[0] + int(rng.integers(10, 50)), p[1] + int(rng.integers(10, 50)))
            cv2.rectangle(tex, p, q, c, -1)
    for _ in range(12):
        p = tuple(int(v) for v in rng.integers(30, size - 60, 2))
        cv2.putText(tex, str(int(rng.integers(10, 99))), p, cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 3)
    return tex


def look_at(cam_pos, target=(0, 0, 0)):
    cam_pos = np.array(cam_pos, float)
    f = np.array(target, float) - cam_pos
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.vstack([right, down, f])
    return R, -R @ cam_pos


def generate_synthetic_images(n_views=6, w=800, h=600):
    os.makedirs(IMG_DIR, exist_ok=True)
    K = np.array([[800, 0, w / 2], [0, 800, h / 2], [0, 0, 1]], float)
    s = 1.0
    faces = {
        "+x": ([[s, -s, -s], [s, -s, s], [s, s, s], [s, s, -s]], [1, 0, 0]),
        "-x": ([[-s, -s, s], [-s, -s, -s], [-s, s, -s], [-s, s, s]], [-1, 0, 0]),
        "+y": ([[-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s]], [0, 1, 0]),
        "-y": ([[-s, -s, s], [s, -s, s], [s, -s, -s], [-s, -s, -s]], [0, -1, 0]),
        "+z": ([[-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]], [0, 0, 1]),
        "-z": ([[s, -s, -s], [-s, -s, -s], [-s, s, -s], [s, s, -s]], [0, 0, -1]),
    }
    textures = {k: make_texture(i + 1) for i, k in enumerate(faces)}
    tsz = 512
    tex_corners = np.float32([[0, 0], [tsz, 0], [tsz, tsz], [0, tsz]])
    angles = np.linspace(25, 25 + 12 * (n_views - 1), n_views)
    images = []
    for i, a in enumerate(angles):
        r = np.deg2rad(a)
        pos = (6 * np.sin(r), 3.0, 6 * np.cos(r))
        R, t = look_at(pos)
        canvas = np.full((h, w, 3), 40, np.uint8)
        order = []
        for k, (corners, normal) in faces.items():
            c = np.mean(corners, axis=0)
            if np.dot(normal, np.array(pos) - c) > 0:
                order.append((np.linalg.norm(np.array(pos) - c), k))
        for _, k in sorted(order, reverse=True):
            corners = np.array(faces[k][0], float)
            cam = (R @ corners.T).T + t
            px = (K @ cam.T).T
            px = (px[:, :2] / px[:, 2:3]).astype(np.float32)
            H = cv2.getPerspectiveTransform(tex_corners, px)
            warped = cv2.warpPerspective(textures[k], H, (w, h))
            mask = cv2.warpPerspective(np.full((tsz, tsz), 255, np.uint8), H, (w, h))
            canvas[mask > 0] = warped[mask > 0]
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        cv2.imwrite(os.path.join(IMG_DIR, f"view{i + 1}.jpg"), canvas)
        images.append(canvas)
    print(f"[info] No images found -> generated {n_views} test images in: {IMG_DIR}")
    return images


# ---------------- 2. LOAD IMAGES ----------------
def load_images():
    paths = []
    if os.path.isdir(IMG_DIR):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            paths += glob.glob(os.path.join(IMG_DIR, ext))
    paths = sorted(set(paths))
    if len(paths) < 2:
        generate_synthetic_images()
        paths = sorted(glob.glob(os.path.join(IMG_DIR, "view*.jpg")))
    imgs = [cv2.imread(p) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if len(imgs) < 2:
        sys.exit("Need at least 2 readable images.")
    out = []
    for im in imgs:
        scale = 1200 / max(im.shape[:2])
        out.append(cv2.resize(im, None, fx=scale, fy=scale) if scale < 1 else im)
    print(f"[info] Using {len(out)} images from: {IMG_DIR}")
    return out


# ---------------- 3. RECONSTRUCTION ----------------
def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    return sift.detectAndCompute(gray, None)


def match(d1, d2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    return [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < ratio * n.distance]


def reconstruct(images):
    h, w = images[0].shape[:2]
    f = 1.2 * max(h, w)
    if (w, h) == (800, 600):
        f = 800.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)

    feats = [detect(im) for im in images]

    kp1, d1 = feats[0]
    kp2, d2 = feats[1]
    good = match(d1, d2, ratio=0.8)
    if len(good) < 30:
        sys.exit("Too few matches between image 1 and 2. Use photos with more overlap/texture.")
    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, inl = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, inl2 = cv2.recoverPose(E, p1, p2, K, mask=inl)
    keep = inl2.ravel() > 0
    p1, p2 = p1[keep], p2[keep]
    good = [g for g, k in zip(good, keep) if k]

    cv2.imwrite(os.path.join(OUT_DIR, "matches.png"),
                cv2.drawMatches(images[0], kp1, images[1], kp2, good[:80], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    X = (X[:3] / X[3]).T

    def reproj_err(P, X, p):
        xh = (P @ np.hstack([X, np.ones((len(X), 1))]).T).T
        return np.linalg.norm(xh[:, :2] / xh[:, 2:3] - p, axis=1)

    z1 = X[:, 2]
    z2 = (R @ X.T + t).T[:, 2]
    ok = (z1 > 0) & (z2 > 0) & (reproj_err(P1, X, p1) < 2) & (reproj_err(P2, X, p2) < 2)
    pts = X[ok]
    cols = np.array([images[0][int(round(y)), int(round(x))][::-1] for x, y in p1[ok]])

    kp_to_3d = {}
    for j, gi in enumerate(np.where(ok)[0]):
        kp_to_3d[good[gi].trainIdx] = j
    cams = [(np.eye(3), np.zeros((3, 1))), (R, t)]
    print(f"[info] Views 1+2: {len(pts)} 3D points")

    prev = 1
    for vi in range(2, len(images)):
        kp_p, d_p = feats[prev]
        kp, d = feats[vi]
        mm = match(d_p, d)

        obj, img_pts = [], []
        for m in mm:
            if m.queryIdx in kp_to_3d:
                obj.append(pts[kp_to_3d[m.queryIdx]])
                img_pts.append(kp[m.trainIdx].pt)
        if len(obj) < 10:
            print(f"[warn] View {vi + 1}: only {len(obj)} 2D-3D matches, skipped")
            continue
        okp, rvec, tvec, inl = cv2.solvePnPRansac(
            np.float32(obj), np.float32(img_pts), K, None,
            reprojectionError=3.0, iterationsCount=300)
        if not okp or inl is None or len(inl) < 8:
            print(f"[warn] View {vi + 1}: pose estimation failed, skipped")
            continue
        Rn, _ = cv2.Rodrigues(rvec)
        Pp = K @ np.hstack(cams[-1])
        Pn = K @ np.hstack([Rn, tvec])
        cams.append((Rn, tvec))
        print(f"[info] View {vi + 1}: registered with {len(inl)} inliers")

        q = np.float32([kp_p[m.queryIdx].pt for m in mm])
        r = np.float32([kp[m.trainIdx].pt for m in mm])
        Xn = cv2.triangulatePoints(Pp, Pn, q.T, r.T)
        Xn = (Xn[:3] / Xn[3]).T
        zp = (cams[-2][0] @ Xn.T + cams[-2][1]).T[:, 2]
        zn = (Rn @ Xn.T + tvec).T[:, 2]
        good_n = (zp > 0) & (zn > 0) & (reproj_err(Pp, Xn, q) < 2) & (reproj_err(Pn, Xn, r) < 2)

        new_map = {}
        base = len(pts)
        add_pts, add_cols = [], []
        for k in np.where(good_n)[0]:
            old = kp_to_3d.get(mm[k].queryIdx)
            if old is not None:
                new_map[mm[k].trainIdx] = old
            else:
                new_map[mm[k].trainIdx] = base + len(add_pts)
                add_pts.append(Xn[k])
                x, y = r[k]
                add_cols.append(images[vi][int(round(y)), int(round(x))][::-1])
        if add_pts:
            pts = np.vstack([pts, np.array(add_pts)])
            cols = np.vstack([cols, np.array(add_cols)])
        kp_to_3d = new_map
        prev = vi
        print(f"        total points so far: {len(pts)}")

    centre = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - centre, axis=1)
    inliers = dist < np.percentile(dist, 97)
    return pts[inliers], cols[inliers], cams


# ---------------- 4. SAVE + VIEW ----------------
def save_ply(path, pts, cols):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def show(pts, cols, cams):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols / 255.0, s=3)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("-Y")
    ax.set_title(f"Reconstructed 3D point cloud ({len(pts)} points, {len(cams)} cameras)")
    mid = np.median(pts, axis=0)
    span = np.percentile(np.abs(pts - mid), 98) * 1.2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[2] - span, mid[2] + span)
    ax.set_zlim(-mid[1] - span, -mid[1] + span)
    fig.savefig(os.path.join(OUT_DIR, "preview.png"), dpi=120)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def main():
    images = load_images()
    pts, cols, cams = reconstruct(images)
    print(f"[done] Final point cloud: {len(pts)} points, {len(cams)} cameras registered")
    ply = os.path.join(OUT_DIR, "model.ply")
    save_ply(ply, pts, cols)
    print(f"[saved] {ply}")
    print(f"[saved] {os.path.join(OUT_DIR, 'matches.png')}")
    show(pts, cols, cams)
    print(f"[saved] {os.path.join(OUT_DIR, 'preview.png')}")


if __name__ == "__main__":
    main()