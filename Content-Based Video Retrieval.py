import cv2
import numpy as np
import os
import glob
DB_DIR, QUERY = "videos/database", "videos/query.mp4"
os.makedirs(DB_DIR, exist_ok=True)
def make_video(path, colors, shape="circle", frames_per_shot=30):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (160, 120))
    rng = np.random.default_rng(abs(hash(path)) % 1000)
    t = 0
    for color in colors:                        # each color = one shot
        for _ in range(frames_per_shot):
            f = np.full((120, 160, 3), color, np.uint8)
            f = cv2.add(f, rng.integers(0, 15, f.shape, dtype=np.uint8))
            x = 20 + (t * 3) % 120
            if shape == "circle":
                cv2.circle(f, (x, 60), 12, (255, 255, 255), -1)
            else:
                cv2.rectangle(f, (x - 12, 48), (x + 12, 72), (255, 255, 255), -1)
            out.write(f)
            t += 1
    out.release()
if not glob.glob(os.path.join(DB_DIR, "*.mp4")):
    demo = {
        "red_orange": [(0, 0, 200), (0, 100, 230)],
        "green_cyan": [(0, 180, 0), (180, 200, 0)],
        "blue_purple": [(200, 0, 0), (200, 0, 150)],
        "yellow_white": [(0, 220, 220), (230, 230, 230)],
        "dark_scene": [(30, 30, 30), (60, 60, 90)],
    }
    for name, cols in demo.items():
        make_video(f"{DB_DIR}/{name}.mp4", cols)
    print("Demo database created.")
if not os.path.exists(QUERY):
    make_video(QUERY, [(20, 20, 190), (10, 110, 220)], shape="square")
    print("Demo query created.")
def histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                     [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()
def extract_keyframes(path, shot_threshold=0.8):
    """Detect shots by histogram difference; return the middle frame of each shot."""
    cap = cv2.VideoCapture(path)
    frames, hists = [], []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        hists.append(histogram(f))
    cap.release()
    bounds = [0]
    for i in range(1, len(hists)):
        sim = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL)
        if sim < shot_threshold:               # abrupt change = new shot
            bounds.append(i)
    bounds.append(len(frames))
    keyframes = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        keyframes.append(frames[(s + e) // 2])
    return keyframes
database = {}
for path in sorted(glob.glob(os.path.join(DB_DIR, "*.mp4"))):
    kfs = extract_keyframes(path)
    database[path] = {"keyframes": kfs, "feats": [histogram(k) for k in kfs]}
    print(f"{os.path.basename(path):20s} shots/keyframes: {len(kfs)}")
q_kfs = extract_keyframes(QUERY)
q_feats = [histogram(k) for k in q_kfs]
print("Query keyframes:", len(q_kfs))
def video_similarity(q, d):
    scores = [max(cv2.compareHist(a, b, cv2.HISTCMP_CORREL) for b in d)
              for a in q]
    return float(np.mean(scores))
results = sorted(((video_similarity(q_feats, v["feats"]), p)
                  for p, v in database.items()), reverse=True)
print("\nRanked results:")
for rank, (score, path) in enumerate(results, 1):
    print(f"{rank}. {os.path.basename(path):20s} similarity = {score:.3f}")
def tile(img, text):
    img = cv2.resize(img, (240, 180)).copy()
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return img
rows = [np.hstack([tile(q_kfs[0], "QUERY")] +
                  [tile(q_kfs[min(1, len(q_kfs) - 1)], "QUERY shot 2")])]
for rank, (score, path) in enumerate(results[:3], 1):
    kfs = database[path]["keyframes"]
    name = os.path.basename(path).replace(".mp4", "")
    rows.append(np.hstack([tile(kfs[0], f"#{rank} {name} {score:.2f}"),
                         tile(kfs[min(1, len(kfs) - 1)], "shot 2")]))
result_img = np.vstack(rows)
cv2.imwrite("retrieval_output.jpg", result_img)
cv2.imshow("Content Based Video Retrieval", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()