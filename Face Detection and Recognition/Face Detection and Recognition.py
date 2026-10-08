import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# ---------- 1. Enroll: capture 40 samples of your face ----------
name = input("Enter your name: ")
faces, labels = [], []
print("Look at the camera. Capturing samples...")
while len(faces) < 40:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    for (x, y, w, h) in rects[:1]:
        faces.append(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
        labels.append(0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, f"Capturing {len(faces)}/40", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Enroll", frame)
    cv2.waitKey(100)
cv2.destroyWindow("Enroll")

# ---------- 2. Train recognizer ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
names = {0: name}
print("Training done. Press 's' to save a screenshot, 'q' to quit.")

# ---------- 3. Live detection + recognition ----------
THRESHOLD = 80
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    for (x, y, w, h) in rects:
        roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, dist = recognizer.predict(roi)
        text = names[label] if dist < THRESHOLD else "Unknown"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{text} ({dist:.0f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Face Detection and Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("face_output.jpg", frame)
        print("Saved face_output.jpg")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()