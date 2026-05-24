from ultralytics import YOLO
import numpy as np
import cv2
import time
import imutils
import requests
from collections import deque

model = YOLO("yolov8n.pt")

buffer = []

fx, fy = 600, 600
cx, cy = 300, 300
REAL_WIDTH = 0.1

URL = "http://127.0.0.1:8000/predict"

vs = cv2.VideoCapture(0)

def estimate_depth(w):
    if w == 0:
        return 1e-4
    return (fx * REAL_WIDTH) / w


start_time = time.time()
RUN_TIME = 9  

while True:
    ret, frame = vs.read()
    if not ret:
        continue

    frame = imutils.resize(frame, width=600)

    results = model(frame, verbose=False)

    for r in results:
        if len(r.boxes) == 0:
            continue

        box = r.boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        cx_pixel = int((x1 + x2) / 2)
        cy_pixel = int((y1 + y2) / 2)

        w = x2 - x1

        Z = estimate_depth(w)
        X = (cx_pixel - cx) * Z / fx
        Y = (cy_pixel - cy) * Z / fy

        buffer.append([X, Y, Z])

        cv2.circle(frame, (cx_pixel, cy_pixel), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        break

    cv2.imshow("YOLO", frame)
    if time.time() - start_time > RUN_TIME and len(buffer) > 20:

        arr = np.array(buffer)
        center = np.median(arr, axis=0)

        print("FINAL STABLE GOAL:", center)

        try:
            requests.post(
                URL,
                json={
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "roll": 0,
                    "pitch": 0,
                    "yaw": 0
                },
                timeout=2
            )
        except Exception as e:
            print("API error:", e)

        break  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
print("Demo finished.")
