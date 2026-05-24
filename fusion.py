import threading
import numpy as np
import time
import requests
import serial
from ultralytics import YOLO
import cv2
import imutils
from collections import deque

yolo_buffer = deque(maxlen=50)
latest_depth = None
lock = threading.Lock()
running = True

def arduino_reader():
    global latest_depth, running

    ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
    time.sleep(2)

    while running:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                val = float(line)

                with lock:
                    latest_depth = val

        except:
            pass

def yolo_reader():
    global running

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = imutils.resize(frame, width=600)
        results = model(frame, verbose=False)

        for r in results:
            if len(r.boxes) == 0:
                continue

            box = r.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            yolo_buffer.append([cx, cy])

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)

            break

        cv2.imshow("YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

def fusion_loop():
    global running, latest_depth

    time.sleep(3)

    fx, fy = 600, 600
    cx, cy = 300, 300

    while running:

        if len(yolo_buffer) < 10:
            continue

        with lock:
            z_raw = latest_depth

        if z_raw is None:
            continue

        z = z_raw / 100.0

        yolo_arr = np.array(yolo_buffer)
        u, v = np.mean(yolo_arr, axis=0)

        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy

        goal = np.array([X, Y, z])

        print("FINAL GOAL:", goal)

        try:
            requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "x": float(X),
                    "y": float(Y),
                    "z": float(z),
                    "roll": 0,
                    "pitch": 0,
                    "yaw": 0
                },
                timeout=1.0
            )
        except:
            pass

        time.sleep(1)

def start_system(duration=30):
    global running

    running = True

    t1 = threading.Thread(target=arduino_reader)
    t2 = threading.Thread(target=yolo_reader)
    t3 = threading.Thread(target=fusion_loop)

    t1.start()
    t2.start()
    t3.start()

    print("System running...")

    time.sleep(duration)

    running = False

    time.sleep(2)

    t2.join()
    t3.join()
    t1.join()

    print("System stopped cleanly.")


if __name__ == "__main__":
    input("Press ENTER to start demo...")
    start_system(30)
