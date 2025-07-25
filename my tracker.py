import cv2
import time
import numpy as np
from ultralytics import YOLO
import random
from filterpy.kalman import KalmanFilter
from numpy.fft import fft2, ifft2
import os
from datetime import datetime

def get_class_colors(class_names, excluded_ids):
    random.seed(42)
    return {
        name: tuple(random.choices(range(50, 256), k=3))
        for i, name in enumerate(class_names)
        if i not in excluded_ids
    }

def create_kalman_filter(x, y):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x, y, 0, 0])
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 500.
    kf.R *= 5.
    kf.Q = np.array([[1, 0, 0.5, 0],
                     [0, 1, 0, 0.5],
                     [0.5, 0, 1, 0],
                     [0, 0.5, 0, 1]]) * 2
    return kf

def apply_frequency_filter(patch):
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    patch_norm = (patch_gray - patch_gray.mean()) / (patch_gray.std() + 1e-5)
    win = np.outer(np.hanning(patch_norm.shape[0]), np.hanning(patch_norm.shape[1]))
    fft_img = fft2(patch_norm * win)
    freqs_y = np.fft.fftfreq(fft_img.shape[0])[:, None]
    freqs_x = np.fft.fftfreq(fft_img.shape[1])
    gauss = np.exp(-0.5 * (freqs_y**2 + freqs_x**2) / 0.005)
    return ifft2(fft_img * gauss).real

class CustomTracker:
    def __init__(self, init_bbox, class_id, class_name):
        x1, y1, x2, y2 = init_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.center = np.array([cx, cy])
        self.size = np.array([x2 - x1, y2 - y1])
        self.class_id = class_id
        self.class_name = class_name
        self.kalman = create_kalman_filter(cx, cy)
        self.lost_counter = 0
        self.max_lost = 15
        self.alpha = 0.6

    def update(self, new_bbox):
        x1, y1, x2, y2 = new_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_size = np.array([x2 - x1, y2 - y1])
        self.kalman.update(np.array([cx, cy]))
        self.center = self.kalman.x[:2]
        self.size = self.alpha * new_size + (1 - self.alpha) * self.size
        self.lost_counter = 0

    def predict(self):
        self.kalman.predict()
        self.center = self.kalman.x[:2]
        self.lost_counter += 1

    def is_lost(self):
        return self.lost_counter >= self.max_lost

    def get_bbox(self):
        cx, cy = self.center.astype(int)
        w, h = self.size.astype(int)
        return (max(0, cx - w // 2), max(0, cy - h // 2), cx + w // 2, cy + h // 2)

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_w, inter_h = max(0, x2 - x1), max(0, y2 - y1)
    if inter_w == 0 or inter_h == 0:
        return 0.0
    inter = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

class TrackerStats:
    def __init__(self):
        self.detection_counts = {}

    def update(self, class_name):
        if class_name not in self.detection_counts:
            self.detection_counts[class_name] = 0
        self.detection_counts[class_name] += 1

    def log(self):
        print("Detection stats:", self.detection_counts)

EXCLUDED_CLASS_IDS = set()
model = YOLO("yolov8m.pt")
model.conf = 0.3
class_names = list(model.names.values())
class_colors = get_class_colors(class_names, EXCLUDED_CLASS_IDS)

cap = cv2.VideoCapture(r"C:\Users\AsusIran\Desktop\videos\car1.mp4")
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(script_dir, f"yolo_kalman_output_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

detections = model(frame)[0].boxes
trackers = []
stats = TrackerStats()
for box in detections:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cls_id = int(box.cls[0].item())
    if cls_id in EXCLUDED_CLASS_IDS:
        continue
    trackers.append(CustomTracker((x1, y1, x2, y2), cls_id, model.names[cls_id]))
    stats.update(model.names[cls_id])

frame_id = 1
REDETECT_INTERVAL = 15
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value_mean = np.mean(hsv[:, :, 2])
    if value_mean < 60 or value_mean > 200:
        cv2.putText(frame, "Lighting Variation Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    detections = []
    if frame_id % REDETECT_INTERVAL == 0:
        boxes = model(frame)[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in EXCLUDED_CLASS_IDS:
                continue
            detections.append((box.xyxy[0].tolist(), cls_id))

    matched = set()
    for tracker in trackers:
        tracker.predict()
        best_iou, best_det, best_idx = 0, None, -1
        for i, (bbox, cls_id) in enumerate(detections):
            if cls_id != tracker.class_id or i in matched:
                continue
            iou = compute_iou(bbox, tracker.get_bbox())
            if iou > best_iou:
                best_iou, best_det, best_idx = iou, bbox, i
        if best_iou > 0.3:
            tracker.update(best_det)
            matched.add(best_idx)
            stats.update(tracker.class_name)

    for i, (bbox, cls_id) in enumerate(detections):
        if i not in matched:
            trackers.append(CustomTracker(bbox, cls_id, model.names[cls_id]))
            stats.update(model.names[cls_id])

    trackers = [t for t in trackers if not t.is_lost()]

    for t in trackers:
        x1, y1, x2, y2 = t.get_bbox()
        color = class_colors.get(t.class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, t.class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if frame_id % 30 == 0:
        stats.log()

    out.write(frame)
    cv2.imshow("YOLOv8 + Enhanced Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
