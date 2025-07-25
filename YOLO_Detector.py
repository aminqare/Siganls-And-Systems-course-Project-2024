import cv2
import torch
import random
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
model.eval()

model.conf = 0.3
model.iou = 0.4
model.classes = None

def get_class_colors(class_names):
    random.seed(42)
    return {
        name: tuple(random.choices(range(50, 256), k=3))
        for name in class_names
    }

video_path = r"C:\Users\AsusIran\Desktop\test-image3.jpg"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame.")
    exit()

results = model(frame)
detections = results.xyxy[0]

if len(detections) == 0:
    print("No objects detected.")
    exit()

class_names = list(model.names.values())
class_colors = get_class_colors(class_names)

for det in detections:
    x1, y1, x2, y2, conf, cls_id = det.tolist()
    class_name = model.names[int(cls_id)]
    label = f"{class_name} {conf:.2f}"
    color = class_colors[class_name]

    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))

    cv2.rectangle(frame, pt1, pt2, color, 1)
    cv2.putText(frame, label, (pt1[0], pt1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("YOLOv5l - Color Coded Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(script_dir, f"detection_output_{timestamp}.jpg")
cv2.imwrite(output_path, frame)

