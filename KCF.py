import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy.fft import fft2, ifft2
import time
import os
from datetime import datetime

video_path = r"C:\Users\AsusIran\Desktop\videos\person4.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Error: Cannot load video.")
    cap.release()
    exit()

bbox = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = map(int, bbox)
target_patch = frame[y:y+h, x:x+w]
cap.release()

def preprocess(image):
    if image is None or image.size == 0 or np.any(np.isnan(image)) or np.any(np.isinf(image)):
        raise ValueError("Invalid image for preprocessing")
    image = np.log(image.astype(np.float32) + 1)
    image -= image.mean()
    image /= (image.std() + 1e-5)
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    return image * window

def create_gaussian_response(size, sigma=2):
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = (x - cx)**2 + (y - cy)**2
    g = np.exp(-0.5 * dist / (sigma ** 2))
    return g

def gaussian_kernel(xf, yf, sigma):
    N = xf.shape[0] * xf.shape[1]
    xx = np.sum(np.abs(xf)**2) / N
    yy = np.sum(np.abs(yf)**2) / N
    xyf = xf * np.conj(yf)
    ixyf = np.real(ifft2(xyf))
    dist = np.maximum(0, xx + yy - 2 * ixyf)
    kf = fft2(np.exp(-1 / (sigma**2) * dist / N))
    return kf

target_size = (230, 230)
gray_target = rgb2gray(target_patch)
gray_target = resize(gray_target, target_size, anti_aliasing=True)
x_feat = preprocess(gray_target)
y_resp = create_gaussian_response(target_size, sigma=2)

xf = fft2(x_feat)
yf = fft2(y_resp)
kf = gaussian_kernel(xf, xf, sigma=0.5)

alpha = yf / (kf + 1e-4)
model_xf = xf.copy()
model_alpha = alpha.copy()

cap = cv2.VideoCapture(video_path)
cap.read()
position = (x + w // 2, y + h // 2)
size = (w, h)
learning_rate = 0.075

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(script_dir, f"kcf_output_{timestamp}.mp4")
out = cv2.VideoWriter(output_path, fourcc, fps_video, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    frame_h, frame_w = frame.shape[:2]

    cx, cy = position
    w, h = size

    cx = max(w//2, min(frame_w - w//2 - 1, cx))
    cy = max(h//2, min(frame_h - h//2 - 1, cy))
    position = (cx, cy)

    x1 = int(cx - w // 2)
    y1 = int(cy - h // 2)
    x2 = int(cx + w // 2)
    y2 = int(cy + h // 2)

    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h or x2 - x1 <= 1 or y2 - y1 <= 1:
        print("Skipping frame: invalid window")
        continue

    patch = frame[y1:y2, x1:x2]

    try:
        gray_patch = rgb2gray(patch)
        gray_patch = resize(gray_patch, target_size, anti_aliasing=True)
        if gray_patch.size == 0 or np.any(np.isnan(gray_patch)) or np.any(np.isinf(gray_patch)):
            raise ValueError("Patch invalid after resizing")

        z = preprocess(gray_patch)
        zf = fft2(z)
        kzf = gaussian_kernel(zf, model_xf, sigma=0.5)
        response = np.real(ifft2(model_alpha * kzf))

        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        shift_y = dy - response.shape[0] // 2
        shift_x = dx - response.shape[1] // 2

        cx += shift_x
        cy += shift_y
        position = (cx, cy)

        new_xf = fft2(z)
        new_kf = gaussian_kernel(new_xf, new_xf, sigma=0.5)
        new_alpha = yf / (new_kf + 1e-4)

        model_alpha = (1 - learning_rate) * model_alpha + learning_rate * new_alpha
        model_xf = (1 - learning_rate) * model_xf + learning_rate * new_xf

        x_draw = int(cx - w / 2)
        y_draw = int(cy - h / 2)
        cv2.rectangle(frame, (x_draw, y_draw), (x_draw + w, y_draw + h), (0, 255, 0), 2)

        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("KCF Tracker (230x230)", frame)
        out.write(frame)

    except Exception as e:
        print("Skipping frame due to error:", e)
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
