import cv2
import numpy as np
from numpy.fft import fft2, ifft2
import time
import os
from datetime import datetime

RESIZE_W, RESIZE_H = 192, 192
UPDATE_EVERY_N_FRAMES = 10

video_path = r"C:\Users\AsusIran\Desktop\videos\car1.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    cap.release()
    exit()

bbox = cv2.selectROI("Select Target", frame, fromCenter=False)
cv2.destroyAllWindows()

x, y, w, h = [int(v) for v in bbox]
target = frame[y:y+h, x:x+w]
gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
target_resized = cv2.resize(gray_target, (RESIZE_W, RESIZE_H))

def create_gaussian_response(shape, sigma=2.0):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))
    return gauss / gauss.max()

desired = create_gaussian_response((RESIZE_H, RESIZE_W))

def preprocess(img):
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    return img * win

F = fft2(preprocess(target_resized))
G = fft2(desired)

Ai = G * np.conj(F)
Bi = F * np.conj(F)
H = Ai / (Bi + 1e-5)

cap = cv2.VideoCapture(video_path)
cap.read()
pos = [x, y, w, h]
frame_count = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(script_dir, f"mosse_output_{timestamp}.avi")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_writer = cv2.VideoWriter(out_path, fourcc, out_fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    x, y, w, h = pos
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame_h, frame_w = gray_frame.shape

    x = max(0, min(x, frame_w - w))
    y = max(0, min(y, frame_h - h))
    pos = [x, y, w, h]

    search_window = gray_frame[y:y+h, x:x+w]

    try:
        search_resized = cv2.resize(search_window, (RESIZE_W, RESIZE_H))
        F_frame = fft2(preprocess(search_resized))
        response = np.real(ifft2(H * F_frame))

        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        shift_y = dy - RESIZE_H // 2
        shift_x = dx - RESIZE_W // 2

        x += shift_x
        y += shift_y
        pos = [x, y, w, h]

        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        pos = [x, y, w, h]

        if frame_count % UPDATE_EVERY_N_FRAMES == 0:
            new_patch = gray_frame[y:y+h, x:x+w]
            new_patch_resized = cv2.resize(new_patch, (RESIZE_W, RESIZE_H))
            F_new = fft2(preprocess(new_patch_resized))

            Ai = 0.125 * (G * np.conj(F_new)) + 0.875 * Ai
            Bi = 0.125 * (F_new * np.conj(F_new)) + 0.875 * Bi
            H = Ai / (Bi + 1e-5)

        frame_count += 1

    except Exception as e:
        print("Skipping frame due to error:", e)
        continue

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    end_time = time.time()
    fps = 1.0 / (end_time - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out_writer.write(frame)
    cv2.imshow("MOSSE Tracker (FPS shown)", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()
