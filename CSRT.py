import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from numpy.fft import fft2, ifft2
import time
import os
from datetime import datetime

def extract_features(image, size=(64, 64)):
    image_resized = resize(image, size, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    gray = rgb2gray(image_resized)
    _, hog_image = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       visualize=True, channel_axis=None)
    hog_image_resized = resize(hog_image, size, anti_aliasing=True)
    return [gray, hog_image_resized]

def create_gaussian_response(size, sigma=2.0):
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))
    return gauss

def compute_spatial_reliability_map(patch, target_mask):
    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    except:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    mask_flat = target_mask.flatten()
    fg_pixels = pixels[mask_flat > 0]
    bg_pixels = pixels[mask_flat == 0]
    if fg_pixels.shape[0] == 0 or bg_pixels.shape[0] == 0:
        return np.ones(patch.shape[:2])
    bins = [16, 16, 16]
    ranges = [[0, 180], [0, 256], [0, 256]]
    fg_hist, _ = np.histogramdd(fg_pixels, bins=bins, range=ranges)
    bg_hist, _ = np.histogramdd(bg_pixels, bins=bins, range=ranges)
    fg_hist += 1e-6
    bg_hist += 1e-6
    fg_hist /= fg_hist.sum()
    bg_hist /= bg_hist.sum()
    h_bin = np.clip((hsv[:, :, 0] / 180.0 * 15).astype(int), 0, 15)
    s_bin = np.clip((hsv[:, :, 1] / 256.0 * 15).astype(int), 0, 15)
    v_bin = np.clip((hsv[:, :, 2] / 256.0 * 15).astype(int), 0, 15)
    fg_prob = fg_hist[h_bin, s_bin, v_bin]
    bg_prob = bg_hist[h_bin, s_bin, v_bin]
    prob_map = fg_prob / (fg_prob + bg_prob)
    prob_map = cv2.normalize(prob_map, None, 0, 1, cv2.NORM_MINMAX)
    return prob_map

def learn_filter(x, y, w, lamb=1e-4):
    X = fft2(w * x)
    Y = fft2(w * y)
    H = np.conj(X) * Y / (np.conj(X) * X + lamb)
    return H

def apply_filter(x, H):
    X = fft2(x)
    response = np.real(ifft2(X * H))
    return response

class CSRTTracker:
    def __init__(self, resize_shape=(64, 64), sigma=2.0, learning_rate=0.025):
        self.resize_shape = resize_shape
        self.sigma = sigma
        self.lr = learning_rate
        self.filters = None
        self.position = None
        self.size = None
        self.mask = None

    def init(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.position = (y + h // 2, x + w // 2)
        self.size = (h, w)
        patch = frame[y:y + h, x:x + w]
        resized_patch = resize(patch, self.resize_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        features = extract_features(resized_patch, self.resize_shape)
        self.y = create_gaussian_response(self.resize_shape, sigma=self.sigma)
        h_, w_ = self.resize_shape
        cy, cx = h_ // 2, w_ // 2
        Y, X = np.ogrid[:h_, :w_]
        radius = min(h_, w_) * 0.3
        self.mask = (((Y - cy)**2 + (X - cx)**2) < radius**2).astype(np.uint8)
        self.spatial_map = compute_spatial_reliability_map(resized_patch, self.mask)
        self.filters = [learn_filter(f, self.y * self.spatial_map, self.spatial_map) for f in features]

    def update(self, frame):
        y, x = self.position
        h, w = self.size
        y1, y2 = max(0, y - h // 2), min(frame.shape[0], y + h // 2)
        x1, x2 = max(0, x - w // 2), min(frame.shape[1], x + w // 2)
        patch = frame[y1:y2, x1:x2]
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            print("Patch too small. Skipping frame.")
            return (x - w // 2, y - h // 2, w, h)
        try:
            patch_resized = resize(patch, self.resize_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        except Exception as e:
            print(f"Resize error during update: {e}")
            return (x - w // 2, y - h // 2, w, h)
        features = extract_features(patch_resized)
        response = sum(apply_filter(f, H) for f, H in zip(features, self.filters))
        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        dy -= self.resize_shape[0] // 2
        dx -= self.resize_shape[1] // 2
        self.position = (y + dy, x + dx)
        new_y, new_x = self.position
        y1, y2 = max(0, new_y - h // 2), min(frame.shape[0], new_y + h // 2)
        x1, x2 = max(0, new_x - w // 2), min(frame.shape[1], new_x + w // 2)
        new_patch = frame[y1:y2, x1:x2]
        if new_patch.shape[0] < 2 or new_patch.shape[1] < 2:
            print("New patch too small. Skipping update.")
            return (new_x - w // 2, new_y - h // 2, w, h)
        try:
            new_patch_resized = resize(new_patch, self.resize_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        except Exception as e:
            print(f"Resize error during model update: {e}")
            return (new_x - w // 2, new_y - h // 2, w, h)
        new_features = extract_features(new_patch_resized)
        new_spatial_map = compute_spatial_reliability_map(new_patch_resized, self.mask)
        for i, f in enumerate(new_features):
            H_new = learn_filter(f, self.y * new_spatial_map, new_spatial_map)
            self.filters[i] = (1 - self.lr) * self.filters[i] + self.lr * H_new
        return (self.position[1] - w // 2, self.position[0] - h // 2, w, h)

if __name__ == "__main__":
    video_path = r"C:\Users\AsusIran\Desktop\videos\car1.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        exit()

    init_box = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    tracker = CSRTTracker()
    tracker.init(frame, init_box)

    frame_height, frame_width = frame.shape[:2]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(script_dir, f"csrt_tracking_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        bbox = tracker.update(frame)
        end_time = time.time()
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        fps = 1.0 / (end_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.2f}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.write(frame)
        cv2.imshow("CSRT Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
