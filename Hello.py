import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

VIDEO_PATH = r"C:\Users\hp032\Downloads\Traffic_Management_System_final-main\Traffic_Management_System_final-main\Videos\video_traffic.mp4"
MODEL_PATH = "yolov8s.pt"
TARGET_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video: {VIDEO_PATH}")
    exit()

# Step 1: Let user draw ROI
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] Could not read first frame.")
    exit()

roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# Convert ROI to polygon points
x, y, w, h = roi
ROI_POINTS = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

# Tracker and counting variables
tracker = sv.ByteTrack()
counted_ids = set()
total_count = 0
mask = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
cv2.fillPoly(mask, [np.array(ROI_POINTS, dtype=np.int32)], 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, TARGET_CLASSES)]
    detections = tracker.update_with_detections(detections)

    cv2.polylines(frame, [np.array(ROI_POINTS, dtype=np.int32)], True, (0, 255, 255), 2)

    for xyxy, track_id, class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if mask[cy, cx] == 255 and track_id not in counted_ids:
            counted_ids.add(track_id)
            total_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Count: {total_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"[INFO] Final count: {total_count}")
cap.release()
cv2.destroyAllWindows()
