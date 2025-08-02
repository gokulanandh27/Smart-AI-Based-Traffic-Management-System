import math
import os
import cv2
import cvzone
from ultralytics import YOLO

# Ensure video file exists
video_path = 'Videos/video_traffic.mp4'
if not os.path.exists(video_path):
    print("Error: Video file not found!")
    exit()

cap = cv2.VideoCapture(video_path)

# Ensure template image exists
mask_path = "template.png"
if not os.path.exists(mask_path):
    print("Error: template.png not found!")
    exit()

mask = cv2.imread(mask_path)
if mask is None:
    print("Error: Could not load template.png")
    exit()

model = YOLO('yolov8s.pt')
className = ['bicycle', 'bus', 'car', 'motorcycle', 'truck']
print(f"Loaded {len(className)} classes.")

while True:
    success, img = cap.read()
    if not success:
        print("End of video or error in reading frame.")
        break  # Exit the loop if video ends

    imgRegion = cv2.bitwise_and(img, mask) if mask is not None else img
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = box.conf[0] if hasattr(box, 'conf') else 0
            conf = math.ceil(conf * 100) / 100

            cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
            if 0 <= cls < len(className):
                label = className[cls]
            else:
                label = "Unknown"

            cvzone.putTextRect(img, f'{label}: {conf}', (max(0, x1), max(0, y1)), thickness=1, scale=0.6, offset=3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

cap.release()
cv2.destroyAllWindows()
