
# import cv2
# from collections import defaultdict
# import supervision as sv
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('yolov8s.pt')

# # Set up video capture
# cap = cv2.VideoCapture(r"C:\Users\hp\Desktop\New folder\Traffic-management\a.mp4")

# # Define the line coordinates
# START = sv.Point(182, 254)
# END = sv.Point(462, 254)

# # Store the track history
# track_history = defaultdict(lambda: [])

# # Create a dictionary to keep track of objects that have crossed the line
# crossed_objects = {}

# # Open a video sink for the output video
# video_info = sv.VideoInfo.from_video_path(r"C:\Users\hp\Desktop\New folder\Traffic-management\a.mp4")
# with sv.VideoSink("new.mp4", video_info) as sink:
#     while cap.isOpened():
#         success, frame = cap.read()

#         if success:
#             # Run YOLOv8 tracking on the frame, persisting tracks between frames
#             results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")

#             # Get the boxes and track IDs
#             boxes = results[0].boxes.xywh.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()

#             # Visualize the results on the frame
#             annotated_frame = results[0].plot()
#             detections = sv.Detections.from_yolov8(results[0])

#             # Plot the tracks and count objects crossing the line
#             for box, track_id in zip(boxes, track_ids):
#                 x, y, w, h = box
#                 track = track_history[track_id]
#                 track.append((float(x), float(y)))  # x, y center point
#                 if len(track) > 30:  # retain 30 tracks for 30 frames
#                     track.pop(0)
#                 # Check if the object crosses the line
#                 if START.x < x < END.x and abs(y - START.y) < 5:  # Assuming objects cross horizontally
#                     if track_id not in crossed_objects:
#                         crossed_objects[track_id] = True

#                     # Annotate the object as it crosses the line
#                     cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
#                                   (0, 255, 0), 2)

#             # Draw the line on the frame
#             cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)

#             # Write the count of objects on each frame
#             count_text = f"Objects crossed: {len(crossed_objects)}"
#             cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Write the frame with annotations to the output video
#             sink.write_frame(annotated_frame)

#         else:
#             break

# # Release the video capture
# cap.release()

import math
from ultralytics import YOLO
import cv2
import cvzone
# from sort import *
# cap=cv2.VideoCapture(0) =>Web cam
# cap.set(3,1280) => these two are size of the camera
# cap.set(4,720)
cap=cv2.VideoCapture('Videos/video_traffic.mp4')
#D:\MachineLearning\Project1\SIH\main.py
model = YOLO('yolov8s.pt')
className = ['bicycle', 'bus', 'car','motorcycle', 'truck']
print(len(className))
mask=cv2.imread('template.png')
# tracker=Sort(max_age=20,min_hits=2,iou_threshold=0.3)
while True:
    success,img=cap.read()
    imgRegion=cv2.bitwise_and(img,mask)
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
           x1,y1,x2,y2=box.xyxy[0]
           x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
           w,h=x2-x1,y2-y1
           cvzone.cornerRect(img,(x1,y1,w,h))
           conf=box.conf[0]
           conf=math.ceil(conf*100)/100
           cls=int(box.cls[0])
           #currclass=className[cls]

           cvzone.putTextRect(img,f' {conf}',(max(0,x1),max(0,y1)),
                              thickness=1,scale=0.6,offset=3)

    cv2.imshow("Image",img)
   # cv2.imshow("image",imgRegion)
    cv2.waitKey(1)

