from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("./runs-20240920T214004Z-001/runs/detect/train2/weights/best.pt")

# Open video capture
cap = cv2.VideoCapture("/home/bahaeddine09/Videos/Screencasts/Screencast from 2024-09-22 23-30-50.webm")

# Initialize variables for tracking
track_history = {}
next_id = 1

def get_iou(bb1, bb2):
    # Calculate the Intersection over Union of two bounding boxes
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    
    iou = intersection / float(area1 + area2 - intersection)
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)

    current_detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf.numpy()[0])
            xyxy = box.xyxy.numpy()[0]
            current_detections.append(xyxy)

    # Simple tracking algorithm
    new_track_history = {}
    for detection in current_detections:
        matched = False
        for track_id, track in track_history.items():
            if get_iou(detection, track[-1]) > 0.5:  # If IOU > 0.5, consider it the same object
                new_track_history[track_id] = track + [detection]
                matched = True
                break
        if not matched:
            new_track_history[next_id] = [detection]
            next_id += 1
    track_history = new_track_history

    # Draw bounding boxes and IDs
    for track_id, track in track_history.items():
        if len(track) < 3:  # Only draw if we've seen this object for at least 3 frames
            continue
        box = track[-1]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()