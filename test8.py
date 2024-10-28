import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import utils

# Load YOLO models for vehicle and license plate detection
model = YOLO("yolov10n.pt")  # Car detection model
license_plate_detector = YOLO(
    "./runs-20240920T214004Z-001/runs/detect/train2/weights/best.pt"
)  # License plate detector

# Open video capture
cap = cv2.VideoCapture("VID_20240923_141113.mp4")

# Initialize variables for tracking
track_history = {}
next_id = 1
max_track_length = 30  # Maximum number of frames to keep in track history
min_area_threshold = 10000
iou_threshold = 0.7  # IOU threshold for tracking
lp_conf = 0
stable_track_threshold = 5  # Number of frames a car must be tracked before detecting license plates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run vehicle detection model
    results = model(frame, conf=0.7)

    current_detections = []
    for result in results:
        boxes = result.boxes
        for idx, box in enumerate(boxes):
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            # Draw the bounding box
            if cls_id == 2:  # Check for cars (YOLO class ID for cars is typically 2)
                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                if area and (area < min_area_threshold):
                    continue
                current_detections.append(xyxy)

    # Improved tracking algorithm
    new_track_history = {}
    unmatched_detections = current_detections.copy()

    for track_id, track in track_history.items():
        predicted_box = utils.predict_next_bbox(track)
        best_iou = 0
        best_detection = None
        for detection in unmatched_detections:
            iou = utils.get_iou(predicted_box, detection)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_detection = detection

        if best_detection is not None:
            new_track = track.copy()
            new_track.append(best_detection)
            if len(new_track) > max_track_length:
                new_track = deque(
                    list(new_track)[-max_track_length:], maxlen=max_track_length
                )
            new_track_history[track_id] = new_track
            unmatched_detections = [
                det
                for det in unmatched_detections
                if not np.array_equal(det, best_detection)
            ]

            # Run license plate detection only when the car has been tracked for a certain number of frames
            if len(new_track) >= stable_track_threshold:
                x1, y1, x2, y2 = [int(coord) for coord in best_detection]
                car_roi = frame[y1:y2, x1:x2]

                # Run license plate detection model on the cropped car region
                lp_results = license_plate_detector(car_roi, conf=0.7)
                for lp_result in lp_results:
                    lp_boxes = lp_result.boxes
                    for lp_idx, lp_box in enumerate(lp_boxes):
                        lp_xyxy = lp_box.xyxy.cpu().numpy()[0]
                        lp_conf = lp_box.conf.cpu().numpy()[0]

                        lp_x1, lp_y1, lp_x2, lp_y2 = [
                            int(lp_coord) for lp_coord in lp_xyxy
                        ]

                        absolute_lp_x1 = x1 + lp_x1
                        absolute_lp_y1 = y1 + lp_y1
                        absolute_lp_x2 = x1 + lp_x2
                        absolute_lp_y2 = y1 + lp_y2

                        # Draw license plate bounding box
                        cv2.rectangle(
                            frame,
                            (absolute_lp_x1, absolute_lp_y1),
                            (absolute_lp_x2, absolute_lp_y2),
                            (255, 0, 0),
                            2,
                        )

                        cv2.putText(
                            frame,
                            f"Confidance: {lp_conf:.2f}",
                            (absolute_lp_x1, absolute_lp_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2,
                        )

                        # Save license plate bounding box image
                        if lp_conf >= 0.6:
                            utils.save_bounding_box_image(
                                frame,
                                [
                                    absolute_lp_x1,
                                    absolute_lp_y1,
                                    absolute_lp_x2,
                                    absolute_lp_y2,
                                ],
                                track_id,
                                len(new_track),
                                True,
                            )
        elif len(track) < 20:
            new_track = track.copy()
            new_track.append(predicted_box)
            new_track_history[track_id] = new_track

    # Add new tracks for unmatched detections
    for detection in unmatched_detections:
        new_track_history[next_id] = deque([detection], maxlen=max_track_length)
        next_id += 1

    track_history = new_track_history

    # Draw bounding boxes and IDs for cars
    for track_id, track in track_history.items():
        if len(track) < 3:
            continue
        box = track[-1]
        cv2.rectangle(
            frame,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (int(box[0]), int(box[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    cv2.imshow("YOLO Detection and License Plate Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

