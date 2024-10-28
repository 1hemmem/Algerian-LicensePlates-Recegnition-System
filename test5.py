from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import os
import io
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
min_area_threshold = 2000


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


def predict_next_bbox(track):
    if len(track) < 2:
        return track[-1]

    # Simple linear motion model
    last_box = track[-1]
    prev_box = track[-2]
    dx = last_box[0] - prev_box[0]
    dy = last_box[1] - prev_box[1]
    dw = last_box[2] - prev_box[2]
    dh = last_box[3] - prev_box[3]

    predicted_box = [
        last_box[0] + dx,
        last_box[1] + dy,
        last_box[2] + dw,
        last_box[3] + dh,
    ]
    return predicted_box


def save_bounding_box_image(frame, bbox, track_id, i, isPlate=False):

    if isPlate == False:
        object = "car"
    elif isPlate == True:
        object = "plate"
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Crop the image (Region of Interest - ROI)
    roi = frame[y1:y2, x1:x2]

    # Define the filename for saving
    directory = f"./output/{track_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, f"bbox_{object}_{track_id}_{i}.png")

    # Save the cropped image (bounding box)
    cv2.imwrite(filename, roi)
    print(f"Saved bounding box as {filename}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run vehicle detection model
    results = model(frame, conf=0.8)

    current_detections = []
    for result in results:
        boxes = result.boxes
        for idx, box in enumerate(boxes):
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]

            # Draw the bounding box
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            width = x2 - x1
            height = y2 - y1
            area = width * height
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            current_detections.append(xyxy)
            save_bounding_box_image(frame, [x1, y1, x2, y2], idx, len(track_history))
            # Skip small objects below area threshold
            if area and area < min_area_threshold:
                print("Ignored small area")
                continue

            # Crop the detected car for license plate detection
            car_roi = frame[y1:y2, x1:x2]

            # Run license plate detection model on the cropped car region
            lp_results = license_plate_detector(car_roi, conf=0.6)
            for lp_result in lp_results:
                lp_boxes = lp_result.boxes
                for lp_idx, lp_box in enumerate(lp_boxes):
                    lp_xyxy = lp_box.xyxy.cpu().numpy()[0]
                    lp_conf = lp_box.conf.cpu().numpy()[0]
                    lp_x1, lp_y1, lp_x2, lp_y2 = [int(lp_coord) for lp_coord in lp_xyxy]

                    # Adjust coordinates relative to the original frame
                    absolute_lp_x1 = x1 + lp_x1
                    absolute_lp_y1 = y1 + lp_y1
                    absolute_lp_x2 = x1 + lp_x2
                    absolute_lp_y2 = y1 + lp_y2

                    # Draw the license plate bounding box
                    cv2.rectangle(
                        frame,
                        (absolute_lp_x1, absolute_lp_y1),
                        (absolute_lp_x2, absolute_lp_y2),
                        (255, 0, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Confidance: {lp_conf}",
                        (absolute_lp_x1, absolute_lp_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )
                    if lp_conf >= 0.9:

                        # Save license plate image
                        save_bounding_box_image(
                            frame,
                            [
                                absolute_lp_x1,
                                absolute_lp_y1,
                                absolute_lp_x2,
                                absolute_lp_y2,
                            ],
                            idx,
                            len(track_history),
                            True,
                        )

    # Improved tracking algorithm
    new_track_history = {}
    unmatched_detections = current_detections.copy()

    for track_id, track in track_history.items():
        predicted_box = predict_next_bbox(track)
        best_iou = 0
        best_detection = None
        for detection in unmatched_detections:
            iou = get_iou(predicted_box, detection)
            if iou > best_iou and iou > 0.7:  # IOU threshold for tracking
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
        elif len(track) < 10:  # Keep short tracks alive for a few frames
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
        if (
            len(track) < 3
        ):  # Only draw if the object has been tracked for at least 3 frames
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

    # Show the frame
    cv2.imshow("YOLO Detection and License Plate Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
