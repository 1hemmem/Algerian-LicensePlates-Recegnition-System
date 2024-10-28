import cv2
# import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort, Tracker, Detection

# Load YOLO models for vehicle and license plate detection
model = YOLO("yolov10n.pt")  # Car detection model
license_plate_detector = YOLO(
    "./runs-20240920T214004Z-001/runs/detect/train2/weights/best.pt"
)  # License plate detector

# Open video capture
cap = cv2.VideoCapture("VID_20240923_141113.mp4")

# Initialize DeepSORT tracker
# Specify an embedder model path or use the default one
deepsort = DeepSort()
tracker = Tracker(
    deepsort.embedder,
    max_age=30,
    n_init=3,# iou_threshold=0.7  # Adjust the IoU threshold as needed
)

min_area_threshold = 500000


def save_bounding_box_image(frame, bbox, track_id, i, isPlate=False):
    if not isPlate:
        object_type = "car"
    else:
        object_type = "plate"

    if bbox is not None:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Crop the image (Region of Interest - ROI)
        roi = frame[y1:y2, x1:x2]

        # Define the filename for saving
        directory = f"./output/{track_id}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"bbox_{object_type}_{track_id}_{i}.png")

        # Save the cropped image (bounding box)
        cv2.imwrite(filename, roi)
        print(f"Saved bounding box as {filename}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run vehicle detection model
    results = model(frame, conf=0.7)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if cls_id == 2:  # Check if the detected object is a car (YOLO class ID for cars is typically 2)
                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                if area and area < min_area_threshold:
                    continue
                detection = Detection(xyxy, conf, cls_id)
                detections.append(detection)

    # Update DeepSORT tracker
    tracker.predict()
    tracker.update(detections)

    # Process tracked objects
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        track_id = track.track_id

        # Save the bounding box for the car
        save_bounding_box_image(
            frame, bbox, track_id, len(track.xyzy_history), isPlate=False
        )

        # Crop the detected car for license plate detection
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        car_roi = frame[y1:y2, x1:x2]

        # Run license plate detection model on the cropped car region
        lp_results = license_plate_detector(car_roi, conf=0.7)
        for lp_result in lp_results:
            lp_boxes = lp_result.boxes
            for lp_box in lp_boxes:
                lp_xyxy = lp_box.xyxy.cpu().numpy()[0]
                lp_conf = lp_box.conf.cpu().numpy()[0]

                lp_x1, lp_y1, lp_x2, lp_y2 = [int(lp_coord) for lp_coord in lp_xyxy]

                absolute_lp_x1 = x1 + lp_x1
                absolute_lp_y1 = y1 + lp_y1
                absolute_lp_x2 = x1 + lp_x2
                absolute_lp_y2 = y1 + lp_y2

                cv2.rectangle(
                    frame,
                    (absolute_lp_x1, absolute_lp_y1),
                    (absolute_lp_x2, absolute_lp_y2),
                    (255, 0, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Confidence: {lp_conf:.2f}",
                    (absolute_lp_x1, absolute_lp_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                if lp_conf >= 0.9:
                    save_bounding_box_image(
                        frame,
                        [
                            absolute_lp_x1,
                            absolute_lp_y1,
                            absolute_lp_x2,
                            absolute_lp_y2,
                        ],
                        track_id,
                        len(track.xyzy_history),
                        True,
                    )

    # Display the frame
    cv2.imshow("YOLO Detection and License Plate Extraction", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
