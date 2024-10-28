import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO models for vehicle and license plate detection
model = YOLO("yolov10n.pt")  # Car detection model
license_plate_detector = YOLO(
    "./runs-20240920T214004Z-001/runs/detect/train2/weights/best.pt"
)  # License plate detector

# Open video capture
cap = cv2.VideoCapture("VID_20240923_141113.mp4")

# Initialize DeepSORT tracker
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100, embedder_gpu=True)

min_area_threshold = 5000

# Function to save the cropped image from the bounding box
def save_bounding_box_image(frame, bbox, track_id, i, isPlate=False):
    object_name = "plate" if isPlate else "car"
    if bbox is not None:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Crop the image (Region of Interest - ROI)
        roi = frame[y1:y2, x1:x2]

        # Define the filename for saving
        directory = f"./output/{track_id}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"bbox_{object_name}_{track_id}_{i}.png")

        # Save the cropped image (bounding box)
        cv2.imwrite(filename, roi)
        print(f"Saved bounding box as {filename}")
        return filename  # Return the saved file path
    return None

# Function to extract license plate number using pytesseract
def extract_license_plate_text(image_path):
    configuration = r"-c tessedit_char_whitelist=' 0123456789' --psm 10"
    try:
        data = pytesseract.image_to_data(
            Image.open(image_path), config=configuration, output_type=pytesseract.Output.DICT
        )
        text = data["text"]
        confidences = data["conf"]

        for i in range(len(text)):
            if int(confidences[i]) > 0:  # Filter low-confidence text
                print(f"Detected Text: {text[i]}, Confidence: {confidences[i]}")
    except Exception as e:
        print(f"Error during OCR: {e}")

# Main loop for video processing
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run vehicle detection model
    results = model(frame, conf=0.5)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if cls_id == 2:  # Check if the detected object is a car (class id 2 for cars)
                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                if area and area < min_area_threshold:
                    continue

                # Ensure the correct format for DeepSORT: [x1, y1, x2, y2, conf]
                detections.append([x1, y1, x2, y2, conf])

    # Update DeepSORT tracker with detections
    if detections:  # Only update if detections are available
        tracks = deepsort.update_tracks(detections, frame=frame)

        # Process tracked objects
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_ltrb()  # Get the bounding box in the format (left, top, right, bottom)
            track_id = track.track_id

            # Save the bounding box for the car
            saved_image_path = save_bounding_box_image(
                frame, bbox, track_id, frame_id, isPlate=False
            )

            # Crop the detected car for license plate detection
            if saved_image_path:
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
                            lp_image_path = save_bounding_box_image(
                                frame,
                                [absolute_lp_x1, absolute_lp_y1, absolute_lp_x2, absolute_lp_y2],
                                track_id,
                                frame_id,
                                True
                            )
                            # Perform OCR on the detected license plate
                            if lp_image_path:
                                extract_license_plate_text(lp_image_path)

    cv2.imshow("YOLO Detection and License Plate Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
