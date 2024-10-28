import cv2
import os


def save_bounding_box_image(frame, bbox, track_id, i, isPlate=False):

    if isPlate == False:
        object = "car"
    else:
        object = "plate"
    if bbox is not None:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        roi = frame[y1:y2, x1:x2]

        directory = f"../../output/{track_id}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"bbox_{object}_{track_id}_{i}.png")

        # Save the cropped image (bounding box)
        cv2.imwrite(filename, roi)
        print(f"Saved bounding box as {filename}")


def get_iou(bb1, bb2):

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
