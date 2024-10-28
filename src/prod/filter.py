import cv2
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Variance of the Laplacian
    return laplacian_var


def count_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.count_nonzero(edges)
    return edge_count


def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val = gray.min(), gray.max()
    contrast = max_val - min_val
    return contrast


def detect_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = np.var(gray)
    return noise


def quality_score(image):
    laplacian_variance = calculate_sharpness(image)
    edge_count = count_edges(image)
    contrast = compute_contrast(image)
    noise = detect_noise(image)

    # You can adjust the weightings of each metric based on importance
    return 0.4 * laplacian_variance + 0.3 * edge_count + 0.2 * contrast - 0.1 * noise


def plate_exist(model, image):
    results = model(image)
    plate_image = None  # Default to None if no plate is found

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box:
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                plate_image = image[y1:y2, x1:x2]
                return True, plate_image

    # If no boxes or plates were found, return False and None
    return False, plate_image


def preprocess_image(image):
    # Resize image (if needed)
    height, width = image.shape[:2]
    desired_height, desired_width = (
        640,
        640,
    )  # Adjust based on your YOLO model's input size
    if (height, width) != (desired_height, desired_width):
        image = cv2.resize(image, (desired_width, desired_height))

    # Denoise image
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Sharpen image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Simple sharpening kernel
    image = cv2.filter2D(image, -1, kernel)

    return image


dir = "../../output/1"

ls = os.listdir(dir)
model = YOLO("../../runs-20240920T214004Z-001/runs/detect/train2/weights/best.pt")

max = 0
id = ""
for image in ls:
    imagepath = os.path.join(dir, image)
    img = cv2.imread(imagepath)
    img = preprocess_image(img)
    plate, imag = plate_exist(model=model, image=img)
    if plate and imag is not None:
        score = quality_score(imag)
        print(f"image: {image}, plate sharpness: {score}")
        if score > max:
            best_im = imag
            max = score
            id = image
    else:
        print("no plate found")
if max == 0:
    print("no picture with a plate found")
else:
    print(f"max sharpness: {max} in image {id}")
    cv2.imwrite("../../output/plateimage.png", best_im)
    print("image have been created")
