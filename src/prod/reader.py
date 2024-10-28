from PIL import Image
import pytesseract
import cv2
import numpy as np

def preprocess_for_ocr(image):
    # Convert PIL Image to cv2 format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize image (2x upscale can help with recognition)
    scale_factor = 2
    enlarged = cv2.resize(
        gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
    )

    # Apply bilateral filter to reduce noise while keeping edges sharp
    denoised = cv2.bilateralFilter(enlarged, 11, 17, 17)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # Thresholding
    _, thresh = cv2.threshold(
        contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Ensure black text on white background
    if np.mean(cleaned[0]) > 127:
        cleaned = cv2.bitwise_not(cleaned)

    return cleaned

def process_license_plate(image):
    # Handle both file paths and image objects
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError("Could not read the image")

    # Basic preprocessing
    processed = preprocess_for_ocr(image)

    # Add padding
    padded = cv2.copyMakeBorder(
        processed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )

    return padded


def main():
    try:
        # Load and process image
        image_path = "../../output/plateimage.png"
        # image_path = "/home/hemmem/programming/Algerian_License_Recegnition_System/output/1/bbox_car_1_22.png"
        image = process_license_plate(image_path)

        # Configure Tesseract
        custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

        # Perform OCR
        text = pytesseract.image_to_string(image, config=custom_config)
        print(f"Detected Text: {text.strip()}")

        # Save processed image for debugging (optional)
        cv2.imwrite("../../output/processed_plate.jpg", image)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
