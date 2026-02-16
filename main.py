import logging
import os
import pathlib
import sys
from typing import List, Tuple
import argparse

import cv2
from PIL import Image

class CroppedImageInfo:
    cropped_image: cv2.Mat
    lost_pixels = (0, 0)

class OutInfo:
    out_name: str
    lost_pixels: Tuple[int, int]

def process_image(image: cv2.Mat, leeway: int = 0) -> cv2.Mat:
    # Example image processing: convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.25, minNeighbors=7, minSize=(50, 50)
    )
    cropped_images = []

    for x, y, w, h in face:
        size = max(w, h) + leeway
        x_start = x - (size - w) // 2
        y_start = y - (size - h) // 2 
        x_end = x_start + size
        y_end = y_start + size
        if x_start < 0:
            x_start = 0
            logging.warning("It seems leeway includes out-of-bounds area, please make sure subject(s) are in the center or at least not at the edge or reduce leeway.")
        if y_start < 0:
            y_start = 0
            logging.warning("It seems leeway includes out-of-bounds area, please make sure subject(s) are in the center or at least not at the edge or reduce leeway.")
        if x_end > image.shape[1]:
            x_end = image.shape[1]
            logging.warning("It seems leeway includes out-of-bounds area, please make sure subject(s) are in the center or at least not at the edge or reduce leeway.")
        if y_end > image.shape[0]:
            y_end = image.shape[0]
            logging.warning("It seems leeway includes out-of-bounds area, please make sure subject(s) are in the center or at least not at the edge or reduce leeway.")
        info = CroppedImageInfo()
        info.cropped_image = image[y_start:y_end, x_start:x_end]
        info.lost_pixels = (size - (x_end - x_start), size - (y_end - y_start))
        cropped_images.append(info)
    return cropped_images
def main():
    parser = argparse.ArgumentParser(
                prog='ID Image Generator',
                description='Crops images to 1x1 images focues on their faces. Can detect multiple faces within single image.',
                epilog='Text at the bottom of help'
        )
    
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('output', type=str, help='Path to the output directory')
    parser.add_argument('--leeway', type=int, default=0, help='How much of surrounding area to include around the face')
    args = parser.parse_args()
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    offset = 0

    out_names = []
    for i, cropped_image in enumerate(process_image(cv2.imread(args.input), leeway=args.leeway)):
        name = f"{args.output}/cropped_{i + offset}.jpg"
        while pathlib.Path(name).exists():
            offset += 1
            logging.info(f"File {name} already exists, changing offset to {offset}")
            name = f"{args.output}/cropped_{i + offset}.jpg"  

        logging.info(f"Writing cropped image to {name}")
        cv2.imwrite(name, cropped_image.cropped_image)
        out_info = OutInfo()
        out_info.out_name = name
        out_info.lost_pixels = cropped_image.lost_pixels
        out_names.append(out_info)

    if not out_names:
        logging.warning(f"No faces were detected in the image: {args.input}")
        return 1

    for out_name in out_names:
        img = Image.open(out_name.out_name)
        w, h = img.size
        canvas_size = (w + out_name.lost_pixels[0], h + out_name.lost_pixels[1])
        print(out_name.lost_pixels)
        canvas = Image.new(mode="RGB", size=canvas_size,   color=(255, 255, 255))
        canvas.paste(img, (out_name.lost_pixels[0] // 2, out_name.lost_pixels[1] // 2))
        canvas.save(out_name.out_name)
    return 0
if __name__ == "__main__":
    sys.exit(main())