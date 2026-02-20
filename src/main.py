import logging
import os
import pathlib
import shutil
import sys
from typing import List, Tuple
import argparse

import cv2
from PIL import Image

import requests

class CroppedImageInfo:
    cropped_image: cv2.Mat
    lost_pixels = (0, 0)

class OutInfo:
    out_name: str
    lost_pixels: Tuple[int, int]

DPI = 300

SIZES_INCH = {
    "1x1": (1, 1),
    "2x2": (2, 2)
}

with open("config/api_key", "r") as f:
    API_KEY = f.read().strip()

SIZES_PIXELS = {k: (v[0] * DPI, v[1] * DPI) for k, v in SIZES_INCH.items()}

def process_image(image: cv2.Mat, leeway: int = 0) -> cv2.Mat:
    # Example image processing: convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.25, minNeighbors=7, minSize=(50, 50)
    )
    x, y, w, h = face[0]
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
    return info

def main():
    parser = argparse.ArgumentParser(
                prog='ID Image Generator',
                description='Crops images to 1x1 and 2x2 images focues on their faces.',
                epilog=' Recommended to use on single face images where its face is positioned at the center'
        )
    
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('output', type=str, help='Path to the output image')
    parser.add_argument('size', type=str, help='Size of the output image (e.g., 1x1, 2x2)')
    parser.add_argument('--leeway', type=int, default=0, help='How much of surrounding area to include around the face')
    parser.add_argument('--rembg', action='store_const', const=True, default=False, help='Remove BG? (Uses remove.bg API)' )
    args = parser.parse_args()

    offset = 0

    if args.size not in ["1x1", "2x2"]:
        logging.error("Invalid size argument. Please use '1x1' or '2x2'.")
        return 1

    cropped_image = process_image(cv2.imread(args.input), leeway=args.leeway)
    if not cropped_image:
        logging.warning(f"No faces were detected in the image: {args.input}")
        return 0  
    
    name = args.output
    
    logging.info(f"Writing cropped image to {name}")
    #cv2.imwrite(name, cropped_image.cropped_image)
    
    img = Image.fromarray(cv2.cvtColor(cropped_image.cropped_image, cv2.COLOR_BGR2RGB))
    w, h = img.size
    canvas_size = (w + cropped_image.lost_pixels[0], h + cropped_image.lost_pixels[1])
    canvas = Image.new(mode="RGB", size=canvas_size,   color=(255, 255, 255))
    canvas.paste(img, (cropped_image.lost_pixels[0] // 2, cropped_image.lost_pixels[1] // 2))
    
    resized_canvas = canvas.resize(SIZES_PIXELS[args.size], resample=Image.LANCZOS)
    try:
        resized_canvas.save(name)


    except Exception as e:
        logging.error(f"Failed to save image {name}: {e}")
        return 1
    
    
    if args.rembg:
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': open(name, 'rb')},
            data={'size': 'auto'},
            headers={'X-Api-Key': API_KEY},
        )
        if response.status_code == requests.codes.ok:
            with open(name, 'wb') as out:
                out.write(response.content)
            face = Image.open(name)
            final_canvas = Image.new(mode="RGBA", size=face.size, color=(255, 255, 255, 255))
            final_canvas.paste(face, (0, 0), face)
            final_canvas.save(name)
        else:
            logging.error(f"API Error {response.status_code} - {response.text}")
            os.remove(name)

    #
    return 0