# main imports
import argparse
import numpy as np

# image processing
import cv2
from PIL import Image




def main():

    parser = argparse.ArgumentParser(description="blur image")

    parser.add_argument('--image', type=str, help="folder with train/test folders within all features sub folders")
    parser.add_argument('--output', type=str, help="folder with train/test folders within all features sub folders")

    args = parser.parse_args()

    p_image  = args.image
    p_output = args.output

    img = Image.open(p_image)
    blur = cv2.GaussianBlur(np.array(img),(7,7),0)

    Image.fromarray(blur).save(p_output)

if __name__ == "__main__":
    main()