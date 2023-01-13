import os
import glob
from PIL import Image


def read_images_from_directory(image_directory):
    """A brief description."""

    list_of_images = []
    for ext in ("*.gif", "*.png", "*.jpg"):
        list_of_images.extend(glob.glob(os.path.join(image_directory, ext)))

    return list_of_images


def read_with_pil(list_of_images, resize=False):
    """A brief description."""

    pil_images = []
    for img_path in list_of_images:
        img = Image.open(img_path)
        if resize:
            img.thumbnail((512, 512))  #! No hard code
        pil_images.append(img)

    return pil_images
