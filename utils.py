import os
from tqdm import tqdm
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def read_images_from_directory(image_directory: str) -> list:
    """
    > It takes a directory as input and returns a list of all the images in that directory

    :param image_directory: The directory where the images are stored
    :type image_directory: str
    :return: A list of images
    """

    list_of_images = list()
    for ext in ("*.gif", "*.png", "*.jpg"):
        list_of_images.extend(glob.glob(os.path.join(image_directory, ext)))
    print(f"Images found: {len(list_of_images)}")

    return list_of_images


def read_with_pil(list_of_images: list, resize=False) -> list:
    """
    > Reads a list of images and returns a list of PIL images

    :param list_of_images: list of image paths
    :type list_of_images: list
    :param resize: If True, resize the image to 512x512, defaults to False (optional)
    :return: A list of PIL images
    """

    pil_images = list()
    for img_path in tqdm(list_of_images):
        img = Image.open(img_path)
        if resize:  #! No hard code
            img.thumbnail((512, 512))
        pil_images.append(img)

    return pil_images


def prep_images(pil_images, device) -> list:
    """
    > Takse a list of PIL images, resize them to 384x384, convert them to tensors, and normalize them

    :param pil_images: A list of PIL images
    :param device: The device to run the model on
    :return: A list of tensors
    """
    image_size = 384
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # Transform Images
    t_images = [transform(img).unsqueeze(0).to(device) for img in pil_images]
    return t_images
