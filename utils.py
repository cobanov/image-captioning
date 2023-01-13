import os
from tqdm import tqdm
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def read_images_from_directory(image_directory):
    """A brief description."""

    list_of_images = []
    for ext in ("*.gif", "*.png", "*.jpg"):
        list_of_images.extend(sorted(glob.glob(os.path.join(image_directory, ext))))
    print(f"Images found: {len(list_of_images)}")

    return list_of_images


def read_with_pil(list_of_images, resize=False):
    """A brief description."""

    pil_images = []
    for img_path in tqdm(list_of_images):
        img = Image.open(img_path)
        if resize:
            img.thumbnail((512, 512))  #! No hard code
        pil_images.append(img)

    return pil_images


def prep_images(pil_images, device):
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
