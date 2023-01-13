from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import utils
import torch
from models.blip import blip_decoder
from tqdm import tqdm

# to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read Images
list_of_images = utils.read_images_from_directory("./images")
pil_images = utils.read_with_pil(list_of_images)


# Transform Object
image_size = 384
transform = transforms.Compose(
    [
        transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)


# Transform Images
t_images = [transform(img).unsqueeze(0).to(device) for img in pil_images]
print("Images are transformed...")

# Model
print("Model Loading:")

model = blip_decoder(
    pretrained="./checkpoints/model_large_caption.pth", image_size=384, vit="large"
)
model.eval()
model = model.to(device)
print('model to device')

# Inference
with torch.no_grad():
    print('inference started')
    with open('captions.txt', 'w+') as file:
        for image in tqdm(t_images):

            caption = model.generate(
                image, sample=False, num_beams=3, max_length=20, min_length=5
            )
            file.write(caption[0]+'\n')

