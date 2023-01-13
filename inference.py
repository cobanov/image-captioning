import utils
import torch
from models.blip import blip_decoder
from tqdm import tqdm
import argparse
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model():
    # Model initialization
    #
    # This repo may be deprecated later
    # It is kept like this in order not to
    # wait for the model to load every time
    # during the development phase.

    model = blip_decoder(
        pretrained="./checkpoints/model_large_caption.pth", image_size=384, vit="large"
    )
    model.eval()
    model = model.to(device)
    print("model to device")
    return model


if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description="Image caption CLI")
    parser.add_argument("-i", "--input", help="Input directoryt path, such as ./images")
    parser.add_argument("-b", "--batch", help="Batch size", default=1, type=int)
    parser.add_argument(
        "-p", "--paths", help="A any.txt files contains all image paths."
    )

    args = parser.parse_args()

    if args.paths:  # If filepath.txt file does not exists
        with open(args.paths, "r") as file:  #! Not tested yet
            list_of_images = file.read()
    else:
        list_of_images = utils.read_images_from_directory(args.input)

    # Batch processing
    split_size = len(list_of_images) // args.batch
    print(f"Split size: {split_size}")
    batches = np.array_split(list_of_images, split_size)

    # Inference
    model = init_model()
    with torch.no_grad():
        print("Inference started")
        for batch_idx, batch in enumerate(batches):
            pil_images = utils.read_with_pil(list_of_images)
            transformed_images = utils.prep_images(pil_images, device)

            with open("{batch_idx}_captions.txt", "w+") as file:
                for path, image in zip(list_of_images, transformed_images):

                    caption = model.generate(
                        image, sample=False, num_beams=3, max_length=20, min_length=5
                    )
                    file.write(path + ", " + caption[0] + "\n")
