import utils
import torch
from pathlib import Path

from models.blip import blip_decoder
from tqdm import tqdm
import argparse
import numpy as np


def init_parser(**parser_kwargs):
    """
    This function initializes the parser and adds arguments to it
    :return: The parser object is being returned.
    """
    parser = argparse.ArgumentParser(description="Image caption CLI")
    parser.add_argument("-i", "--input", help="Input directoryt path, such as ./images")
    parser.add_argument("-b", "--batch", help="Batch size", default=1, type=int)
    parser.add_argument(
        "-p", "--paths", help="A any.txt files contains all image paths."
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        default=0,
        help="gpu device to use (default=None) can be 0,1,2 for multi-gpu",
    )

    return parser


def init_model():
    """
    > Loads the model from the checkpoint file and sets it to eval mode
    :return: The model is being returned.
    """

    print("Checkpoint loading...")
    model = blip_decoder(
        pretrained="./checkpoints/model_large_caption.pth", image_size=384, vit="large"
    )
    model.eval()
    model = model.to(device)
    print(f"\nModel to {device}")
    return model


if __name__ == "__main__":

    parser = init_parser()
    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    if opt.paths:  # If filepath.txt file does not exists
        with open("filepaths.txt", "r") as file:
            list_of_images = file.read().split("\n")
    else:
        list_of_images = utils.read_images_from_directory(opt.input)

    # Batch processing
    split_size = len(list_of_images) // opt.batch
    print(f"Split size: {split_size}")
    batches = np.array_split(list_of_images, split_size)

    if not Path("checkpoints").is_dir():
        print(f"checkpoint directory did not found.")
        utils.create_dir("checkpoints")

    if not Path("checkpoints/model_large_caption.pth").is_file():
        utils.download_checkpoint()

    model = init_model()
    with torch.no_grad():
        print("Inference started")
        for batch_idx, batch in tqdm(enumerate(batches), unit="batch"):
            pil_images = utils.read_with_pil(batch)
            transformed_images = utils.prep_images(pil_images, device)

            if not Path("captions").is_dir():
                print(f"captions directory did not found.")
                utils.create_dir("captions")
                 
            with open(f"captions/{batch_idx}_captions.txt", "w+") as file:
                for path, image in zip(batch, transformed_images):

                    caption = model.generate(
                        image, sample=False, num_beams=3, max_length=20, min_length=5
                    )
                    file.write(path + ", " + caption[0] + "\n")
