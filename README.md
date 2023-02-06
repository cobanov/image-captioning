# Image Captioning

Captioning is an img2txt model that uses the BLIP. Exports captions of images.

## Checkpoints [Required]

If there is no 'Checkpoints' folder, the script will automatically create the folder and download the model file, you can do this manually if you want.

Download the fine-tuned checkpoint and copy into 'checkpoints' folder (create if does not exists)

- [BLIP-Large](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth)

## Demo

<img src='./demo.jpg' width=500px>

```txt
datasets\0.jpg, a piece of cheese with figs and a piece of cheese
datasets\1002.jpg, a close up of a yellow flower with a green background
datasets\1005.jpg, a planter filled with lots of colorful flowers
datasets\1008.jpg, a teacher standing in front of a classroom full of children
datasets\1011.jpg, a tortoise on a white background with a white background
datasets\1014.jpg, a glass of wine sitting on top of a table
datasets\1017.jpg, a close up of a plant with pink flowers
datasets\102.jpg, a platter of different types of sushi
datasets\1020.jpg, a frog sitting on top of a bamboo stick
datasets\1023.jpg, a revolver on a white background
datasets\1026.jpg, a woman holding a small white dog in her arms
datasets\1029.jpg, a woman in a business suit standing in front of a building
datasets\1032.jpg, sliced cucumber on a white background
datasets\1035.jpg, a woman in glasses and a pair of boxing gloves
datasets\1038.jpg, a pile of sliced potatoes on a white surface
datasets\1041.jpg, two glasses of orange juice on a wooden table
datasets\1044.jpg, a woman sitting on the floor in front of a door

```

## Usage

```bash
usage: inference.py [-h] [-i INPUT] [-b BATCH] [-p PATHS] [-g GPU_ID]        

Image caption CLI

optional arguments:
  -h, --help                      show this help message and exit
  -i INPUT,  --input INPUT        Input directoryt path, such as ./images
  -b BATCH,  --batch BATCH        Batch size
  -p PATHS,  --paths PATHS        A any.txt files contains all image paths.
  -g GPU_ID, --gpu-id GPU_ID      gpu device to use (default=0) can be 0,1,2 for multi-gpu
```

### Example

```bash
python inference.py -i /path/images/folder --batch 8 --gpu 0
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
