import argparse
import os
import torch

from torchvision import datasets

from datasets import *

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAMPLE_INTERVAL = 250
MODEL_INTERVAL = 10
N_EPOCHS = 200
LR = 0.0002
B1 = 0.5
B2 = 0.999

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Pix2Pix - Image to Image Translation')
    # fill parser with information about program arguments
    parser.add_argument('--dataset_name', default='facades',
                        help='Define the dataset name')
    parser.add_argument('--image_height', default=256,
                        help='Define size of image heigth')
    parser.add_argument('--image_width', default=256,
                        help='Define size of image width')
    parser.add_argument('--start_epoch', default=0,
                        help='Define epoch to start training from')
    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # add one more empty line for better output
    print()

def main():
    # Check device available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on: {}".format(device))
    # Parse and print arguments
    args = make_args_parser()
    print_args(args)
    # Make export directories
    os.makedirs("generated_images/{}".format(args.dataset_name), exist_ok=True)
    os.makedirs("saved_models/{}".format(args.dataset_name), exist_ok=True)
    # Setup dataloaders
    transform_= [
        transforms.Resize((args.image_height, args.image_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_dataloader = get_dataloader(root='./datasets/'+args.dataset_name,
                                      transform=transform_, mode='train', batch_size=1)
    val_dataloader = get_dataloader(root='./datasets/'+args.dataset_name,
                                    mode='val', transform=transform_, batch_size=10)
    # Declare tensor type
    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

if __name__ == '__main__':
    main()

