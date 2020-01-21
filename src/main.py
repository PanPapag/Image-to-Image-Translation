import argparse
import os
import sys
import torch

from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from datasets import *
from models import *

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
    os.makedirs("../generated_images/{}".format(args.dataset_name), exist_ok=True)
    os.makedirs("../saved_models/{}".format(args.dataset_name), exist_ok=True)
    # Setup dataloaders
    transform_= [
        transforms.Resize((args.image_height, args.image_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    train_dataloader = get_dataloader(root='../datasets/'+args.dataset_name,
                                      transform=transform_, mode='train', batch_size=1)
    val_dataloader = get_dataloader(root='../datasets/'+args.dataset_name,
                                    mode='val', transform=transform_, batch_size=10)
    # Declare tensor type
    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()
    # Initialize Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    # Refs: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py
    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, args.image_height // 2 ** 4, args.image_width // 2 ** 4)

    if device == 'cuda':
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_pixelwise = criterion_pixelwise.cuda()

    if args.start_epoch != 0:
        # Load pretrained models
        generator.load_state_dict(
            torch.load("../saved_models/%s/generator_%d.pth" % (args.dataset_name, args.start_epoch)))
        discriminator.load_state_dict(
            torch.load("../saved_models/%s/discriminator_%d.pth" % (args.dataset_name, args.start_epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Setup optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(B1, B2))

    # Train model
    for epoch in range(args.start_epoch, N_EPOCHS):
        for batch_idx, data in enumerate(train_dataloader):
            # Model inputs
            real_A = Variable(data["B"].type(Tensor))
            real_B = Variable(data["A"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            batches_done = epoch * len(train_dataloader) + batch_idx
            sys.stdout.write("\rEpoch: [{}/{}] Batch: [{}/{}] G_Loss: {} D_Loss: {}".format(
                epoch,
                N_EPOCHS,
                batch_idx,
                len(train_dataloader),
                loss_G.item(),
                loss_D.item()))

            if batches_done % SAMPLE_INTERVAL == 0:
                data = next(iter(val_dataloader))
                real_A = Variable(data["B"].type(Tensor))
                real_B = Variable(data["A"].type(Tensor))
                fake_B = generator(real_A)
                img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
                save_image(img_sample, "../saved_images/{}/{}.png".format(args.dataset_name, batches_done),
                           nrow=5, normalize=True)

        if epoch % MODEL_INTERVAL == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(),
                       "../saved_models/%s/generator_%d.pth".format(args.dataset_name, epoch))
            torch.save(discriminator.state_dict(),
                        "../saved_models/%s/discriminator_%d.pth".format(args.dataset_name, epoch))

if __name__ == '__main__':
    main()

