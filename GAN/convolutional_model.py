import torch
import torch.nn as nn
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='The parser for text generation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate of train')
parser.add_argument('--epoch', type=int, default=200, help='epoch of train')
parser.add_argument('--exp', type=str, default='GAN', help='exp name')
parser.add_argument('--latent_size', type=int, default='64', help='latent size')
parser.add_argument('--image_size', type=int, default=1, help='image size')
parser.add_argument('--z_dimention', type=int, default='100', help='z dimention')
parser.add_argument('--G_size', type=int, default='64', help='size of feature maps in generator')
parser.add_argument('--D_size', type=int, default='64', help='size of feature maps in discriminator')
args = parser.parse_args()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(args.z_dimention, args.G_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.G_size * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(args.G_size * 8, args.G_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.G_size * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(args.G_size * 4, args.G_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.G_size * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(args.G_size * 2, args.G_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.G_size),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(args.G_size, args.image_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(args.image_size, args.D_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(args.D_size, args.D_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.D_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(args.D_size * 2, args.D_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.D_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(args.D_size * 4, args.D_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.D_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(args.D_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)