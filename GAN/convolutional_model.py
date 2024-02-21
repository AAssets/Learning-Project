import torch
import torch.nn as nn
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='The parser for text generation')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of train')
parser.add_argument('--epoch', type=int, default=200, help='epoch of train')
parser.add_argument('--exp', type=str, default='GAN', help='exp name')
parser.add_argument('--latent_size', type=int, default='64', help='latent size')
parser.add_argument('--image_size', type=tuple, default=[1,28,28], help='image size')
parser.add_argument('--z_dimention', type=int, default='100', help='z dimention')
args = parser.parse_args()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(args.z_dimention, args.latent_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.latent_size * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(args.latent_size * 8, args.latent_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.latent_size * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(args.latent_size * 4, args.latent_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.latent_size * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(args.latent_size * 2, args.latent_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.latent_size),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(args.latent_size, args.image_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(args.image_size, args.latent_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(args.latent_size, args.latent_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.latent_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(args.latent_size * 2, args.latent_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.latent_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(args.latent_size * 4, args.latent_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.latent_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(args.latent_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)