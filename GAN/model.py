import torch
import torch.nn as nn
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='The parser for text classification')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate of train')
parser.add_argument('--epoch', type=int, default=200, help='epoch of train')
parser.add_argument('--exp', type=str, default='GAN', help='exp name')
parser.add_argument('--latent_size', type=int, default='64', help='latent size')
parser.add_argument('--image_size', type=int, default='784', help='image size')
args = parser.parse_args()



class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, np.prod(args.image_size, dtype=np.int32)),
            nn.Sigmoid(),
        )

    def forward(self, z):

        output = self.model(z)
        image = output.reshape(z.shape[0], *args.image_size)

        return image


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(args.image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):

        prob = self.model(image.reshape(image.shape[0], -1))

        return prob