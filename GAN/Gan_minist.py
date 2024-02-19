import torch
import torchvision
import torch.nn as nn
import numpy as np
import argparse
import swanlab

parser = argparse.ArgumentParser(description='The parser for text classification')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate of train')
parser.add_argument('--epoch', type=int, default=200, help='epoch of train')
parser.add_argument('--exp', type=str, default='GAN', help='exp name')
parser.add_argument('--latent_size', type=int, default='64', help='latent size')
parser.add_argument('--image_size', type=int, default='784', help='image size')
args = parser.parse_args()

logdir="./logs"

swanlab.init(
    experiment_name=args.exp,
    config=args,
    logdir=logdir
    )

use_gpu = torch.cuda.is_available()

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

# Training
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                         ]
                                                                             )
                                     )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

generator = Generator()
discriminator = Discriminator()


g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(args.batch_size, 1)
labels_zero = torch.zeros(args.batch_size, 1)

if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

for epoch in range(args.epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch


        z = torch.randn(args.batch_size, args.latent_size)

        if use_gpu:
            gt_images = gt_images.to("cuda")
            z = z.to("cuda")

        pred_images = generator(z)
        g_optimizer.zero_grad()

        recons_loss = torch.abs(pred_images-gt_images).mean()

        g_loss = recons_loss*0.05 + loss_fn(discriminator(pred_images), labels_one)

        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()

        real_loss = loss_fn(discriminator(gt_images), labels_one)
        fake_loss = loss_fn(discriminator(pred_images.detach()), labels_zero)
        d_loss = (real_loss + fake_loss)


        d_loss.backward()
        d_optimizer.step()

        if i % 50 == 0:
            print(f"step:{len(dataloader)*epoch+i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if i % 400 == 0:
            image = pred_images[:16].data
            torchvision.utils.save_image(image, f"image_{len(dataloader)*epoch+i}.png", nrow=4)