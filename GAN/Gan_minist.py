import torch
import torchvision
import torch.nn as nn
import numpy as np
import argparse
import swanlab
import convolutional_model
from convolutional_model import Generator
from convolutional_model import Discriminator
from convolutional_model import args

logdir="./logs"

swanlab.init(
   experiment_name=args.exp,
   config=args,
  logdir=logdir
    )

use_gpu = torch.cuda.is_available()

# Data setting
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(64),
                                             torchvision.transforms.ToTensor(),
                                         ]
                                                                             )
                                     )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

generator = Generator()
discriminator = Discriminator()


g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(args.batch_size, 1)
labels_one = labels_one.squeeze(1)
labels_zero = torch.zeros(args.batch_size, 1)
labels_zero = labels_one.squeeze(0)

if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

# Training
iteration_num=0
for epoch in range(args.epoch):
    for i, mini_batch in enumerate(dataloader):
        iteration_num+=1
        gt_images, _ = mini_batch


        z = torch.randn(args.batch_size, args.z_dimention, 1, 1,)

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
            torchvision.utils.save_image(image, f'C:\LearningGit\Learning Project\GAN\Conv test train 1\image_{len(dataloader)*epoch+i}.png', nrow=4)

    swanlab.log({"g loss": g_loss / 100},iteration_num)
    swanlab.log({"d loss": d_loss / 100},iteration_num)
    swanlab.log({"real loss": real_loss / 100},iteration_num)
    swanlab.log({"fake loss": fake_loss / 100},iteration_num)