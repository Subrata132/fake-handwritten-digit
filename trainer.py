import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from data_loader.data_loader import LoadData
from model.model import Generator, Discriminator


class Trainer:
    def __init__(
            self,
            noise_dim=100,
            batch_size=16,
            epoch=5,
            gen_lr=0.002,
            dis_lr=0.002,
            real_label=0.9,
            fake_label=0,
            csv_path='data/train.csv',
    ):
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epoch
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.real_label = real_label
        self.fake_label = fake_label
        self.csv_path = csv_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.print_every = 100

    def train(self):
        data_loader = LoadData(csv_path='data/train.csv').load_data()
        generator_model = Generator(input_size=self.noise_dim).to(device=self.device)
        discriminator_model = Discriminator().to(device=self.device)
        print("Generator parameters: ", sum(p.numel() for p in generator_model.parameters() if p.requires_grad))
        print("Discriminator parameters: ", sum(p.numel() for p in discriminator_model.parameters() if p.requires_grad))
        criterion = nn.BCEWithLogitsLoss()
        optimizer_generator = optim.Adam(generator_model.parameters(), self.gen_lr)
        optimizer_discriminator = optim.Adam(discriminator_model.parameters(), self.dis_lr)
        generator_model.train()
        discriminator_model.train()
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        plt.axis("off")
        all_dis_loss = []
        all_gen_loss = []
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            fixed_noise = np.random.uniform(-1, 1, size=(16, self.noise_dim)).astype(np.float32)
            fixed_noise = torch.from_numpy(fixed_noise)
            fixed_noise = fixed_noise.to(device=self.device)
            current_dis_loss = 0
            current_gen_loss = 0
            for real_images in tqdm(data_loader):
                batch_size = real_images.shape[0]
                real_images = real_images.to(device=self.device)
                optimizer_discriminator.zero_grad()
                output = discriminator_model(real_images)
                real_labels = torch.full((batch_size,), self.real_label, device=self.device)
                dis_loss_real = criterion(output.squeeze(), real_labels)
                noise = np.random.uniform(-1, 1, size=(batch_size, self.noise_dim)).astype(np.float32)
                noise = torch.from_numpy(noise)
                noise = noise.to(device=self.device)
                fake_images = generator_model(noise)
                output = discriminator_model(fake_images)
                fake_labels = torch.full((batch_size,), self.fake_label, device=self.device)
                dis_loss_fake = criterion(output.squeeze(), fake_labels.float())
                dis_loss_total = dis_loss_real + dis_loss_fake
                dis_loss_total.backward()
                optimizer_discriminator.step()

                optimizer_generator.zero_grad()
                noise = np.random.uniform(-1, 1, size=(batch_size, self.noise_dim)).astype(np.float32)
                noise = torch.from_numpy(noise)
                noise = noise.to(device=self.device)
                fake_images = generator_model(noise)
                output = discriminator_model(fake_images)
                gen_loss = criterion(output.squeeze(), real_labels.float())
                gen_loss.backward()
                optimizer_generator.step()

                current_dis_loss = current_dis_loss + dis_loss_total.item()
                current_gen_loss = gen_loss.item()
            all_dis_loss.append(current_dis_loss/len(data_loader))
            all_gen_loss.append(current_gen_loss/len(data_loader))
            print(
                'Discriminator loss: {:6.4f} | Generator loss: {:6.4f}'.format(all_dis_loss[-1], all_gen_loss[-1]))
            with torch.no_grad():
                generator_model.eval()
                generated_images = generator_model(fixed_noise).detach().cpu()
                generator_model.train()
                for i, ax in enumerate(axes.ravel()):
                    ax.imshow(np.transpose(generated_images[i].reshape((1, 28, 28)), (1, 2, 0)), cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.pause(1)









