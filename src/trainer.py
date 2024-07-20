import torch
import torch.nn as nn

import torch.optim as optim
import torchvision.transforms as transforms 

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from datasets import CustomDataset
import wandb

def save_checkpoint(model, path):
    print("=> Saving checkpoint")
    torch.save(model.state_dict(), path)

def train(gen, disc, train_loader, val_loader, g_optim, d_optim, loss, device, n_epochs=100, regularization=100):
    wandb.init(project="hlcv-gan-training", entity="jaybhagiya")  # Initialize wandb project

    gen_checkpoint_path = "~/hlcv-project-gans/checkpoints/1907_best_gen_weights.pth"
    dis_checkpoint_path = "~/hlcv-project-gans/checkpoints/1907_best_dis_weights.pth"

    best_vloss = float('inf')

    for epoch in range(n_epochs):
        gen.train()
        disc.train()

        gen_loss = 0.0
        dis_loss = 0.0
        total_loss = 0.0

        for idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)

            # Train Discriminator
            d_optim.zero_grad()
            fake_gen = gen(input)
            real_pred = disc(input, label)
            fake_pred = disc(input, fake_gen)

            d_real_loss = loss(real_pred, torch.ones_like(real_pred))
            d_fake_loss = loss(fake_pred, torch.zeros_like(fake_pred))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Train Generator
            g_optim.zero_grad()
            fake_gen = gen(input)
            fake_pred = disc(input, fake_gen)
            g_fake_loss = loss(fake_pred, torch.ones_like(fake_pred))
            l1_loss = nn.L1Loss()
            g_loss = g_fake_loss + regularization * l1_loss(fake_gen, label)
            g_loss.backward()
            g_optim.step()

            t_loss = g_loss + d_loss
            gen_loss += g_loss.item()
            dis_loss += d_loss.item()
            total_loss += t_loss.item()

            if idx % 10 == 0:
                wandb.log({
                    "Generator Loss (Batch)": g_loss.item(),
                    "Discriminator Loss (Batch)": d_loss.item(),
                    "Total Loss (Batch)": t_loss.item()
                })

            if idx % 100 == 0:
                fake_gen = gen(input).detach().cpu()
                imgs = make_grid(fake_gen)
                wandb.log({"Generated Images": [wandb.Image(imgs.permute(1, 2, 0), caption=f"Epoch {epoch + 1}, Batch {idx}")]})

        avg_gen_loss = gen_loss / len(train_loader)
        avg_dis_loss = dis_loss / len(train_loader)
        avg_loss = total_loss / len(train_loader)

        print(f'[Train] Epoch [{epoch + 1}/{n_epochs}], Generator_loss: {avg_gen_loss:.3f}, Discriminator_loss: {avg_dis_loss:.3f}, Total_loss: {avg_loss:.3f}')
        wandb.log({
            "Epoch": epoch + 1,
            "Generator Loss": avg_gen_loss,
            "Discriminator Loss": avg_dis_loss,
            "Total Loss": avg_loss
        })

        gen.eval()
        disc.eval()

        gen_vloss = 0.0
        dis_vloss = 0.0
        total_vloss = 0.0

        with torch.no_grad():
            for idx, (input, label) in enumerate(val_loader):
                input, label = input.to(device), label.to(device)
                fake_gen = gen(input)
                real_pred = disc(input, label)
                fake_pred = disc(input, fake_gen)

                d_real_loss = loss(real_pred, torch.ones_like(real_pred))
                d_fake_loss = loss(fake_pred, torch.zeros_like(fake_pred))
                d_vloss = d_real_loss + d_fake_loss

                fake_pred = disc(input, fake_gen)
                g_fake_loss = loss(fake_pred, torch.ones_like(fake_pred))
                g_vloss = g_fake_loss + regularization * l1_loss(fake_gen, label)

                t_vloss = g_vloss + d_vloss
                gen_vloss += g_vloss.item()
                dis_vloss += d_vloss.item()
                total_vloss += t_vloss.item()

        avg_gen_vloss = gen_vloss / len(val_loader)
        avg_dis_vloss = dis_vloss / len(val_loader)
        avg_vloss = total_vloss / len(val_loader)

        print(f'[Val] Epoch [{epoch + 1}/{n_epochs}], Generator_vloss: {avg_gen_vloss:.3f}, Discriminator_vloss: {avg_dis_vloss:.3f}, Total_vloss: {avg_vloss:.3f}')
        wandb.log({
            "Epoch": epoch + 1,
            "Generator Validation Loss": avg_gen_vloss,
            "Discriminator Validation Loss": avg_dis_vloss,
            "Total Validation Loss": avg_vloss
        })

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            save_checkpoint(gen, gen_checkpoint_path)
            save_checkpoint(disc, dis_checkpoint_path)

    wandb.finish()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_SHAPE = (3, 368, 544)
    BATCH_SIZE = 2

    real_train_folder = "~/hlcv-project-gans/data/real/train/images"
    synt_train_folder = "~/hlcv-project-gans/data/synthetic/train/images"
    real_val_folder = "~/hlcv-project-gans/data/real/val/images"
    synt_val_folder = "~/hlcv-project-gans/data/synthetic/val/images"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SHAPE[1], IMG_SHAPE[2]))
    ])

    train_dataset = CustomDataset(real_train_folder, synt_train_folder, transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomDataset(real_val_folder, synt_val_folder, transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(IMG_SHAPE)
    discriminator= Discriminator(IMG_SHAPE)
    discriminator.to(device)
    generator.to(device)

    d_optim = optim.Adam(discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
    g_optim = optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
    loss = nn.BCEWithLogitsLoss()
    
    train(generator, discriminator, train_dataloader, val_dataloader, g_optim, d_optim, loss, device)
