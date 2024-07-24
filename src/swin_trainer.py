import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import wandb

from swin_models import SwinUNet, Discriminator
from datasets import CustomAugDataset

def save_checkpoint(model, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(model.state_dict(), filename)

def train(gen, disc, train_loader, val_loader, g_optim, d_optim, loss, checkpoint_folder, n_epochs=100, regularization=100, device='cuda'):

    gen_checkpoint_best = os.path.join(checkpoint_folder, "gen_swin_weights_best_test.pth")
    dis_checkpoint_best = os.path.join(checkpoint_folder, "dis_swin_weights_best_test.pth")
    gen_checkpoint_last = os.path.join(checkpoint_folder, "gen_swin_weights_last_test.pth")
    dis_checkpoint_last = os.path.join(checkpoint_folder, "dis_swin_weights_last_test.pth")

    best_vloss = np.inf
    for epoch in range(n_epochs):
        gen_loss = 0.0
        dis_loss = 0.0
        total_loss = 0.0
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_t_loss = 0.0
        avg_gen_loss = 0.0
        avg_dis_loss = 0.0
        avg_loss = 0.0

        gen_vloss = 0.0
        dis_vloss = 0.0
        total_vloss = 0.0
        avg_gen_vloss = 0.0
        avg_dis_vloss = 0.0
        avg_vloss = 0.0

        img_label_idx = 0

        for idx, pair in enumerate(train_loader):
            input, label = pair[0].to(device), pair[1].to(device)
            
            # Train Discriminator
            d_optim.zero_grad()
            fake_gen = gen(input)
            real_pred = disc(input, label)
            fake_pred = disc(input, fake_gen)

            d_real_loss = loss(real_pred, torch.ones_like(real_pred, device=device))
            d_fake_loss = loss(fake_pred, torch.zeros_like(fake_pred, device=device))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward(retain_graph=True)
            d_optim.step()

            # Train Generator
            g_optim.zero_grad()
            fake_pred = disc(input, fake_gen)
            g_fake_loss = loss(fake_pred, torch.ones_like(fake_pred, device=device))
            l1_loss = nn.L1Loss()
            g_loss = g_fake_loss + regularization * l1_loss(fake_gen, label)
            g_loss.backward(retain_graph=True)
            g_optim.step()

            # Log statistics
            t_loss = g_loss + d_loss
            gen_loss += g_loss.item()
            dis_loss += d_loss.item()
            total_loss += t_loss.item()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            running_t_loss += t_loss.item()

            if idx % 25 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Generator_loss: {running_g_loss/25:.3f}%, Discriminator_loss: {running_d_loss/25:.3f}%, Total_loss: {running_t_loss/25:.3f}')
                running_g_loss = 0
                running_d_loss = 0
                running_t_loss = 0

            if idx % 100 == 0:
                fake_gen = gen(input)
                imgs = make_grid(fake_gen).cpu()
                
                combined_label = torch.cat((label[0], label[1]), dim=2)
                wandb.log({"Generated Images": [wandb.Image(imgs.permute(1, 2, 0).detach().numpy(), caption=f"Epoch {epoch + 1}, Index {img_label_idx + 1}")]})
                wandb.log({"Real Images": [wandb.Image(combined_label.permute(1, 2, 0).detach().numpy(), caption=f"Epoch {epoch + 1}, Index {img_label_idx + 1}")]})
                img_label_idx += 1
                
                # plt.imshow(imgs.permute(1, 2, 0).detach().numpy())
                # plt.show()
                # plt.imshow(combined_label.permute(1, 2, 0).detach().numpy())
                # plt.show()

        avg_gen_loss = gen_loss / (idx + 1)
        avg_dis_loss = dis_loss / (idx + 1)
        avg_loss = total_loss / (idx + 1)
        print(f'Epoch [{epoch + 1}/{n_epochs}], Generator_loss: {avg_gen_loss:.3f}%, Discriminator_loss: {avg_dis_loss:.3f}%, Total_loss: {avg_loss:.3f}')
        img_label_idx = 0

        # Log to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Generator Loss": avg_gen_loss,
            "Discriminator Loss": avg_dis_loss,
            "Total Loss": avg_loss
        })

        # Validation
        for idx, pair in enumerate(val_loader):
            input, label = pair[0].to(device), pair[1].to(device)
            fake_gen = gen(input)
            real_pred = disc(input, label)
            fake_pred = disc(input, fake_gen)

            d_real_loss = loss(real_pred, torch.ones_like(real_pred, device=device))
            d_fake_loss = loss(fake_pred, torch.zeros_like(fake_pred, device=device))
            d_vloss = d_real_loss + d_fake_loss

            fake_pred = disc(input, fake_gen)
            g_fake_loss = loss(fake_pred, torch.ones_like(fake_pred, device=device))
            l1_loss = nn.L1Loss()
            g_vloss = g_fake_loss + regularization * l1_loss(fake_gen, label)

            t_vloss = g_vloss + d_vloss
            gen_vloss += g_vloss.item()
            dis_vloss += d_vloss.item()
            total_vloss += t_vloss.item()

        avg_gen_vloss = gen_vloss / (idx + 1)
        avg_dis_vloss = dis_vloss / (idx + 1)
        avg_vloss = total_vloss / (idx + 1)
        print(f'Epoch [{epoch + 1}/{n_epochs}], Generator_vloss: {avg_gen_vloss:.3f}%, Discriminator_vloss: {avg_dis_vloss:.3f}%, Total_vloss: {avg_vloss:.3f}')

        wandb.log({
            "Epoch": epoch + 1,
            "Generator Validation Loss": avg_gen_vloss,
            "Discriminator Validation Loss": avg_dis_vloss,
            "Total Validation Loss": avg_vloss
        })

        if avg_vloss < best_vloss:
            print("Best_vloss:", best_vloss)
            print("Avg Loss:", avg_vloss)
            best_vloss = avg_vloss
            save_checkpoint(gen, gen_checkpoint_best)
            save_checkpoint(disc, dis_checkpoint_best)

        save_checkpoint(gen, gen_checkpoint_last)
        save_checkpoint(disc, dis_checkpoint_last)


        wandb.log({
            "Generator Model": wandb.Artifact("generator", type="model", description="Generator model", metadata={"epoch": epoch + 1}),
            "Discriminator Model": wandb.Artifact("discriminator", type="model", description="Discriminator model", metadata={"epoch": epoch + 1})
        })

if __name__ == "__main__":
    IMG_SHAPE = (3, 384, 576)
    BATCH_SIZE = 2

    real_train_folder = "~/hlcv-project-gans/data/real/train/images"
    synt_train_folder = "~/hlcv-project-gans/data/synthetic/train/images"
    real_val_folder = "~/hlcv-project-gans/data/real/val/images"
    synt_val_folder = "~/hlcv-project-gans/data/synthetic/val/images"

    checkpoint_folder = "~/hlcv-project-gans/checkpoints/"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SHAPE[1], IMG_SHAPE[2]))
    ])

    train_dataset = CustomAugDataset(real_train_folder, synt_train_folder, transform, augment=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomAugDataset(real_val_folder, synt_val_folder, transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = SwinUNet(IMG_SHAPE[1], IMG_SHAPE[2], 3, 32, 3).to(device)
    discriminator = Discriminator(IMG_SHAPE).to(device)

    d_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optim = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    loss = nn.BCEWithLogitsLoss()

    # Initialize wandb
    wandb.init(project="hlcv-swin-gan", entity="jaybhagiya")

    train(
        generator, 
        discriminator, 
        train_dataloader, 
        val_dataloader, 
        g_optim, d_optim, 
        loss, 
        checkpoint_folder, 
        n_epochs=100, 
        regularization=100, 
        device=device
    )