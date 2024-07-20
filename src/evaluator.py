import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import ToTensor, Resize, Normalize

from models import Generator
from datasets import CustomDataset

class GAN_Evaluator:
    def __init__(self, generator, val_dataloader, log_on_wandb=False, device='cuda'):
        self.generator = generator.to(device).eval()
        self.val_dataloader = val_dataloader
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()

        self.transform = transforms.Compose([
            Resize((299, 299)),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.n_splits = 10

        if log_on_wandb:
	        # Initialize wandb
	        wandb.init(project="hlcv-gan-evaluation", entity="jaybhagiya")

    def calculate_rmse(self, real_images, generated_images):
        real_images = real_images.view(real_images.size(0), -1).cpu().detach().numpy()
        generated_images = generated_images.view(generated_images.size(0), -1).cpu().detach().numpy()
        return np.sqrt(mean_squared_error(real_images, generated_images))

    def perceptual_loss(self, real_images, generated_images):
        def extract_features(images, model):
            features = []
            for image in images:
                image = image.to(self.device)
                with torch.no_grad():
                    feature = model(image.unsqueeze(0))
                features.append(feature.cpu().numpy().reshape(-1))
            return np.array(features)

        real_features = extract_features(real_images, self.generator)
        gen_features = extract_features(generated_images, self.generator)
        perceptual_loss = np.mean(np.square(real_features - gen_features))
        return perceptual_loss

    # scale an array of images to a new size
    def scale_images(slef, images, new_shape):
        transform = Resize(new_shape[:2])
        images_list = [transform(image) for image in images]
        return torch.stack(images_list)

    def inception_score(self, preds, splits=10):
        N = preds.shape[0]
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py)))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    def evaluate(self):
        idx = 0
        rmse_scores = []
        perceptual_losses = []
        all_preds = []
        gen_image_list = []

        for synt_img, real_img in tqdm(self.val_dataloader, desc="Evaluating"):
            synt_img = synt_img.to(self.device)
            real_img = real_img.to(self.device)

            with torch.no_grad():
                generated_img = self.generator(synt_img)
                gen_image_list.append(generated_img)

            # Log RMSE and Perceptual Loss after each datapoint
            if log_on_wandb:
	            if (idx + 1) % 10 == 0:
	                wandb.log({"RMSE": self.calculate_rmse(real_img, generated_img)})
	                wandb.log({"Perceptual Loss": self.perceptual_loss(real_img, generated_img)})

            rmse_scores.append(self.calculate_rmse(real_img, generated_img))
            perceptual_losses.append(self.perceptual_loss(real_img, generated_img))

            # Resize images to the size expected by InceptionV3
            if (idx + 1) % self.n_splits == 0:
                all_gen_images = torch.cat(gen_image_list, dim=0)
                generated_image_resized = self.scale_images(all_gen_images, (299, 299, 3))

                with torch.no_grad():
                    gen_pred = self.inception_model(generated_image_resized)

                all_preds.append(F.softmax(gen_pred, dim=1).cpu().numpy())
                gen_image_list = []

            idx += 1

        all_preds = np.concatenate(all_preds, axis=0)
        is_mean, is_std = self.inception_score(all_preds)

        if log_on_wandb:
        	wandb.log({"Final RMSE": np.mean(rmse_scores), "Final Perceptual Loss": np.mean(perceptual_losses), "Inception Mean": is_mean, "Inception Std": is_std})
        	wandb.finish()

        return {
            "RMSE": np.mean(rmse_scores),
            "Perceptual Loss": np.mean(perceptual_losses),
            "Inception Score": (is_mean, is_std)
        }

if __name__ == '__main__':
	IMG_SHAPE = (3, 368, 544)
	BATCH_SIZE = 1
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	real_val_folder = "~/hlcv-project-gans/data/real/val/images"
	synt_val_folder = "~/hlcv-project-gans/data/synthetic/val/images"

	transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Resize((IMG_SHAPE[1], IMG_SHAPE[2]))
	])

	val_dataset = CustomDataset(real_val_folder, synt_val_folder, transform)
	val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

	# Load generator and discriminator weights
	gen_weights_path = '~/hlcv-project-gans/checkpoints/gen_weights_last'
	generator = Generator(img_shape=(3, 368, 544))
	generator.load_state_dict(torch.load(gen_weights_path))

	# Evaluate the generator
	evaluator = GAN_Evaluator(generator, val_dataloader)
	results = evaluator.evaluate()

	print(results)