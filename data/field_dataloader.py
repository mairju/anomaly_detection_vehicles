import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from data_augmentations.perlin import rand_perlin_2d_np  
import matplotlib.pyplot as plt

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class FieldDataset(Dataset):

    def __init__(self, root_dir, resize_shape=(256, 256*2)):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        print(self.resize_shape)

        self.good_images = sorted(glob.glob(os.path.join(root_dir, "good", "*.png")))
        self.abnormal_images = sorted(glob.glob(os.path.join(root_dir, "abnormal", "*.png")))

        self.images = self.good_images + self.abnormal_images

        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path=None):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found or unable to read: {mask_path}")
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.resize_shape:
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        mask = np.expand_dims(mask, axis=2)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        file_name = os.path.basename(img_path)
        dir_name = os.path.basename(os.path.dirname(img_path))

        if dir_name == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(self.root_dir, 'ground_truth', file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {
            'image': image,
            'mask': mask,
            'has_anomaly': has_anomaly,
            'idx': idx
        }

        return sample


class FieldTrainDataset(Dataset):

    def __init__(self, image_dir, resize_shape=(256, 256*2), anomaly_source_dir=None):

        self.image_dir = image_dir
        self.anomaly_source_dir = anomaly_source_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(f"{image_dir}/*/*.png") + glob.glob(f"{image_dir}/*/*.jpg") + glob.glob(f"{image_dir}/*/*.jpeg"))
        
        if self.anomaly_source_dir is not None:
            self.anomaly_source_paths = sorted(glob.glob(f"{anomaly_source_dir}/*/*.jpg") + glob.glob(f"{anomaly_source_dir}/*/*.png"))
            
            if len(self.anomaly_source_paths) == 0:
                raise ValueError(f"No anomaly source images found in {anomaly_source_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_indices = np.random.choice(len(self.augmenters), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[i] for i in aug_indices])
        return aug

# TODO: Add Poisson, CutPaste, FPI
    def augment_image(self, image, anomaly_source_path=None):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0

        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB) 
        anomaly_source_img = cv2.resize(anomaly_source_img, (self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)

        perlin_scalex = 2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).item()
        perlin_scaley = 2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).item()
        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)

        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, 1, 0).astype(np.float32)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).item() * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr

        no_anomaly = torch.rand(1).item()
        if no_anomaly > 0.5:
            return image.astype(np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            mask = perlin_thr.astype(np.float32)
            augmented_image = mask * augmented_image + (1 - mask) * image
            has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
            return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))

        if torch.rand(1).item() > 0.7:
            image = self.rot(image=image)

        image = image.astype(np.float32) / 255.0

        # TODO: ADD HERE the other argumentations
        if anomaly_source_path is not None:
            augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
            augmented_image = np.transpose(augmented_image, (2, 0, 1))
            
            return image, anomaly_mask, has_anomaly, augmented_image

        else:
            anomaly_mask = np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32)
            has_anomaly = np.array([0.0], dtype=np.float32)

        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        return image, anomaly_mask, has_anomaly, augmented_image
        

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        
        if self.anomaly_source_dir is not None:
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            anomaly_source_path = self.anomaly_source_paths[anomaly_source_idx]
        else:
            anomaly_source_path = None
        
        image_path = self.image_paths[idx]
        image, anomaly_mask, has_anomaly, augmented_image = self.transform_image(image_path, anomaly_source_path)

        sample = {
            'image': image,
            'augmented_image': augmented_image,
            'anomaly_mask': anomaly_mask,
            'has_anomaly': has_anomaly,
            'idx': idx
        }

        return sample
