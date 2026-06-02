import cv2
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

def create_data_loader(data_dir, img_size=224, batch_size=32, val_split=0.2):

    transform = transforms.Compose([
        transforms.Resize((img_size, 2*img_size)), 
        transforms.ToTensor(), 
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_random_image_path(data_dir):

    all_files = os.listdir(data_dir)
    image_files = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise ValueError("Nothing found.")
    
    random_image = random.choice(image_files)
    return os.path.join(data_dir, random_image)

def read_agco(data_dir, resize, path_image=None):
    
    if path_image is None:
        random_image_path = get_random_image_path(data_dir=data_dir)
        print(random_image_path)
    else:
        random_image_path = path_image

    ima_dest = cv2.imread(random_image_path) 
    ima_dest = cv2.cvtColor(ima_dest, cv2.COLOR_BGR2RGB) 
    if resize:
        ima_dest = cv2.resize(ima_dest, resize)
    return ima_dest

def plot_images_from_loader(loader, num_images=5):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    num_images = min(num_images, len(images))
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i] if num_images > 1 else axes
        img = images[i].permute(1, 2, 0)  
        ax.imshow(img)
        ax.set_title(f"Label: {int(labels[i].item())}")
        ax.axis("off")

    plt.show()
