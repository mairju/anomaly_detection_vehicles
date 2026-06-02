import sys
sys.path.append('/home/maria/Documents/projects/anomaly_detection_study')

import matplotlib.pyplot as plt

def plot_first_step(dest_mask, placement_mask, centre_x, centre_y):
    plt.scatter(centre_x, centre_y, color='red', marker='o', s=30)
    plt.imshow(dest_mask, cmap='grey')
    plt.imshow(placement_mask, cmap='grey', alpha=0.7)
    plt.axis('off')
    plt.show()

def mask_on_top(image1, mask,  image2=None, save_path: str=None, verbose=False, titles=["Image1", "Image2"], fig_size=(15, 5)):

    fig, axes = plt.subplots(1, 2, figsize=fig_size)

    plt.figure(figsize=(5, 5))
    
    if image2 is not None:  
        axes[0].imshow(image2)
        axes[0].imshow(mask, cmap='viridis', alpha=0.5) 
    else: 
        axes[0].imshow(image1)
    axes[0].axis('off') 
    axes[0].set_title(titles[0])

    axes[1].imshow(image1)  
    axes[1].imshow(mask, cmap='viridis', alpha=0.5)  
    axes[1].axis('off') 
    axes[1].set_title(titles[1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  
        plt.close()
    else:  
        plt.show()

def display_images_side_by_side(images: list, titles: list=None, config_cmap: list[None, str]=None, save_path:list=None):

    num_images = len(images)
    
    if num_images < 1 or num_images > 4:
        raise ValueError("This function supports between 1 and 4 images.")
    
    fig, axes = plt.subplots(1, num_images, figsize=(8 * num_images, 8))
    
    if num_images == 1:
        axes = [axes]
    
    for i, image in enumerate(images):
        if config_cmap[i] is not None:
            axes[i].imshow(image, cmap=config_cmap[i])
        else:
            axes[i].imshow(image)
            
        axes[i].axis('off')  
        if titles and i < len(titles):  
            axes[i].set_title(titles[i])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()