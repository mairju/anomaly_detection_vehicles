import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
#from constants import GAMMA_PARAMS
from scipy.ndimage import median_filter
import cv2

## TODO: organize this file

def plot_patched_image(patched_image, patch_mask, patch_coordinates):

    (y1, x1), (y2, x2) = patch_coordinates

    cut_patch = patched_image[y1:y2, x1:x2]
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 2)
    plt.imshow(patched_image)
    plt.title('Patched Image')
    plt.axis('off')

    plt.subplot(1, 3, 1)
    plt.imshow(patch_mask, cmap='gray')
    plt.title('Patch Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cut_patch)
    plt.title('Patch Location')
    plt.axis('off')
    
    plt.tight_layout()
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

def plot_gamma_distribution(alpha, beta):

    samples = np.random.gamma(alpha, beta, 100000)  

    random_value = np.random.choice(samples)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=100, density=True, alpha=0.6, color='grey', edgecolor='black') 

    x = np.linspace(0, np.max(samples), 1000)
    pdf = (x ** (alpha - 1) * np.exp(-x / beta)) / (beta ** alpha * np.math.gamma(alpha))

    plt.plot(x, pdf, color='blue', lw=2, label='Theoretical PDF')  

    plt.axvline(random_value, color='red', linestyle='--', linewidth=2, label=f'Randomly Selected Value: {random_value:.2f}')
    plt.scatter(random_value, 0, color='red', s=200, zorder=5, edgecolor='grey', linewidth=2)

    plt.annotate(f'{random_value:.2f}', 
                 xy=(random_value, 0), 
                 xytext=(random_value + 7 * beta, 20 * beta),  
                 arrowprops=dict(facecolor='blue', shrink=0.03),
                 fontsize=12,
                 color='red')

    plt.title(f'Samples from Gamma Distribution (α={alpha}, β={beta})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()

"""
def compute_hw_distributions(hmin, hmax, wmin, wmax, H, W, gamma_params=GAMMA_PARAMS, iterations=1000):

    h_values = []
    w_values = []


    for _ in range(iterations):
        rh, rw = np.random.gamma(shape=gamma_params['shape'], scale=gamma_params['scale'], size=2)

        h_perc = np.minimum(np.maximum(hmin, 0.06 + rh), hmax)
        w_perc = np.minimum(np.maximum(wmin, 0.06 + rw), wmax)
        
        h = (H * h_perc).round().astype(int)
        w = (W * w_perc).round().astype(int)
        
        h_values.append(h)
        w_values.append(w)
        
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(h_values, bins=30, color='grey', edgecolor='black')
    plt.title("Distribution of h")
    plt.xlabel("h values")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(w_values, bins=30, color='grey', edgecolor='black')
    plt.title("Distribution of w")
    plt.xlabel("w values")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
"""
def plot_patch_mask(x,y, patch_size, mask, mask_patch, image, perc):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    center_x = patch_size[1] // 2
    center_y = patch_size[0] // 2

    ax1.imshow(mask, cmap='gray')
    rect = patches.Rectangle((x, y), patch_size[1], patch_size[0], linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title("Original Mask with Patch Location")
    ax1.axis('off')

    ax2.imshow(image)
    rect = patches.Rectangle((x, y), patch_size[1], patch_size[0], linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title("Original Mask with Patch Location")
    ax2.axis('off')

    ax3.imshow(mask_patch, cmap='gray')
    ax3.set_title(f"Extracted Patch percentage={round(perc*100, 2)}")
    ax3.axis('off')

    plt.show()

def plot_rectangule_img(rect_coods, image):

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    #plt.axis('off')

    #rect_coods = (y_range, x_range) -> this sequence
    rect = patches.Rectangle(
        (rect_coods[1][0], rect_coods[0][0]), 
        rect_coods[1][1] - rect_coods[1][0],   
        rect_coods[0][1] - rect_coods[0][0],  
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    
    plt.gca().add_patch(rect)

    plt.title("Search Area for Patch (Red Rectangle)")
    plt.show()

def plot_patch_mask(x,y, patch_size1, mask, image, titles, patch_size2=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image)
    rect = patches.Rectangle((x, y), patch_size1[1], patch_size1[0], linewidth=1, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title(titles[0])
    #ax2.axis('off')


    ax2.imshow(mask, cmap='gray')
    if patch_size2 is None:
        rect = patches.Rectangle((x, y), patch_size1[1], patch_size1[0], linewidth=1, edgecolor='red', facecolor='none')
    else:
        rect = patches.Rectangle((x, y), patch_size2[1], patch_size2[0], linewidth=1, edgecolor='red', facecolor='none')

    ax2.add_patch(rect)
    ax2.set_title(titles[1])
    #ax1.axis('off')
    
    plt.show()

def mask_on_top_only(image, mask, save_path: str=None, verbose=False):
    if len(image.shape) == 2:  
        img_rgb = np.stack((image,) * 3, axis=-1)  
    else:
        img_rgb = image.copy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)  
    plt.imshow(mask, cmap='jet', alpha=0.5)  
    plt.axis('off') 

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if verbose:     
        plt.show()


def mask_on_top(image1, mask,  image2=None, save_path: str=None, verbose=False, titles=["Image1", "Image2"], fig_size=(15, 5)):

    fig, axes = plt.subplots(1, 2, figsize=fig_size)

    #plt.figure(figsize=(5, 5))
    
    if image2 is not None:  
        axes[0].imshow(image2)
        axes[0].imshow(mask, cmap='jet', alpha=0.5) 
    else: 
        axes[0].imshow(image1)
    axes[0].axis('off') 
    axes[0].set_title(titles[0])

    axes[1].imshow(image1)  
    axes[1].imshow(mask, cmap='jet', alpha=0.5)  
    axes[1].axis('off') 
    axes[1].set_title(titles[1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if verbose:     
        plt.show()

def post_blend_label(ima_dest: np.ndarray, patchex: np.ndarray, label_mode: str, intensity_logistic_params: tuple = (1.0, 0.0), tol=1) -> tuple:

    difference = np.abs(ima_dest - patchex)
    label_mask = (np.mean(difference, axis=-1, keepdims=True) > tol).astype(np.uint8)
    label_mask[..., 0] = median_filter(label_mask[..., 0], size=5)  

    if label_mode == 'continuous':
        factor = np.random.uniform(0.05, 0.95)
        label = label_mask * factor
    elif label_mode in ['logistic-intensity', 'intensity']:
        k, x0 = intensity_logistic_params
        label = np.mean(difference * label_mask, axis=-1, keepdims=True)
        label[..., 0] = median_filter(label[..., 0], size=5)
        if label_mode == 'logistic-intensity':
            label = label_mask / (1 + np.exp(-k * (label - x0)))
    elif label_mode == 'binary':
        label = label_mask
    else:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    return patchex, label

def get_neighbors(i: int, j: int, max_i: int, max_j: int):
    return [(i + di, j) for di in (-1, 1) if 0 <= i + di <= max_i] + \
        [(i, j + dj) for dj in (-1, 1) if 0 <= j + dj <= max_j]
        

def visualize_neighbors(ax, i, j, max_i, max_j):
    grid = np.zeros((max_i + 1, max_j + 1))

    grid[i, j] = 2

    for ni, nj in get_neighbors(i, j, max_i, max_j):
        grid[ni, nj] = 1

    ax.imshow(grid, cmap="cool", origin="upper", extent=[0, max_j, max_i, 0])



def extract_layer_info(model):
    layer_data = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.LeakyReLU, nn.ReLU, nn.Sigmoid)):
            layer_type = type(layer).__name__
            if hasattr(layer, 'in_channels'):
                in_channels = layer.in_channels
            else:
                in_channels = '-'
            if hasattr(layer, 'out_channels'):
                out_channels = layer.out_channels
            else:
                out_channels = '-'
            if hasattr(layer, 'kernel_size'):
                kernel_size = layer.kernel_size
            else:
                kernel_size = '-'
            if hasattr(layer, 'stride'):
                stride = layer.stride
            else:
                stride = '-'
            if hasattr(layer, 'padding'):
                padding = layer.padding
            else:
                padding = '-'
            if hasattr(layer, 'output_padding'):
                output_padding = layer.output_padding
            else:
                output_padding = '-'
            activation = 'LeakyReLU' if isinstance(layer, nn.LeakyReLU) else type(layer).__name__
            layer_data.append([name, layer_type, in_channels, out_channels, kernel_size, stride, padding, output_padding, activation])
    
    return layer_data

#layer_info = extract_layer_info(cae_model)

#df = pd.DataFrame(layer_info, columns=["Layer Name", "Layer Type", "Input Channels", "Output Channels", "Kernel Size", "Stride", "Padding", "Output Padding", "Activation"])

#display(df)