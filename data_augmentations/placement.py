import sys
sys.path.append('/home/maria/Documents/projects/anomaly_detection_vehicles')

import numpy as np
import cv2
import random

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

class PlaceFinder:
    def __init__(self, dest_img, src_img, src_mask, min_target_height=250, max_target_height=430, min_object_ratio=0.3, verbose=True):
        self.dest_img = dest_img
        self.src_img = src_img
        self.src_mask = src_mask
        self.min_target_height = min_target_height
        self.max_target_height = max_target_height
        self.min_object_ratio = min_object_ratio
        self.verbose = verbose

    def field_mask(self):

        h = self.dest_img.shape[0] 
        middle_point_y = ((h)//2)+1

        height, width = self.dest_img.shape[:2]

        x1, y1 = 50, middle_point_y-25
        x2, y2 = width - 50, middle_point_y + 100

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=cv2.FILLED)

        return mask
    
    def get_bbox_extraction(self):
    
        contours, _ = cv2.findContours(self.src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        #bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)

        mask_with_bbox = cv2.cvtColor(self.src_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cropped_mask = self.src_mask[y_min:y_max, x_min:x_max]
        cropped_image = self.src_img[y_min:y_max, x_min:x_max]

        object_extracted = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

        return cropped_mask, cropped_image, object_extracted
    
    def calculate_resize_parameters(self, original_height):

        mean = (self.min_target_height + self.max_target_height) / 2
        std_dev = (self.max_target_height - self.min_target_height) / 4  # 95% within ±2std

        heights = np.random.normal(mean, std_dev, 1000)
        heights = np.clip(heights, self.min_target_height, self.max_target_height)

        target_height = np.random.choice(heights)

        return target_height / original_height
    
    def resize_object(self, src_image, src_mask, scale_factor):

        h, w = src_image.shape[:2]
        
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)

        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)

        resized_object = cv2.resize(src_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(src_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        return resized_object, resized_mask
    
    def get_placement_mask(self, source_image, source_mask, y_centre, x_centre):
    
        patch_height, patch_width = source_image.shape[:2]
        dest_height, dest_width = self.dest_img.shape[:2]
        
        top_left_x = x_centre - patch_width // 2
        top_left_y = y_centre - patch_height // 2

        valid_top_left_x = max(top_left_x, 0)
        valid_top_left_y = max(top_left_y, 0)
        valid_bottom_right_x = min(top_left_x + patch_width, dest_width)
        valid_bottom_right_y = min(top_left_y + patch_height, dest_height)

        crop_left = max(0, -top_left_x)
        crop_top = max(0, -top_left_y)
        crop_right = crop_left + (valid_bottom_right_x - valid_top_left_x)
        crop_bottom = crop_top + (valid_bottom_right_y - valid_top_left_y)

        cropped_source = source_image[crop_top:crop_bottom, crop_left:crop_right]
        cropped_mask = source_mask[crop_top:crop_bottom, crop_left:crop_right]

        placement_mask = np.zeros(self.dest_img.shape[:2], dtype=np.uint8)
        placement_mask[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x] = cropped_mask

        placement_image = self.dest_img.copy() #np.zeros(dest_img.shape, dtype=np.uint8)
        placement_image[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x] = cropped_source

        return cropped_source, cropped_mask, placement_mask, placement_image
    
    def check_contains_object(self, placement_mask, min_object_ratio, dest_mask):

        y_range, x_range = np.where(dest_mask > 0)

        obj_inside = placement_mask[y_range.min():y_range.max(), x_range.min():x_range.max()]
        
        object_pixels_inside = np.sum(obj_inside)
        object_pixels_outside = np.sum(placement_mask)
        total_pixels = (y_range.max()-y_range.min()) * (x_range.max()-x_range.min())
        object_ratio = object_pixels_inside / object_pixels_outside

        return object_ratio, object_ratio>=min_object_ratio
    
    def sample_centre(self, resized_image, resized_mask, dest_mask):
        # sample a pair until it finds (constrain)
        max_inter=200
        contains=False
        idx=0
        while (not contains) and (idx < max_inter):
            y_range, x_range = np.where(dest_mask > 0)
            centre_y = random.randint(y_range[0], y_range[-1]) # H
            centre_x = random.randint(x_range[0], x_range[-1]) # W

            #print(centre_y, centre_x)

            # crop the image if needed
            cropped_source, cropped_mask, placement_mask, placement_image = self.get_placement_mask(source_image=resized_image, source_mask=resized_mask, y_centre=centre_y, x_centre=centre_x)

            # check if at least 30% of the object is inside the 
            object_ratio, contains = self.check_contains_object(placement_mask=placement_mask, min_object_ratio=self.min_object_ratio, dest_mask=dest_mask)
            #print(object_ratio, contains)

            idx += 1
        
        return cropped_source, cropped_mask, placement_mask, placement_image, centre_x, centre_y
    

    def run(self):

        if self.verbose: print("1. Define a region where the object can be placed in the destination image. For this a mask is created.")
        dest_mask = self.field_mask()

        if self.verbose: 
            print("Create the mask for the destination image.")
            mask_on_top(image1=self.dest_img, mask=dest_mask,  image2=None, save_path=None, titles=["Target", "Target + Mast"], fig_size=(15, 5))

        if self.verbose: print("\n2. Randomly resize the object to introduce stochasticity in the synthetic image generation process.")
        
        cropped_mask, cropped_image, object_extracted = self.get_bbox_extraction()
        if self.verbose:
            print("a. Crop a bbox around the object in the source image so we can have an approximately size of it.")
            display_images_side_by_side(images=[cropped_image, cropped_mask, object_extracted], titles=["Image", "Mask", "Extracted Object"], config_cmap=[None, 'grey', None], save_path=None)

        scale_factor = self.calculate_resize_parameters(original_height=cropped_image.shape[0])
        if self.verbose:
            print(f"b. Sample from a normal destribution the new height of the image and compute the scale_factor={scale_factor}.")

        resized_image, resized_mask = self.resize_object(src_image=cropped_image, src_mask=cropped_mask, scale_factor=scale_factor)
        if self.verbose:
            print(f"c. Object resized - initial_size=({cropped_image.shape[0]},{cropped_image.shape[1]}), new_size=({resized_image.shape[0]},{resized_image.shape[1]}).")

        if self.verbose: print("\n3. Select a random center point for the object. If the object exceeds the destination image boundaries, crop it to fit while meeting the placement criterion.")
        cropped_fit_source, cropped_fit_mask, placement_mask, placement_image, centre_x, centre_y = self.sample_centre(resized_image=resized_image, resized_mask=resized_mask, dest_mask=dest_mask)
        if self.verbose:
            print(f"The sample_centre=({centre_y},{centre_x}).")
            plot_first_step(dest_mask=dest_mask, placement_mask=placement_mask, centre_x=centre_x, centre_y=centre_y)

        
        return resized_image, cropped_fit_source, cropped_fit_mask, placement_mask, placement_image, centre_x, centre_y