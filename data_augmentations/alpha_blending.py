import numpy as np
import cv2

class AlphaBlender:
    def __init__(self, dest_img, cropped_source, cropped_mask, feather_radius=15):

        self.dest_img = dest_img
        self.cropped_source = cropped_source
        self.cropped_mask = cropped_mask
        self.feather_radius = feather_radius

    def apply(self, centre_x, centre_y):

        result = self.dest_img.copy()

        patch_height, patch_width = self.cropped_source.shape[:2]
        dest_height, dest_width = self.dest_img.shape[:2]

        top_left_x = centre_x - patch_width // 2
        top_left_y = centre_y - patch_height // 2

        valid_top_left_x = max(top_left_x, 0)
        valid_top_left_y = max(top_left_y, 0)
        valid_bottom_right_x = min(top_left_x + patch_width, dest_width)
        valid_bottom_right_y = min(top_left_y + patch_height, dest_height)

        cropped_source = self.cropped_source[:valid_bottom_right_y - valid_top_left_y, :valid_bottom_right_x - valid_top_left_x]
        cropped_mask = self.cropped_mask[:valid_bottom_right_y - valid_top_left_y, :valid_bottom_right_x - valid_top_left_x]

        feathered_mask = cv2.GaussianBlur(cropped_mask.astype(np.float32), (self.feather_radius, self.feather_radius), 0)
        feathered_mask = feathered_mask[..., None]  

        for c in range(3):  
            result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x, c] = (
                cropped_source[:, :, c] * feathered_mask[:, :, 0] +
                result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x, c] * (1 - feathered_mask[:, :, 0])
            )
        
        return result
