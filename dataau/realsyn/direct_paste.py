import sys
sys.path.append('/home/maria/Documents/projects/anomaly_detection_study')

import numpy as np

class DirectPaste:
    def __init__(self, dest_img, src_img, cropped_source, cropped_mask, centre_y, centre_x):

        self.dest_img = dest_img
        self.src_img = src_img
        self.cropped_source = cropped_source
        self.cropped_mask = cropped_mask
        self.centre_x = centre_x
        self.centre_y = centre_y

    def apply(self):

        result = self.dest_img.copy()

        patch_height, patch_width = self.src_img.shape[:2]
        dest_height, dest_width = self.dest_img.shape[:2]

        top_left_x = self.centre_x - patch_width // 2
        top_left_y = self.centre_y - patch_height // 2

        valid_top_left_x = max(top_left_x, 0)
        valid_top_left_y = max(top_left_y, 0)

        valid_bottom_right_x = min(top_left_x + patch_width, dest_width)
        valid_bottom_right_y = min(top_left_y + patch_height, dest_height)



        result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x] = np.where(
                self.cropped_mask[..., None] == 1,
                self.cropped_source,
                result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x]
            )
        
        return result