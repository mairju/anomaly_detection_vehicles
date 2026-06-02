import sys
sys.path.append('/home/maria/Documents/projects/anomaly_detection_study')

"""
Reference:

Pérez, Patrick, Michel Gangnet, and Andrew Blake. "Poisson image editing." Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 2023. 577-582.
"""


import cv2
import numpy as np
from PIL import Image
import scipy as sp

class PoissonBlending:
    def __init__(self, placement_image, dest_img, placement_mask):
        self.placement_image = placement_image
        self.dest_img = dest_img
        self.placement_mask = placement_mask
    
    def read_image(self, path_to_image: str, mask_image: bool, scale: bool=False):
    
        img = Image.open(path_to_image)

        if mask_image:
            img = img.convert("L") # greyscale
            binary_mask = np.array(img) > 127
            return binary_mask.astype(np.uint8)

        img = np.array(img.convert('RGB'))
        if scale:
            return img.astype('double') / 255.0

        return img

    def apply_clahe_mask(self, image, mask):

        image_uint8 = (image * 255).clip(0, 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        l_clahe_full = clahe.apply(l)

        l_clahe = l.copy()
        l_clahe[mask > 0] = l_clahe_full[mask > 0]

        lab_clahe = cv2.merge((l_clahe, a, b))

        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return result

    def get_neighbours(self, i: int, j: int, max_i: int, max_j: int):
        return [(i + di, j) for di in (-1, 1) if 0 <= i + di <= max_i] + \
            [(i, j + dj) for dj in (-1, 1) if 0 <= j + dj <= max_j]
            
    def populate_normal(self, A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W):

        counter = 0
        num_mask_pixels = len(y_coords)

        for index in range(num_mask_pixels):
            y, x = y_coords[index], x_coords[index]

            for ny, nx in self.get_neighbours(y, x, H-1, W-1):
                A[counter, pixel_idx_map[y][x]] = 1
                
                b[counter] = src_image_test[y][x] - src_image_test[ny][nx]

                if pixel_idx_map[ny][nx] != -1:
                    A[counter, pixel_idx_map[ny][nx]] = -1
                else:
                    b[counter] += dest_image_test[ny][nx]

                counter += 1
        
        return A, b

    def populate_mixed(self, A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W):

        counter = 0
        num_mask_pixels = len(y_coords)

        for index in range(num_mask_pixels):
            y, x = y_coords[index], x_coords[index]

            for ny, nx in self.get_neighbours(y, x, H-1, W-1):
                d1 = src_image_test[y][x] - src_image_test[ny][nx]
                d2 = dest_image_test[y][x] - dest_image_test[ny][nx]

                strongest = d1 if abs(d1) > abs(d2) else d2

                A[counter, pixel_idx_map[y][x]] = 1
                
                b[counter] = strongest

                if pixel_idx_map[ny][nx] != -1:
                    A[counter, pixel_idx_map[ny][nx]] = -1
                else:
                    b[counter] += dest_image_test[ny][nx]

                counter += 1
        
        return A, b

    def compute_poisson_blend_channel(self, src_image_test, dest_image_test, mask, mode):

        H, W = src_image_test.shape

        num_mask_pixels = mask.sum().astype(int)
        pixel_idx_map = np.full(mask.shape, -1, dtype=int)
        y_coords, x_coords = np.where(mask == 1)
        pixel_idx_map[mask > 0] = np.arange(num_mask_pixels)

        A = sp.sparse.lil_matrix((4 * num_mask_pixels, num_mask_pixels))
        b = np.zeros(4 * num_mask_pixels)

        if mode=='normal':

            A, b = self.populate_normal(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W)

        if mode=='mixed':
            A, b = self.populate_mixed(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W)


        A = sp.sparse.csr_matrix(A)
        v = sp.sparse.linalg.lsqr(A, b)[0]

        copy_dest = dest_image_test.copy()

        for index in range(num_mask_pixels):
            y, x = y_coords[index], x_coords[index]
            copy_dest[y][x] = v[pixel_idx_map[y][x]]

        
        return np.clip(copy_dest, 0, 1)
    
    def apply(self):
        pcomp_result_initial = np.dstack([
            self.compute_poisson_blend_channel(self.placement_image[:, :, c]/255., self.dest_img[:, :, c]/255., self.placement_mask, mode='normal')
            for c in range(3)
        ])
        
        pcomp_result = self.apply_clahe_mask(pcomp_result_initial, self.placement_mask)

        return pcomp_result, pcomp_result_initial