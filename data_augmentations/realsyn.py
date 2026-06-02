import numpy as np
from PIL import Image
import scipy as sp
import cv2

def direct_paste(dest_img, src_img, cropped_source, cropped_mask, centre_x, centre_y):
    
    result = dest_img.copy()

    patch_height, patch_width = src_img.shape[:2]
    dest_height, dest_width = dest_img.shape[:2]

    top_left_x = centre_x - patch_width // 2
    top_left_y = centre_y - patch_height // 2

    valid_top_left_x = max(top_left_x, 0)
    valid_top_left_y = max(top_left_y, 0)

    valid_bottom_right_x = min(top_left_x + patch_width, dest_width)
    valid_bottom_right_y = min(top_left_y + patch_height, dest_height)



    result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x] = np.where(
            cropped_mask[..., None] == 1,
            cropped_source,
            result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x]
        )
    
    return result


def alpha_blending(dest_img, cropped_source, cropped_mask, centre_x, centre_y, feather_radius=15):
    
    result = dest_img.copy()

    patch_height, patch_width = cropped_source.shape[:2]
    dest_height, dest_width = dest_img.shape[:2]

    top_left_x = centre_x - patch_width // 2
    top_left_y = centre_y - patch_height // 2

    valid_top_left_x = max(top_left_x, 0)
    valid_top_left_y = max(top_left_y, 0)
    valid_bottom_right_x = min(top_left_x + patch_width, dest_width)
    valid_bottom_right_y = min(top_left_y + patch_height, dest_height)

    cropped_source = cropped_source[:valid_bottom_right_y - valid_top_left_y, :valid_bottom_right_x - valid_top_left_x]
    cropped_mask = cropped_mask[:valid_bottom_right_y - valid_top_left_y, :valid_bottom_right_x - valid_top_left_x]

    feathered_mask = cv2.GaussianBlur(cropped_mask.astype(np.float32), (feather_radius, feather_radius), 0)
    feathered_mask = feathered_mask[..., None]  

    for c in range(3):  
        result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x, c] = \
            (cropped_source[:, :, c] * feathered_mask[:, :, 0] +
             result[valid_top_left_y:valid_bottom_right_y, valid_top_left_x:valid_bottom_right_x, c] * (1 - feathered_mask[:, :, 0]))
    
    return result



def read_image(path_to_image: str, mask_image: bool, scale: bool=False):
    
    img = Image.open(path_to_image)

    if mask_image:
        img = img.convert("L") # greyscale
        binary_mask = np.array(img) > 127
        return binary_mask.astype(np.uint8)

    img = np.array(img.convert('RGB'))
    if scale:
        return img.astype('double') / 255.0

    return img

def apply_clahe_mask(image, mask):

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

def get_neighbors(i: int, j: int, max_i: int, max_j: int):
    return [(i + di, j) for di in (-1, 1) if 0 <= i + di <= max_i] + \
        [(i, j + dj) for dj in (-1, 1) if 0 <= j + dj <= max_j]
        
def populate_normal(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W):

    counter = 0
    num_mask_pixels = len(y_coords)

    for index in range(num_mask_pixels):
        y, x = y_coords[index], x_coords[index]

        for ny, nx in get_neighbors(y, x, H-1, W-1):
            A[counter, pixel_idx_map[y][x]] = 1
            
            b[counter] = src_image_test[y][x] - src_image_test[ny][nx]

            if pixel_idx_map[ny][nx] != -1:
                A[counter, pixel_idx_map[ny][nx]] = -1
            else:
                b[counter] += dest_image_test[ny][nx]

            counter += 1
    
    return A, b

def populate_mixed(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W):

    counter = 0
    num_mask_pixels = len(y_coords)

    for index in range(num_mask_pixels):
        y, x = y_coords[index], x_coords[index]

        for ny, nx in get_neighbors(y, x, H-1, W-1):
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

def compute_poisson_blend_channel(src_image_test, dest_image_test, mask, mode='normal'):

    H, W = src_image_test.shape

    num_mask_pixels = mask.sum().astype(int)
    pixel_idx_map = np.full(mask.shape, -1, dtype=int)
    y_coords, x_coords = np.where(mask == 1)
    pixel_idx_map[mask > 0] = np.arange(num_mask_pixels)

    A = sp.sparse.lil_matrix((4 * num_mask_pixels, num_mask_pixels))
    b = np.zeros(4 * num_mask_pixels)

    if mode=='normal':

        A, b = populate_normal(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W)

    if mode=='mixed':
        A, b = populate_mixed(A, b, y_coords, x_coords, pixel_idx_map, src_image_test, dest_image_test, H, W)


    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]

    copy_dest = dest_image_test.copy()

    for index in range(num_mask_pixels):
        y, x = y_coords[index], x_coords[index]
        copy_dest[y][x] = v[pixel_idx_map[y][x]]

    
    return np.clip(copy_dest, 0, 1)
