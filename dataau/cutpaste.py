import sys
sys.path.append('/home/maria/Documents/projects/anomaly_detection_study')

"""
Reference:

Li, Chun-Liang, et al. "Cutpaste: Self-supervised learning for anomaly detection and localization." Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition. 2021.
"""



import random
import cv2
import numpy as np


def cutpaste(source, destination):

    augmented_image = destination.copy()
    source = source.astype(np.float32)/255.
    height, width, _ = destination.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    num_patches = random.randint(1, 5)

    for _ in range(num_patches):
        use_scar = random.choice([True, False])

        if use_scar:
            patch_width = random.randint(10, 40)
            patch_height = random.randint(50, 200)

            source_top_left_x = random.randint(0, source.shape[1] - patch_width)
            source_top_left_y = random.randint(0, source.shape[0] - patch_height)
            patch = source[source_top_left_y:source_top_left_y + patch_height, source_top_left_x:source_top_left_x + patch_width]

            angle = random.uniform(0, 360)
            center = (patch_width // 2, patch_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            patch = cv2.warpAffine(patch, rotation_matrix, (patch_width, patch_height))
        else:
            patch_area_ratio = random.uniform(0.02, 0.15)
            aspect_ratio = random.uniform(0.3, 3.3)
            patch_width = int(np.sqrt(patch_area_ratio * width * height * aspect_ratio))
            patch_height = int(np.sqrt(patch_area_ratio * width * height / aspect_ratio))

            patch_width = min(patch_width, width)
            patch_height = min(patch_height, height)

            source_top_left_x = random.randint(0, source.shape[1] - patch_width)
            source_top_left_y = random.randint(0, source.shape[0] - patch_height)
            patch = source[source_top_left_y:source_top_left_y + patch_height, source_top_left_x:source_top_left_x + patch_width]

            patch = cv2.flip(patch, 1) if random.random() > 0.5 else patch

        patch_height, patch_width = patch.shape[:2]

        paste_x = random.randint(0, width - patch_width)
        paste_y = random.randint(0, height - patch_height)

        if paste_y + patch_height > augmented_image.shape[0]:
            patch_height = augmented_image.shape[0] - paste_y
        if paste_x + patch_width > augmented_image.shape[1]:
            patch_width = augmented_image.shape[1] - paste_x

        patch = patch[:patch_height, :patch_width]

        augmented_image[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width] = patch

        mask[paste_y:paste_y + patch_height, paste_x:paste_x + patch_width] = 1

    return augmented_image, np.expand_dims(mask, axis=2)