'''
Dataset download: https://cocodataset.org/#download

This implementation is done on the "2017 Train/Val/Test"
'''

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def get_coco_ids(data_dir, data_type, cls):
    ann_file = os.path.join(data_dir, f'annotations/instances_{data_type}.json')  
    image_dir = os.path.join(data_dir, data_type)  # Path to images
    
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=[cls])

    img_ids = coco.getImgIds(catIds=cat_ids)

    return coco, img_ids


def load_image_by_id(coco, img_ids, data_dir, data_type, cls, specific_id=None):
    
    if specific_id is None:
        img_info = coco.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]
        print(np.random.randint(0, len(img_ids)))
    else:
        img_info = coco.loadImgs(specific_id)[0]

    image_dir = os.path.join(data_dir, data_type)
    image_path = os.path.join(image_dir, img_info['file_name'])
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cat_ids = coco.getCatIds(catNms=[cls])
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    labels = []
    for ann in anns:
        cat_id = ann['category_id']
        cat_name = coco.loadCats(cat_id)[0]['name']
        labels.append(cat_name)

    print(f"Image ID: {img_info['id']}, File: {img_info['file_name']}")
    print("Labels found in this image:", labels)

    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    for ann in anns:
        rle = coco.annToRLE(ann)  
        m = maskUtils.decode(rle)  
        mask = np.maximum(mask, m) 

    return image_rgb, mask