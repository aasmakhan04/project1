import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

# Paths
image_dir = "train2017"
ann_file = "annotations/instances_train2017.json"
mask_dir = "masks_train2017"
os.makedirs(mask_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(ann_file)
image_ids = list(coco.imgs.keys())

for img_id in tqdm(image_ids, desc="Generating masks"):
    img_info = coco.loadImgs(img_id)[0]
    height, width = img_info['height'], img_info['width']
    mask = np.zeros((height, width), dtype=np.uint8)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        if ann.get("iscrowd", 0): continue
        cat_id = ann["category_id"]
        m = coco.annToMask(ann)
        mask[m == 1] = cat_id

    mask_image = Image.fromarray(mask)
    mask_image.save(os.path.join(mask_dir, f"{img_info['file_name'].split('.')[0]}.png"))
