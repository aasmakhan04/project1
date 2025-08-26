import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
import torchvision.transforms.functional as TF

class CocoSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, ann_file, transform=None, target_size=(128, 128), max_samples=500):
        self.coco = COCO(ann_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = list(self.coco.imgs.keys())[:max_samples]
        self.transform = transform
        self.target_size = target_size


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename.replace(".jpg", ".png"))

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = TF.resize(image, self.target_size)
        mask = TF.resize(mask, self.target_size, interpolation=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        if self.transform:
            image = self.transform(image)

        return image, mask

