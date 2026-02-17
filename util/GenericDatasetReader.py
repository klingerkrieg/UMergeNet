#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#jupyter nbconvert --to script GenericDatasetReader.ipynb
#Version 1.0
import os
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class GenericDatasetReader(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes, image_transform=None, mask_transform=None):
        self.image_dir       = image_dir
        self.mask_dir        = mask_dir
        self.image_transform = image_transform
        self.mask_transform  = mask_transform
        self.num_classes     = num_classes

        # list only .png and order for consistency
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png",".jpg", ".jpeg"))])

        # assembles pairs (img_path, mask_path) guaranteeing the same name
        self.pairs = []
        for img_file in self.image_files:
            img_path = os.path.join(image_dir, img_file)

            mask_name, ext = os.path.splitext(img_file)
            # Masks always assumed to be .png
            mask_path = os.path.join(mask_dir, mask_name + ".png") 
            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
            else:
                print(f"[Warning] Mask not found for{mask_dir}/{img_file}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            if self.num_classes == 2:
                mask_np = np.array(mask)
                mask_np = (mask_np > 0).astype(np.uint8) * 255
                mask = Image.fromarray(mask_np)
            mask = self.mask_transform(mask)

        return image, mask


def get_datasets(dataset_dir, resolution, batch_size=16, num_workers=4, num_classes=2):

    # Transformations
    image_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if num_classes == 2:
        # Binary segmentation
        mask_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
    else:
        # Multi-class segmentation
        mask_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=Image.NEAREST),
            transforms.Lambda(lambda x: from_numpy(np.array(x, dtype=np.int64)).unsqueeze(0))
        ])

    train_dataset = GenericDatasetReader(
        image_dir=os.path.join(dataset_dir, "images/train"),
        mask_dir=os.path.join(dataset_dir, "labels/train"),
        image_transform=image_transform,
        mask_transform=mask_transform,
         num_classes=num_classes,
    )

    val_dataset = GenericDatasetReader(
        image_dir=os.path.join(dataset_dir, "images/valid"),
        mask_dir=os.path.join(dataset_dir, "labels/valid"),
        image_transform=image_transform,
        mask_transform=mask_transform,
         num_classes=num_classes,
    )

    test_dataset = GenericDatasetReader(
        image_dir=os.path.join(dataset_dir, "images/test"),
        mask_dir=os.path.join(dataset_dir, "labels/test"),
        image_transform=image_transform,
        mask_transform=mask_transform,
         num_classes=num_classes,
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_datasets(dataset_dir='/home/calculon/0Datasets/membrane/membrane_v2', resolution=256)
    print(len(train_loader.dataset), len(test_loader.dataset), len(val_loader.dataset),'total:', len(train_loader.dataset) + len(test_loader.dataset) + len(val_loader.dataset))
    print(train_loader.dataset.pairs[0])
    print(train_loader.dataset[0][0].shape, train_loader.dataset[0][1].shape)

