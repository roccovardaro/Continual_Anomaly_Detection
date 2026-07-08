
from typing import Iterable

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd


def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable):
            yield from flatten(x)
        else:
            yield x


class VisADataset(Dataset):
    """VisA Anomaly Detection dataset.

    Directory structure expected (official VisA format):
        root_dir/
            <class_name>/
                Data/
                    Images/
                        Normal/   (*.JPG)
                        Anomaly/  (*.JPG)
                    Masks/
                        Anomaly/  (*.png)
            split_csv/
                1cls.csv          (official 1-class split)

    The CSV has columns: object, split, label, image, mask
    """

    def __init__(self, root_dir, task_visa_classes, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the VisA dataset.
            task_visa_classes (list): list of class names to load.
            size (int): image resize dimension.
            transform: Transform to apply to data.
            mode: "train" loads training samples, "test" loads test samples. Default "train".
        """
        self.root_dir = Path(root_dir)
        self.task_visa_classes = task_visa_classes
        self.transform = transform
        self.mode = mode
        self.size = size
        self.all_imgs = []
        self.all_image_names = []
        self.all_labels = []  # used only in test mode

        # Load the CSV split file
        csv_path = self.root_dir / "split_csv" / "1cls.csv"
        self.split_df = pd.read_csv(csv_path)

        for class_name in self.task_visa_classes:
            # Filter CSV for this class and split
            class_df = self.split_df[self.split_df['object'] == class_name]

            if self.mode == "train":
                train_df = class_df[(class_df['split'] == 'train') & (class_df['label'] == 'normal')]
                image_names = [self.root_dir / row['image'] for _, row in train_df.iterrows()]
                self.all_image_names.append(image_names)
                print("loading images")
                # during training we cache the smaller images for performance reasons
                imgs = (Parallel(n_jobs=10)(
                    delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in
                    image_names))
                self.all_imgs.append(imgs)
                print(f"loaded {class_name} : {len(imgs)} images")
            else:
                # test mode: load both normal and anomaly images
                test_df = class_df[class_df['split'] == 'test']
                image_names = [self.root_dir / row['image'] for _, row in test_df.iterrows()]
                labels = [row['label'] != 'normal' for _, row in test_df.iterrows()]
                self.all_image_names.append(image_names)
                self.all_labels.append(labels)

        self.all_imgs = list(flatten(self.all_imgs))
        self.all_image_names = list(flatten(self.all_image_names))
        if self.mode != "train":
            self.all_labels = list(flatten(self.all_labels))

    def __len__(self):
        return len(self.all_image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.all_imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.all_image_names[idx]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            label = self.all_labels[idx]  # False (0) se normal, True (1) se anomaly
            return img, label
