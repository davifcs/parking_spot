import os
import glob

import pandas as pd
import numpy as np
import torch
import PIL.Image as Image
from torch.utils.data import Dataset

class CNRExtDataset(Dataset):
    def __init__(self,
                 targets_dir: str = '',
                 images_dir: str = '',
                 weather: list = [],
                 camera_ids: list = [],
                 image_size: list = [],
                 label_image_size: list = [],
                 ):

        self.images_paths = []
        self.targets_paths = [targets_dir + '/camera{}.csv'.format(i) for i in camera_ids]
        self.targets_df = pd.DataFrame()

        self.image_size = image_size
        self.label_image_size = label_image_size

        self.__parse_bounding_boxes_csv()
        self.__load_images_path(images_dir, weather, camera_ids)

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx: int):
        image_path = self.images_paths[idx]
        target = self.targets_df.loc[self.targets_df['camera'] == os.path.dirname(image_path)[-1:]].values

        image = Image.open(image_path)
        image = image.resize((1280, 640), Image.ANTIALIAS)
        image = torch.from_numpy(np.array(image))
        image = image.permute(2, 1, 0)
        image = image/255.0

        target = torch.from_numpy(target)

        return image, target

    def __parse_bounding_boxes_csv(self):
        for f in self.targets_paths:
            targets_df = pd.read_csv(f)
            targets_df[['X', 'W']] = targets_df[['X', 'W']] * self.image_size[0] / self.label_image_size[0]
            targets_df[['Y', 'H']] = targets_df[['Y', 'H']] * self.image_size[1] / self.label_image_size[1]
            targets_df['camera'] = int(os.path.splitext(f)[0][-1])

            self.targets_df = self.targets_df.append(targets_df)

    def __load_images_path(self, images_dir, weather, camera_ids):
        for w in weather:
            for i in camera_ids:
                self.images_paths.extend(glob.glob(images_dir + "/" + w + '**/*/*{}/*jpg'.format(i), recursive=True))