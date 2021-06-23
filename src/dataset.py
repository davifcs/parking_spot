import os
import glob

import pandas as pd
import numpy as np


class CNRExtDataloader():
    def __init__(self,
                 targets_dir: str = '',
                 images_dir: str = '',
                 weather: list = [],
                 camera_ids: list = [],
                 image_size: list = [],
                 label_image_size: list = [],
                 batch_size: int = 5,
                 shuffle: bool = False
                 ):

        self.images_paths = []
        self.targets = []
        self.targets_paths = [targets_dir + '/camera{}.csv'.format(i) for i in camera_ids]
        self.target_dict = {}

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_size = image_size
        self.label_image_size = label_image_size

        self.__parse_bounding_boxes_csv()
        self.__load_images_path(images_dir, weather, camera_ids)
        self.__image_target_pair()

    def __len__(self) -> int:
        return int(len(self.images_paths) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.images_paths))
        else:
            indices = [idx for idx in range(0, len(self.images_paths))]
        batch = []
        for i in indices:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield self.images_paths[batch].tolist(), self.targets[batch].tolist()
                batch = []

    def __parse_bounding_boxes_csv(self):
        for f in self.targets_paths:
            targets_df = pd.read_csv(f)
            targets_df[['X', 'W']] = targets_df[['X', 'W']] * self.image_size[0] / self.label_image_size[0]
            targets_df[['Y', 'H']] = targets_df[['Y', 'H']] * self.image_size[1] / self.label_image_size[1]

            targets_df['X1'] = targets_df['X']
            targets_df['Y1'] = targets_df['Y']
            targets_df['X2'] = targets_df['X'] + targets_df['W']
            targets_df['Y2'] = targets_df['Y'] + targets_df['H']

            targets_df[['camera']] = int(os.path.splitext(f)[0][-1])

            self.target_dict[int(os.path.splitext(f)[0][-1])] = \
                targets_df[['X1', 'Y1', 'X2', 'Y2', 'SlotId', 'camera']].values

    def __load_images_path(self, images_dir, weather, camera_ids):
        for w in weather:
            for i in camera_ids:
                self.images_paths.extend(glob.glob(images_dir + "/" + w + '**/*/*{}/*jpg'.format(i), recursive=True))
        self.images_paths = np.array(self.images_paths)

    def __image_target_pair(self):
        for path in self.images_paths:
            self.targets.append(self.target_dict[int(os.path.basename(os.path.dirname(path))[-1:])])
        self.targets = np.array(self.targets)
