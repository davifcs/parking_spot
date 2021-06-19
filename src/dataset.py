import os
import glob

import pandas as pd


class DatasetAssemble:
    def __init__(self, labels_path, images_path, weather, camera_ids):
        self.labels_df = pd.DataFrame()
        self.images_list = []
        self.__parse_labels_csv(labels_path, camera_ids)
        self.__load_images_path(images_path, weather, camera_ids)

    def __parse_labels_csv(self, labels_path, camera_ids):
        csv_files = [labels_path + '/camera{}.csv'.format(i) for i in camera_ids]
        for f in csv_files:
            df_labels = pd.read_csv(f)
            df_labels['camera'] = os.path.splitext(f)[0][-1]
            self.labels_df = self.labels_df.append(df_labels)

    def __load_images_path(self, images_path, weather, camera_ids):
        for w in weather:
            for i in camera_ids:
                self.images_list.extend(glob.glob(images_path + "/" + w + '**/*/*{}/*jpg'.format(i), recursive=True))

