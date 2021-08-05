# -*- encoding: utf-8 -*-
# @Time    :   4.08.2021
# @Author  :   Alperen Tahta
# @Contact :   alperentahta@gmail.com
# @Desc    :   None
import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import  pathlib
from dobby.datasets import data_files
from dobby.datasets.loaders.abstract_loader import AbstractLoader


class FerLoader(AbstractLoader):
    _split_filter = {'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}

    def __init__(self,
                 class_names: List[str] = 'all',
                 image_size: Tuple[int] = (48, 48)):

        if class_names == 'all':
            class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        path = pathlib.Path(data_files.__file__).resolve().parent / 'fer_2013.csv'
        super(FerLoader, self).__init__(path, class_names, 'FER')
        self.image_size = image_size
        self.all_data = self._read_data()

    def _read_data(self):
        data = pd.read_csv(self._path)
        data.pixels = data.pixels.apply(lambda x: np.fromstring(x, dtype='uint8', sep=' ').reshape(*self.image_size, 1))
        return data


    def load_data(self, split:str):
        """
        Args:
            split: Dataset split e.g training, val, test.
        Returns:
        """
        data = self.all_data[self.all_data.usage == self._split_filter[split]]
        features = np.stack(data.pixels, axis=0)
        labels = data.emotion.values.reshape(-1, 1)
        return {'features': features, 'labels': labels}

