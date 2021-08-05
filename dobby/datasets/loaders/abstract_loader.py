# -*- encoding: utf-8 -*-
# @Time    :   4.08.2021
# @Author  :   Alperen Tahta
# @Contact :   alperentahta@gmail.com
# @Desc    :   None
from typing import List


class AbstractLoader:

    def __init__(self, path: str, class_names: List[str], name: str):
        """
        Abstract class for loading a dataset.
        Args:
            path: Path to data
            class_names: Label names of the features.
            name: Dataset name
        """
        self._path = path
        self._class_names = class_names
        self._name = name

    def load_data(self, split:str):
        """
        Abstract method for loading a dataset.
        Args:
            split: Dataset split e.g training, val, test.

        Returns: todo:write

        """

        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        self._split = split

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        self._class_names = class_names

    @property
    def num_classes(self):
        if isinstance(self.class_names, list):
            return len(self.class_names)
        else:
            raise ValueError('class names are not a list')

