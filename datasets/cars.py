#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import os

import numpy as np
import scipy
import scipy.io as scio
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    check_integrity
)


class StanfordCars(VisionDataset):
    base_folder = 'stanford_cars'

    urls = {
        "train": "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
        "test": "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
        "devkit": "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
        "test_anno": "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
    }
    md5 = {
        "train": "065e5b463ae28d29e77c1b4b166cfe61",
        "test": "4ce7ebf6a94d07f1952d94dd34c4d501",
        "devkit": "c3b158d763b6e2245038c8ad08e45376",
        "test_anno": "b0a2b23655a3edd16d84508592a98d10",
    }

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform=None,
            target_transform=None,
            download: bool = False,
    ):
        super(StanfordCars, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.data_dir = os.path.join(self.root, self.base_folder)
        mat_anno = os.path.join(self.data_dir, 'devkit', f'cars_{split}_annos.mat') \
            if not split == "test" else os.path.join(self.data_dir,
                                                     'cars_test_annos_withlabels.mat')
        car_names = os.path.join(self.data_dir, 'devkit', 'cars_meta.mat')

        assert (split in ('train', 'test'))
        self.split = split

        if download:
            self.download()

        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])
        self.class_num = self.car_names.shape[0]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img_name = os.path.join(
            self.data_dir, f'cars_{self.split}',
            self.car_annotations[index][-1][0])

        img = Image.open(img_name).convert('RGB')
        car_class = self.car_annotations[index][-2][0][0]

        if self.transform is not None:
            img = self.transform(img)

        target = int(car_class) - 1

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.car_annotations)

    def _check_integrity(self) -> bool:
        for k in self.urls.keys():
            fpath = os.path.join(
                self.data_dir, os.path.basename(self.urls[k]))
            if not check_integrity(fpath, self.md5[k]):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        for k in self.urls.keys():
            if os.path.splitext(self.urls[k])[-1] == '.mat':
                download_url(self.urls[k], self.data_dir,
                             md5=self.md5[k])
            else:
                download_and_extract_archive(
                    self.urls[k], self.data_dir,
                    extract_root=self.data_dir,
                    md5=self.md5[k])

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
