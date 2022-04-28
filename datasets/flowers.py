#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import os
import shutil

import scipy.io as scio
from PIL import Image, ImageFile
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    check_integrity
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Flowers102(VisionDataset):
    source_folder = '102flowers_org'
    base_folder = '102flowers'
    source = os.path.join(source_folder, 'jpg')
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    image_labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    set_id_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    filename = "102flowers.tgz"
    image_labels_filename = "imagelabels.mat"
    set_id_filename = "setid.mat"
    md5 = "52808999861908f626f3c1f4e79d11fa"
    image_labels_md5 = "e0620be6f572b9609742df49c70aed4d"
    set_id_md5 = "a5357ecc9cb78c4bef273ce3793fc85c"

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform=None,
            target_transform=None,
            download: bool = False,
    ):
        super(Flowers102, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        if download:
            self.download_and_arrange()

        assert (split in ('train', 'test'))
        self.split = split
        if split == 'train':
            downloaded_list = os.path.join(self.root, self.base_folder, "train")
        else:
            downloaded_list = os.path.join(self.root, self.base_folder, "test")

        self.data = []
        self.targets = []

        for i in range(102):
            for file_name in os.listdir(os.path.join(downloaded_list, str(i + 1))):
                if not file_name.endswith('.jpg'):
                    continue
                self.data.append(os.path.join(downloaded_list, str(i + 1), file_name))
                self.targets.append(i)

    def __getitem__(self, index: int):

        path, target = self.data[index], self.targets[index]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.base_folder, self.filename)
        if not check_integrity(fpath, self.md5):
            return False
        return True

    def download_and_arrange(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(
            self.url, self.root,
            extract_root=os.path.join(self.root, self.source_folder),
            filename=self.filename, md5=self.md5)
        download_url(self.image_labels_url,
                     os.path.join(self.root, self.source_folder),
                     filename=self.image_labels_filename,
                     md5=self.image_labels_md5)
        download_url(self.set_id_url,
                     os.path.join(self.root, self.source_folder),
                     filename=self.set_id_filename, md5=self.set_id_md5)

        image_labels = scio.loadmat(os.path.join(self.root, self.source_folder,
                                                 self.image_labels_filename))
        set_id = scio.loadmat(
            os.path.join(self.root, self.source_folder, self.set_id_filename))

        self.classify(set_id['trnid'][0], 'train', image_labels['labels'][0])
        self.classify(set_id['valid'][0], 'train', image_labels['labels'][0])
        self.classify(set_id['tstid'][0], 'test', image_labels['labels'][0])
        shutil.rmtree(os.path.join(self.root, self.source_folder))

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def classify(self, set_, split, labels):
        for n, id_ in enumerate(set_):
            cls = labels[id_ - 1]
            filename = f'image_{id_:05d}.jpg'
            dst = os.path.join(self.root, self.base_folder, split)
            path = os.path.join(dst, str(cls))
            path = path.strip()
            path = path.rstrip("/")
            os.makedirs(path, exist_ok=True)
            os.rename(os.path.join(self.root, self.source, filename),
                      os.path.join(dst, str(cls), filename))
