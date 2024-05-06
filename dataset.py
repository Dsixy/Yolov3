import torch
from torch.utils.data import dataset, dataloader


class YoloDataset(dataset):
    def __init__(self, image_dir, label_dir, file):
        super(YoloDataset, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.file = file


    def __len__(self):
        pass

    def __getitem__(self, index):
        pass