import math
import random

import numpy as np
from PIL import Image
from torch.utils import data


def rand_bbox(size, ratio):
    w, h = size
    ratio = math.sqrt(1 - ratio)
    cut_w = int(w * ratio)
    cut_h = int(h * ratio)

    cx = random.randrange(w)
    cy = random.randrange(h)

    x1 = min(max(cx - cut_w // 2, 0), w)
    y1 = min(max(cy - cut_h // 2, 0), h)
    x2 = min(max(cx + cut_w // 2, 0), w)
    y2 = min(max(cy + cut_h // 2, 0), h)

    return x1, y1, x2, y2


class MixDataset(data.Dataset):
    def __init__(self, dataset, transform, mixup=0.2, cutmix=1):
        self.dataset = dataset
        self.transform = transform
        self.mixup = mixup
        self.cutmix = cutmix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]

        index2 = index
        while index2 == index:
            index2 = random.randrange(len(self.dataset))

        img2, label2 = self.dataset[index2]

        apply_mixup = self.mixup > 0
        apply_cutmix = self.cutmix > 0
        ratio = 1

        if apply_mixup and apply_cutmix:
            if index % 2 == 0:
                apply_cutmix = False

            else:
                apply_mixup = False

        if apply_mixup:
            ratio = random.betavariate(self.mixup, self.mixup)
            img1 = Image.blend(img1, img2, 1 - ratio)

        if apply_cutmix:
            if self.cutmix == 1:
                ratio = random.uniform(0, 1)

            else:
                ratio = random.betavariate(self.cutmix, self.cutmix)

            x1, y1, x2, y2 = rand_bbox(img1.size, ratio)
            img1.paste(img2.crop((x1, y1, x2, y2)), (x1, y1, x2, y2))
            ratio = 1 - ((x2 - x1) * (y2 - y1) / (img1.size[0] * img1.size[1]))

        img1 = self.transform(img1)

        return img1, label1, label2, ratio
