import random
import math

import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision import transforms

PIL_INTER_MAP = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}

IMAGENET_EIGVAL = (0.2175, 0.0188, 0.0045)
IMAGENET_EIGVEC = (
    (-0.5675, 0.7192, 0.4009),
    (-0.5808, -0.0045, -0.8140),
    (-0.5836, -0.6948, 0.4203),
)


def check_prob(p):
    return p == 1.0 or random.random() < p


class RandomTransform:
    def __init__(self, p):
        self.p = p

    def apply_img(self, img, **params):
        if not check_prob(self.p):
            return img

        return self._apply_img(img, **params)

    def apply_img_check(self, img, **params):
        if not check_prob(self.p):
            return img, False

        return self._apply_img(img, **params), True

    def _repr_params(self):
        params = dict(self.__dict__)

        return params

    def __call__(self, img):
        sample = self.sample()
        img = self.apply_img(img, **sample)

        return img

    def __repr__(self):
        params = []

        for k, v in self._repr_params().items():
            params.append(f"{k}={v}")

        param_str = ", ".join(params)
        repr_str = f"{self.__class__.__name__}({param_str})"

        return repr_str


class Lighting(RandomTransform):
    def __init__(
        self, alpha_std, eigval=IMAGENET_EIGVAL, eigvec=IMAGENET_EIGVEC, p=1.0
    ):
        super().__init__(p)

        self.alpha_std = alpha_std
        self.eigval = torch.as_tensor(eigval)
        self.eigvec = torch.as_tensor(eigvec)

    def __call__(self, img):
        alpha = img.new_empty(3).normal_(0, self.alpha_std)
        rgb = (
            self.eigvec.to(img)
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img + rgb.view(3, 1, 1)


class Affine(RandomTransform):
    def __init__(self, degrees, translate, scale, shear, p=1.0):
        super().__init__(p)

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def _apply_img(self, img, degrees, translate, scale, shear):
        pass


class Posterize(RandomTransform):
    def __init__(self, bits, p=1.0):
        super().__init__(p)

        self.bits = int(bits)

    def sample(self):
        return {"bits": self.bits}

    def _apply_img(self, img, bits):
        return ImageOps.posterize(img, bits)


class Invert(RandomTransform):
    def __init__(self, p):
        super().__init__(p)

    def sample(self):
        return {}

    def _apply_img(self, img):
        return ImageOps.invert(img)


class AutoContrast(RandomTransform):
    def __init__(self, p):
        super().__init__(p)

    def sample(self):
        return {}

    def _apply_img(self, img):
        return ImageOps.autocontrast(img)


class Equalize(RandomTransform):
    def __init__(self, p):
        super().__init__(p)

    def sample(self):
        return {}

    def _apply_img(self, img):
        return ImageOps.equalize(img)


class Solarize(RandomTransform):
    def __init__(self, threshold, p=1.0):
        super().__init__(p)

        self.threshold = int(threshold)

    def sample(self):
        return {"threshold": self.threshold}

    def _apply_img(self, img, threshold):
        return ImageOps.solarize(img, threshold)


class Saturation(RandomTransform):
    def __init__(self, saturation, p=1.0):
        super().__init__(p)

        self.saturation = saturation

    def sample(self):
        return {"saturation": self.saturation}

    def _apply_img(self, img, saturation):
        return ImageEnhance.Color(img).enhance(saturation)


class Contrast(RandomTransform):
    def __init__(self, contrast, p=1.0):
        super().__init__(p)

        self.contrast = contrast

    def sample(self):
        return {"contrast": self.contrast}

    def _apply_img(self, img, contrast):
        return ImageEnhance.Contrast(img).enhance(contrast)


class Brightness(RandomTransform):
    def __init__(self, brightness, p=1.0):
        super().__init__(p)

        self.brightness = brightness

    def sample(self):
        return {"brightness": self.brightness}

    def _apply_img(self, img, brightness):
        return ImageEnhance.Brightness(img).enhance(brightness)


class GaussianBlur(RandomTransform):
    def __init__(self, radius_min=0.1, radius_max=2, p=0.5):
        super().__init__(p)

        self.radius_min = radius_min
        self.radius_max = radius_max

    def sample(self):
        return {"radius": random.uniform(self.radius_min, self.radius_max)}

    def _apply_img(self, img, radius):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class DINOAugment:
    def __init__(
        self,
        global_crop_size,
        local_crop_size,
        global_crop_scale,
        local_crop_scale,
        n_local_crop,
    ):
        flip_color = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crop_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_color,
                GaussianBlur(p=1.0),
                normalize,
            ]
        )

        self.global_transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crop_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_color,
                GaussianBlur(p=0.1),
                Solarize(threshold=128, p=0.2),
                normalize,
            ]
        )

        self.n_local_crop = n_local_crop
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crop_size, scale=local_crop_scale, interpolation=Image.BICUBIC
                ),
                flip_color,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))

        for _ in range(self.n_local_crop):
            crops.append(self.local_transform(image))

        return crops


# Adapted from timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py

""" Random Erasing (Cutout)

Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman
"""


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
        self,
        p=0.5,
        min_area=0.02,
        max_area=1 / 3,
        min_aspect=0.3,
        max_aspect=None,
        mode="const",
        min_count=1,
        max_count=None,
        num_splits=0,
        device="cuda",
    ):
        self.probability = p
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
        self.device = device
        self.mode = mode

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(p={self.probability}, mode="{self.mode}", max_count={self.max_count},'
            f" num_splits={self.num_splits})"
        )

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for attempt in range(10):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input
