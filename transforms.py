import random

import torch
from PIL import Image, ImageOps, ImageEnhance

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
