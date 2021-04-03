import random

from PIL import Image, ImageOps, ImageEnhance, ImageDraw

import transforms
from transforms import check_prob, PIL_INTER_MAP, RandomTransform


def rescale_float(level, max_val, param_max=10):
    return float(level) * max_val / param_max


def rescale_int(level, max_val, param_max=10):
    return int(level * max_val / param_max)


class AutoAugmentAffine(RandomTransform):
    def __init__(self, mirror=True, resample=Image.NEAREST, fillcolor=None, p=1.0):
        super().__init__(p)

        self.mirror = mirror
        self.resample = resample
        self.fillcolor = fillcolor

    def _mirror(self, val):
        if self.mirror and check_prob(0.5):
            val *= -1

        return val

    def _repr_params(self):
        params = dict(self.__dict__)
        params["resample"] = PIL_INTER_MAP[self.resample]

        return params

    def _apply_img_fn(self, img, translate, shear):
        trans_x, trans_y = translate
        shear_x, shear_y = shear

        return img.transform(
            img.size,
            Image.AFFINE,
            (1, shear_x, trans_x, shear_y, 1, trans_y),
            self.resample,
            fillcolor=self.fillcolor,
        )


class ShearX(AutoAugmentAffine):
    def __init__(
        self, shear_x, mirror=True, resample=Image.NEAREST, fillcolor=None, p=1.0
    ):
        super().__init__(mirror=mirror, resample=resample, fillcolor=fillcolor, p=p)

        self.shear_x = shear_x

    def sample(self):
        shear_x = self._mirror(self.shear_x)

        return {"shear_x": shear_x}

    def _apply_img(self, img, shear_x):
        return self._apply_img_fn(img, (0, 0), (shear_x, 0))


class ShearY(AutoAugmentAffine):
    def __init__(
        self, shear_y, mirror=True, resample=Image.NEAREST, fillcolor=None, p=1.0
    ):
        super().__init__(mirror=mirror, resample=resample, fillcolor=fillcolor, p=p)

        self.shear_y = shear_y

    def sample(self):
        shear_y = self._mirror(self.shear_y)

        return {"shear_y": shear_y}

    def _apply_img(self, img, shear_y):
        return self._apply_img_fn(img, (0, 0), (0, shear_y))


class TranslateX(AutoAugmentAffine):
    def __init__(
        self, translate_x, mirror=True, resample=Image.NEAREST, fillcolor=None, p=1.0
    ):
        super().__init__(mirror=mirror, resample=resample, fillcolor=fillcolor, p=p)

        self.translate_x = translate_x

    def sample(self):
        trans_x = self._mirror(self.translate_x)

        return {"translate_x": trans_x}

    def _apply_img(self, img, translate_x):
        return self._apply_img_fn(img, (translate_x, 0), (0, 0))


class TranslateY(AutoAugmentAffine):
    def __init__(
        self, translate_y, mirror=True, resample=Image.NEAREST, fillcolor=None, p=1.0
    ):
        super().__init__(mirror=mirror, resample=resample, fillcolor=fillcolor, p=p)

        self.translate_y = translate_y

    def sample(self):
        trans_y = self._mirror(self.translate_y)

        return {"translate_y": trans_y}

    def _apply_img(self, img, translate_y):
        return self._apply_img_fn(img, (0, translate_y), (0, 0))


class Rotate(AutoAugmentAffine):
    def __init__(
        self, rotate, mirror=True, resample=Image.NEAREST, fillcolor=None, p=1.0
    ):
        super().__init__(mirror=mirror, resample=resample, fillcolor=fillcolor, p=p)

        self.rotate = rotate

    def sample(self):
        rotate = self._mirror(self.rotate)

        return {"rotate": rotate}

    def _apply_img(self, img, rotate):
        return img.rotate(rotate, resample=self.resample, fillcolor=self.fillcolor)


class Posterize(RandomTransform):
    def __init__(self, bits, p=1.0):
        super().__init__(p)

        self.bits = bits

    def sample(self):
        return {"bits": self.bits}

    def _apply_img(self, img, bits):
        return ImageOps.posterize(img, bits)


class Cutout(RandomTransform):
    def __init__(self, size, fillcolor=(0, 0, 0), p=1.0):
        super().__init__(p)

        self.size = size
        self.fillcolor = fillcolor

    def sample(self):
        x = random.random()
        y = random.random()

        return {"center": (x, y)}

    def _apply_img(self, img, center):
        w, h = img.size
        c_x = int(center[0] * w)
        c_y = int(center[1] * h)

        x0 = max(0, c_x - self.size)
        x1 = w - max(0, w - c_x - self.size) - 1
        y0 = max(0, c_y - self.size)
        y1 = h - max(0, h - c_y - self.size) - 1

        xy = (x0, y0, x1, y1)
        img = img.copy()
        ImageDraw.Draw(img).rectangle(xy, self.fillcolor)

        return img


def solarize_add(img, add, threshold=128):
    lut = []

    for i in range(256):
        if i < threshold:
            lut.append(min(255, i + add))

        else:
            lut.append(i)

    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut

        return img.point(lut)

    else:
        return img


class Solarize(RandomTransform):
    def __init__(self, threshold, p=1.0):
        super().__init__(p)

        self.threshold = threshold

    def sample(self):
        return {"threshold": self.threshold}

    def _apply_img(self, img, threshold):
        return ImageOps.solarize(img, threshold)


class SolarizeAdd(RandomTransform):
    def __init__(self, add, threshold=128, p=1.0):
        super().__init__(p)

        self.add = add
        self.threshold = threshold

    def sample(self):
        return {"add": self.add, "threshold": self.threshold}

    def _apply_img(self, img, add, threshold):
        return solarize_add(img, add, threshold)


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


class Sharpness(RandomTransform):
    def __init__(self, sharpness, p=1.0):
        super().__init__(p)

        self.sharpness = sharpness

    def sample(self):
        return {"sharpness": self.sharpness}

    def _apply_img(self, img, sharpness):
        return ImageEnhance.Sharpness(img).enhance(sharpness)


AUTOAUGMENT_MAP = {
    "ShearX": (ShearX, lambda level: rescale_float(level, 0.3)),
    "ShearY": (ShearY, lambda level: rescale_float(level, 0.3)),
    "TranslateX": (
        TranslateX,
        lambda level, max_translate: rescale_int(level, max_translate),
    ),
    "TranslateY": (
        TranslateY,
        lambda level, max_translate: rescale_int(level, max_translate),
    ),
    "Rotate": (Rotate, lambda level: rescale_int(level, 30)),
    "Solarize": (Solarize, lambda level: rescale_int(level, 256)),
    "SolarizeIncreasing": (Solarize, lambda level: 256 - rescale_int(level, 256)),
    "Posterize": (Posterize, lambda level: rescale_int(level, 4)),
    "PosterizeIncreasing": (Posterize, lambda level: 4 - rescale_int(level, 4)),
    "Contrast": (Contrast, lambda level: rescale_float(level, 1.8) + 0.1),
    "Color": (Saturation, lambda level: rescale_float(level, 1.8) + 0.1),
    "Brightness": (Brightness, lambda level: rescale_float(level, 1.8) + 0.1),
    "Sharpness": (Sharpness, lambda level: rescale_float(level, 1.8) + 0.1),
    "Invert": (transforms.Invert, None),
    "AutoContrast": (transforms.AutoContrast, None),
    "Equalize": (transforms.Equalize, None),
    "Cutout": (Cutout, lambda level, cutout: rescale_int(level, cutout)),
    "SolarizeAdd": (SolarizeAdd, lambda level: rescale_int(level, 110)),
}


def autoaugment_policy():
    policy_list = [
        [("PosterizeIncreasing", 0.4, 8), ("Rotate", 0.6, 9)],
        [("SolarizeIncreasing", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
        [("PosterizeIncreasing", 0.6, 7), ("PosterizeIncreasing", 0.6, 6)],
        [("Equalize", 0.4, 7), ("SolarizeIncreasing", 0.2, 4)],
        [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
        [("SolarizeIncreasing", 0.6, 3), ("Equalize", 0.6, 7)],
        [("PosterizeIncreasing", 0.8, 5), ("Equalize", 1.0, 2)],
        [("Rotate", 0.2, 3), ("SolarizeIncreasing", 0.6, 8)],
        [("Equalize", 0.6, 8), ("PosterizeIncreasing", 0.4, 6)],
        [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
        [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
        [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Rotate", 0.8, 8), ("Color", 1.0, 0)],
        [("Color", 0.8, 8), ("SolarizeIncreasing", 0.8, 7)],
        [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
        [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
        [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
        [("Equalize", 0.4, 7), ("SolarizeIncreasing", 0.2, 4)],
        [("SolarizeIncreasing", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    ]

    reparam_policy = []

    for policy in policy_list:
        sub_pol = []

        for pol in policy:
            augment, prob, magnitude = pol
            augment_fn, reparam_fn = AUTOAUGMENT_MAP[augment]

            if reparam_fn is not None:
                magnitude = reparam_fn(magnitude)
                sub_pol.append(augment_fn(magnitude, p=prob))

            else:
                sub_pol.append(augment_fn(p=prob))

        reparam_policy.append(sub_pol)

    return reparam_policy


class AutoAugment:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        selected_policy = random.choice(self.policy)

        for pol in selected_policy:
            sample = pol.sample()
            img = pol.apply_img(img, **sample)

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.policy}\n)"

    def check(self, img):
        log = []

        selected_policy = random.choice(self.policy)

        for pol in selected_policy:
            sample = pol.sample()
            img, check = pol.apply_img_check(img, **sample)
            log.append((pol, sample, check))

        return img, log


class RandAugment:
    def __init__(
        self, n_augment, magnitude, translate=100, cutout=40, fillcolor=(128, 128, 128)
    ):
        self.n_augment = n_augment
        self.magnitude = magnitude
        self.translate = translate
        self.fillcolor = fillcolor

        # fmt: off
        augment_list = [
            "AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize",
            "Color", "Contrast", "Brightness", "Sharpness", "ShearX",
            "ShearY", "TranslateX", "TranslateY", "Cutout", "SolarizeAdd",
        ]
        # fmt: on
        self.augment = []
        for augment in augment_list:
            augment_fn, reparam_fn = AUTOAUGMENT_MAP[augment]

            if reparam_fn is not None:
                if augment in ("TranslateX", "TranslateY"):
                    mag = reparam_fn(magnitude, translate)

                elif augment == "Cutout":
                    mag = reparam_fn(magnitude, cutout)

                else:
                    mag = reparam_fn(magnitude)

                if augment in (
                    "TranslateX",
                    "TranslateY",
                    "ShearX",
                    "ShearY",
                    "Rotate",
                    "Cutout",
                ):
                    augment_inst = augment_fn(mag, fillcolor=fillcolor, p=1.0)

                else:
                    augment_inst = augment_fn(mag, p=1.0)

            else:
                augment_inst = augment_fn(p=1.0)

            self.augment.append(augment_inst)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_augment={self.n_augment}, magnitude={self.magnitude})"

    def __call__(self, img):
        augments = random.choices(self.augment, k=self.n_augment)

        for augment in augments:
            sample = augment.sample()
            img = augment.apply_img(img, **sample)

        return img
