# src/custom_mmdet/transforms/hed_stain_augment.py
"""
HED stain augmentation for H&E histopathology, as an MMDetection transform.

Implements the stain-jitter scheme of Tellez et al. (2018/2019). RGB H&E images
are deconvolved into Hematoxylin / Eosin / residual channels (Ruifrok &
Johnston colour deconvolution), the H and E channels are perturbed with small
random multiplicative (alpha) and additive (beta) factors, then reconstructed
to RGB. This mimics the stain-intensity variation across scanners/labs in
MIDOG++, the dominant domain shift.

NOTE: the H-optimus backbone is FROZEN and was pretrained on a specific colour
distribution. Keep the jitter MODERATE - large factors push images off that
distribution and can hurt frozen features. sigma/bias are hyperparameters.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

# Ruifrok-Johnston H&E stain matrix (rows = H, E, residual) and its inverse.
_HED_FROM_RGB = np.array([
    [1.87798274, -1.00767869, -0.55611582],
    [-0.06590806, 1.13473037, -0.1355218],
    [-0.60190736, -0.48041419, 1.57358807],
])
_RGB_FROM_HED = np.linalg.inv(_HED_FROM_RGB)


@TRANSFORMS.register_module()
class HEDStainAugment(BaseTransform):
    """Random HED stain jitter.

    Args:
        sigma (float): std of multiplicative factor alpha ~ U(1-sigma, 1+sigma),
            applied to H and E channels.
        bias (float): range of additive factor beta ~ U(-bias, bias),
            applied to H and E channels.
        prob (float): probability of applying the augmentation per image.
        jitter_residual (bool): also jitter the 3rd (residual) channel.

    Required keys:  img  (uint8 RGB, HWC)
    Modified keys:  img
    """

    def __init__(self, sigma=0.05, bias=0.02, prob=0.5, jitter_residual=False):
        assert 0.0 <= prob <= 1.0
        self.sigma = float(sigma)
        self.bias = float(bias)
        self.prob = float(prob)
        self.jitter_residual = bool(jitter_residual)

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        orig_dtype = img.dtype

        rgb = img.astype(np.float64) / 255.0
        rgb = np.clip(rgb, 1e-6, 1.0)            # avoid log(0)
        od = -np.log10(rgb)                       # optical density
        hed = od @ _HED_FROM_RGB.T                # deconvolve to HED

        n_ch = 3 if self.jitter_residual else 2
        alpha = np.random.uniform(1 - self.sigma, 1 + self.sigma, size=n_ch)
        beta = np.random.uniform(-self.bias, self.bias, size=n_ch)
        hed[..., :n_ch] = hed[..., :n_ch] * alpha + beta

        od_aug = hed @ _RGB_FROM_HED.T            # back to OD
        rgb_aug = np.power(10.0, -od_aug)         # OD -> intensity
        rgb_aug = np.clip(rgb_aug, 0.0, 1.0)
        img_aug = (rgb_aug * 255.0).astype(orig_dtype)

        results['img'] = img_aug
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(sigma={self.sigma}, '
                f'bias={self.bias}, prob={self.prob}, '
                f'jitter_residual={self.jitter_residual})')
