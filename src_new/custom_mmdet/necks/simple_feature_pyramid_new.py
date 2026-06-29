# custom_mmdet/necks/simple_feature_pyramid.py

# Adapted from Detectron2 ViTDet SimpleFeaturePyramid
# https://github.com/facebookresearch/detectron2
# Link:
#   https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class SimpleFeaturePyramid(BaseModule):
    """
    SimpleFeaturePyramid (ViT-style).

    Input:
        - Single 2D ViT feature map (token map), shape (B, C, H, W)
                B -> batch size
                C -> channels (e.g. 1536 for H-Optimus-0)
                H -> feature-map height
                W -> feature-map width

        - Example: (1, 1536, 72, 72) for ViT-14 with 1008x1008 input
          (physical stride of this input map = 14, the patch size).

    Output (MULTI-LEVEL, default):
        - Multi-scale feature pyramid with len(scale_factors) levels.
        - With scale_factors=(2.0, 1.0, 0.5, 0.25, 0.125) and a stride-14
          input, the PHYSICAL strides of the outputs are:
                14/2=7, 14, 28, 56, 112
          (NOT 8/16/32/64/128 - the detector head strides must be set to
           match these physical values, or norm_on_bbox / point coords are
           wrong by ~14%.)

    Output (SINGLE-LEVEL, single_level is not None):
        - Exactly ONE feature map at the chosen scale_factor.
        - Use this for fixed-size / monodisperse objects (MIDOG++ mitoses are
          ~50 px boxes), where FCOS-style per-level size binning would
          otherwise dump 100% of GT onto one level and leave the other levels
          training on background only.

    Notes:
        - input_stride is the physical stride of the incoming token map
          (= ViT patch size, 14 for H-Optimus-0 / UNI). It is used to report
          the resulting physical output strides via `out_strides`, which the
          training script can cross-check against the head config.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        scale_factors=(2.0, 1.0, 0.5, 0.25, 0.125),
        norm: str = "LN",
        input_stride: int = 14,
        single_level=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_stride = input_stride

        # Single-level mode: keep only one scale_factor.
        if single_level is not None:
            if not (0 <= single_level < len(scale_factors)):
                raise ValueError(
                    f"single_level={single_level} out of range for "
                    f"scale_factors of length {len(scale_factors)}."
                )
            scale_factors = (scale_factors[single_level],)

        self.single_level = single_level
        self.scale_factors = scale_factors

        # Physical output strides, for cross-checking against head strides.
        self.out_strides = [int(round(self.input_stride / s))
                            for s in self.scale_factors]

        self.stages = nn.ModuleList()
        for scale in scale_factors:
            self.stages.append(
                self._make_stage(scale, in_channels, out_channels, norm)
            )

    def _norm(self, norm: str, num_channels: int):
        if norm == "" or norm is None:
            return nn.Identity()
        if norm.upper() == "LN":
            return nn.GroupNorm(1, num_channels)
        if norm.upper() == "BN":
            return nn.BatchNorm2d(num_channels)
        raise ValueError(f"Unsupported norm: {norm}")

    def _make_stage(self, scale: float, in_ch: int, out_ch: int, norm: str):
        layers = []

        # --- Resize part (Detectron2 logic) ---
        if scale == 4.0:
            layers += [
                nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
                self._norm(norm, in_ch // 2),
                nn.GELU(),
                nn.ConvTranspose2d(in_ch // 2, in_ch // 4, kernel_size=2, stride=2),
            ]
            cur_ch = in_ch // 4
        elif scale == 2.0:
            layers += [nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)]
            cur_ch = in_ch // 2
        elif scale == 1.0:
            cur_ch = in_ch
        elif scale == 0.5:
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            cur_ch = in_ch
        elif scale == 0.25:
            layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
            cur_ch = in_ch
        elif scale == 0.125:
            layers += [nn.MaxPool2d(kernel_size=8, stride=8)]
            cur_ch = in_ch
        else:
            raise NotImplementedError(f"scale_factor={scale} not supported.")

        layers += [
            nn.Conv2d(cur_ch, out_ch, kernel_size=1, bias=(norm == "")),
            self._norm(norm, out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=(norm == "")),
            self._norm(norm, out_ch),
        ]

        return nn.Sequential(*layers)

    def forward(self, inputs):
        """
        inputs: tuple/list of feature maps from backbone.
        We expect a single level: inputs = (x,)
        """
        assert isinstance(inputs, (tuple, list)) and len(inputs) == 1
        x = inputs[0]  # (B, C, H, W)

        outs = [stage(x) for stage in self.stages]
        return tuple(outs)
