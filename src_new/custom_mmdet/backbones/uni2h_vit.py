# custom_mmdet/backbones/uni2h_vit.py

import contextlib
from typing import Tuple
import torch
import torch.nn.functional as F
import timm
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class UNI2hBackbone(BaseModule):
    """
    UNI2-h backbone wrapper for MMDetection.

    UNI2-h is a CUSTOM ViT-H/14 (DINOv2) histopathology encoder
    (MahmoodLab/UNI2-h). It differs from UNI (ViT-L/16) in several ways that
    matter for this wrapper:

        - patch size      : 14   (UNI was 16)
        - embed_dim       : 1536 (UNI was 1024)
        - depth           : 24
        - num_heads       : 24
        - FFN             : SwiGLUPacked, SiLU activation, mlp_ratio 2.66667*2
        - reg_tokens      : 8   (UNI had 0)
        - no_embed_class  : True
        - init_values     : 1e-5 (LayerScale)

    These are all REQUIRED to construct the architecture correctly; timm cannot
    infer them from the hub name alone (it is a custom ViT). See the UNI2-h
    model card.

    Input:
        - Image tensor (B, 3, H, W); e.g. (B, 3, 1008, 1008).
          1008/14 = 72 -> 72x72 token map (use a patch-14-divisible size).

    Output:
        - Single 2D feature map (B, 1536, H/14, W/14); e.g. (B, 1536, 72, 72).
        - Returned as a tuple (features,) for the MMDetection neck interface.

    Token handling:
        - UNI2-h emits CLS + 8 register tokens in addition to the patch tokens.
          We keep ONLY the trailing H*W patch tokens (drop the leading 9 extra
          tokens) before reshaping to a 2D map. The code computes the number of
          extra tokens as (N - H*W) and strips them generically, so it is
          robust whether the registers come before or after the patches.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:MahmoodLab/UNI2-h",
        patch_size: int = 14,
        frozen: bool = True,
        init_values: float = 1e-5,
        dynamic_img_size: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=None)

        self.model_name = model_name
        self.patch_size = patch_size

        # Full architecture spec required by the UNI2-h model card.
        timm_kwargs = dict(
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=init_values,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=dynamic_img_size,
        )

        print(f"[UNI2hBackbone] Lade UNI2-h über timm: {model_name}")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            **timm_kwargs,
        )

        self.embed_dim = getattr(
            self.model, "embed_dim",
            getattr(self.model, "num_features", None)
        )
        if self.embed_dim is None:
            raise RuntimeError(
                "[UNI2hBackbone] Could not determine embed_dim/num_features."
            )

        print(f"[UNI2hBackbone] Architektur bereit: patch_size={self.patch_size}, "
              f"embed_dim={self.embed_dim}")

        self._frozen = frozen
        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            print("[UNI2hBackbone] Parameter erfolgreich eingefroren.")

    def init_weights(self):
        pass

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode permanently (see UNIBackbone)."""
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            B, _, H, W = x.shape
            if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
                raise ValueError(
                    f"[UNI2hBackbone] Input {H}x{W} not divisible by "
                    f"patch_size={self.patch_size}. Use a patch-14-divisible "
                    f"input size (e.g. 1008)."
                )

            feats = self.model.forward_features(x)

            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if isinstance(feats, dict):
                feats = feats.get("x", list(feats.values())[-1])

            if hasattr(self.model, "norm") and callable(getattr(self.model, "norm")):
                feats = self.model.norm(feats)

            # feats: (B, N, C) with N = H*W patch tokens + extra (CLS + 8 reg).
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size
            expected = grid_h * grid_w
            n_tokens = feats.shape[1]
            extra = n_tokens - expected

            if extra > 0:
                # Drop the leading extra tokens (CLS + registers).
                feats = feats[:, extra:, :]
            elif extra < 0:
                raise ValueError(
                    f"[UNI2hBackbone] Too few tokens: tokens={n_tokens} "
                    f"expected={expected} (H={H},W={W},patch={self.patch_size})."
                )

            feats = feats.transpose(1, 2).contiguous().view(
                B, self.embed_dim, grid_h, grid_w
            )

        return (feats,)
