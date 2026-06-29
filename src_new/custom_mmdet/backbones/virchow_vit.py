# custom_mmdet/backbones/virchow_vit.py

import contextlib
from typing import Tuple
import torch
import timm
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class VirchowBackbone(BaseModule):
    """
    Virchow backbone wrapper for MMDetection.

    Virchow (paige-ai/Virchow) is a ViT-H/14 histopathology encoder
    (DINOv2-pretrained):

        - patch size  : 14
        - embed_dim   : 1280
        - layers      : 32
        - heads       : 16
        - FFN         : SwiGLU  (act_layer SiLU)  -- MUST be passed to timm
        - LayerScale  : true
        - extra tokens: 1 CLS only (NO register tokens)

    The model card concatenates CLS + mean patch token into a 2560-dim TILE
    embedding for classification. For DENSE DETECTION we instead use the
    spatial PATCH TOKENS (1280-dim) reshaped to a 2D map, dropping the CLS.

    Input:
        - (B, 3, H, W); e.g. (B, 3, 1008, 1008). 1008/14 = 72 -> 72x72 map.

    Output:
        - (B, 1280, H/14, W/14); e.g. (B, 1280, 72, 72), as a tuple (feats,).
    """

    def __init__(
        self,
        model_name: str = "hf-hub:paige-ai/Virchow",
        patch_size: int = 14,
        frozen: bool = True,
        dynamic_img_size: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=None)

        self.model_name = model_name
        self.patch_size = patch_size

        print(f"[VirchowBackbone] Lade Virchow über timm: {model_name}")
        # Virchow REQUIRES the SwiGLU MLP layer + SiLU activation to init
        # correctly (per the model card); timm infers depth/embed_dim/heads.
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=dynamic_img_size,
        )

        self.embed_dim = getattr(
            self.model, "embed_dim",
            getattr(self.model, "num_features", None)
        )
        if self.embed_dim is None:
            raise RuntimeError(
                "[VirchowBackbone] Could not determine embed_dim/num_features."
            )

        print(f"[VirchowBackbone] Architektur bereit: patch_size={self.patch_size}, "
              f"embed_dim={self.embed_dim}")

        self._frozen = frozen
        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            print("[VirchowBackbone] Parameter erfolgreich eingefroren.")

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
                    f"[VirchowBackbone] Input {H}x{W} not divisible by "
                    f"patch_size={self.patch_size}. Use a patch-14-divisible "
                    f"size (e.g. 1008)."
                )

            feats = self.model.forward_features(x)

            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if isinstance(feats, dict):
                feats = feats.get("x", list(feats.values())[-1])

            if hasattr(self.model, "norm") and callable(getattr(self.model, "norm")):
                feats = self.model.norm(feats)

            # feats: (B, N, C) with N = H*W patch tokens + 1 CLS (no registers).
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size
            expected = grid_h * grid_w
            n_tokens = feats.shape[1]
            extra = n_tokens - expected   # == 1 for Virchow (CLS only)

            if extra > 0:
                feats = feats[:, extra:, :]
            elif extra < 0:
                raise ValueError(
                    f"[VirchowBackbone] Too few tokens: tokens={n_tokens} "
                    f"expected={expected} (H={H},W={W},patch={self.patch_size})."
                )

            feats = feats.transpose(1, 2).contiguous().view(
                B, self.embed_dim, grid_h, grid_w
            )

        return (feats,)
