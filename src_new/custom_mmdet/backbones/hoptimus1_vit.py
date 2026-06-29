# custom_mmdet/backbones/hoptimus1_vit.py

import contextlib
import torch
import torch.nn.functional as F
import timm
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class H1Backbone(BaseModule):
    """
    H-Optimus-1 backbone wrapper for MMDetection.

    Input:
        - Image tensor of shape (B, 3, H, W)
        - Example: (B, 3, 1008, 1008)

    Processing:
        - Loads pretrained H-optimus-1 via timm from Hugging Face Hub
        - Applies patch embedding (patch size = 14)
        - Removes extra tokens (e.g. CLS / register tokens)
        - Reshapes ViT tokens from (B, N, C) to a 2D feature map

    Output:
        - Single 2D feature map for detection neck
        - Shape: (B, C, H/14, W/14)
        - Example: (B, 1536, 72, 72)

    Note:
        - H-optimus-1 is the same architecture as H-optimus-0
          (ViT-g/14, embed_dim=1536). Only the pretrained weights and
          normalization stats differ.
        - timm.create_model uses init_values=1e-5 (LayerScale) per the
          H-optimus-1 model card.
        - Output is returned as a tuple: (features,)
          to match MMDetection neck interface.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:bioptimus/H-optimus-1",
        patch_size: int = 14,
        frozen: bool = True,
        auto_pad: bool = True,
        dynamic_img_size: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=None)

        self.model_name = model_name
        self.patch_size = patch_size
        self.auto_pad = auto_pad

        print(f"[H1Backbone] Lade H1 über timm: {model_name}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            init_values=1e-5,
            dynamic_img_size=dynamic_img_size
        )

        self.embed_dim = getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", None))
        if self.embed_dim is None:
            raise RuntimeError("[H1Backbone] Konnte embed_dim/num_features nicht bestimmen.")

        self._frozen = frozen
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            print("[H1Backbone] Modell eingefroren.")

    def init_weights(self):
        pass

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode permanently.

        MMDetection calls .train() on the whole model at training start, which
        would otherwise flip this backbone's dropout / drop_path back to train
        mode. A frozen feature extractor must not do that, so force eval back
        on. Matches UNIBackbone / VirchowBackbone so the FM comparison is fair.
        """
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self

    def _maybe_pad(self, x: torch.Tensor) -> torch.Tensor:
        if not self.auto_pad:
            return x
        _, _, h, w = x.shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    def forward(self, x: torch.Tensor):
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            x = self._maybe_pad(x)
            B, _, H, W = x.shape

            if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
                raise ValueError(f"[H1Backbone] Input {H}x{W} nicht durch patch_size={self.patch_size} teilbar.")

            feats = self.backbone.forward_features(x)

            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if isinstance(feats, dict):
                feats = feats.get("x", list(feats.values())[-1])

            if not torch.is_tensor(feats):
                raise TypeError(f"[H1Backbone] forward_features returned {type(feats)} statt Tensor.")

            if hasattr(self.backbone, "norm") and callable(getattr(self.backbone, "norm")):
                feats = self.backbone.norm(feats)

            if feats.dim() == 3:
                grid_h = H // self.patch_size
                grid_w = W // self.patch_size
                expected = grid_h * grid_w

                n_tokens = feats.shape[1]
                extra = n_tokens - expected

                if extra > 0:
                    feats = feats[:, extra:, :]
                elif extra < 0:
                    raise ValueError(
                        f"[H1Backbone] Zu wenige Tokens: tokens={n_tokens} expected={expected} "
                        f"(H={H},W={W},patch={self.patch_size})."
                    )

                # Reshape zu [B, C, H, W]
                feats = feats.transpose(1, 2).contiguous().view(B, self.embed_dim, grid_h, grid_w)

        return (feats,)
