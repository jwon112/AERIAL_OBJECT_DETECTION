import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import Conv
from utils import weight_init


class Bottleneck(nn.Module):
    """Multi‑branch bottleneck block with different dilations.

    Parameters
    ----------
    in_dim : int
        Number of input/output channels (block is channel‑preserving).
    dilation : list[int]
        List of 4 dilation rates. Standard YOLOH default `[2,4,6,8]`.
    expand_ratio : float, default 0.25
        Reduction ratio for the inner hidden dimension.
    act_type : str, default "relu"
        Activation used in Conv helper.
    """

    def __init__(
        self,
        in_dim: int,
        dilation: list[int] | tuple[int, int, int, int] = (2, 4, 6, 8),
        expand_ratio: float = 0.25,
        act_type: str = "relu",
    ) -> None:
        super().__init__()
        if len(dilation) != 4:
            raise ValueError("dilation list must have length 4 (got %d)" % len(dilation))

        hid_dim = int(in_dim * expand_ratio)
        self.branch0 = nn.Sequential(
            Conv(in_dim, hid_dim, k=1, act_type=act_type),
            Conv(hid_dim, hid_dim, k=3, p=dilation[0], d=dilation[0], act_type=act_type),
            Conv(hid_dim, in_dim, k=1, act_type=act_type),
        )
        self.branch1 = nn.Sequential(
            Conv(in_dim, hid_dim, k=1, act_type=act_type),
            Conv(hid_dim, hid_dim, k=3, p=dilation[1], d=dilation[1], act_type=act_type),
            Conv(hid_dim, in_dim, k=1, act_type=act_type),
        )
        self.branch2 = nn.Sequential(
            Conv(in_dim, hid_dim, k=1, act_type=act_type),
            Conv(hid_dim, hid_dim, k=3, p=dilation[2], d=dilation[2], act_type=act_type),
            Conv(hid_dim, in_dim, k=1, act_type=act_type),
        )
        self.branch3 = nn.Sequential(
            Conv(in_dim, hid_dim, k=1, act_type=act_type),
            Conv(hid_dim, hid_dim, k=3, p=dilation[3], d=dilation[3], act_type=act_type),
            Conv(hid_dim, in_dim, k=1, act_type=act_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        """Dense residual‑style aggregation (branch‑wise chained)."""
        x1 = self.branch0(x) + x
        x2 = self.branch1(x1 + x) + x1
        x3 = self.branch2(x2 + x1 + x) + x2
        x4 = self.branch3(x3 + x2 + x1 + x) + x3
        return x4


class DilatedEncoder(nn.Module):
    """
    YOLOH Dilated Encoder neck.

    * `projector2_1`  : 1×1 Conv on highest‑level feature (C4).
    * `projector1_1`  : 1×1 Conv on mid‑level feature (C3).
    * `avgpool_2x2`   : 2× downsample of low‑level feature (C2).
    * `projector2`    : 3×3 Conv after fusion.
    * `encoders`      : single multi‑branch Bottleneck (dilated) ― depth can be
                        increased by wrapping Bottleneck in `nn.Sequential` multiple times.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expand_ratio: float = 0.25,
        dilation_list: list[int] | tuple[int, ...] = (2, 4, 6, 8),
        act_type: str = "relu",
    ) -> None:
        super().__init__()

        # 1×1 projectors to unify channel dims
        self.projector2_1 = Conv(in_dim, out_dim, k=1, act_type=None)
        self.projector1_1 = Conv(out_dim * 2, out_dim, k=1, act_type=None)  # C3 has 2×C2 channels for resnet family

        # 3×3 after fusion
        self.projector2 = Conv(out_dim, out_dim, k=3, p=1, act_type=None)

        # pooling for C2 (stride 2)
        self.avgpool_2x2 = nn.AvgPool2d(2, stride=2)

        # single Bottleneck (multi‑dilation). Deeper → use nn.Sequential([Bottleneck,...])
        self.encoders = Bottleneck(
            in_dim=out_dim,
            dilation=dilation_list,
            expand_ratio=expand_ratio,
            act_type=act_type,
        )

        self._init_weight()

    # --------------------------------------------------
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # --------------------------------------------------
    @torch.jit.ignore(if_tracing=True)
    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """`feats` : list[C2,C3,C4]  from backbone.
        Returns fused feature map shape (B,out_dim, H/16, W/16)."""
        # C4 (1/32) → project & upsample to 1/16
        x2 = self.projector2_1(feats[2])            # (B,out,20,20) for 640×640 input
        # C3 (1/16)
        x1 = self.projector1_1(feats[1])            # (B,out,40,40)
        # C2 (1/8) → downsample to 1/16 then use
        x0 = self.avgpool_2x2(feats[0])             # (B,C2/4,40,40) after pool

        # spatial alignment : upsample x2 → x1 size
        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)

        # add & project
        x = x2 + x1 + x0                           # (B,out,40,40)
        x = self.projector2(x)                     # (B,out,40,40)
        x = self.encoders(x)                       # residual dilated blocks
        return x
