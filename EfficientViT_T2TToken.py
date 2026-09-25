# --------------------------------------------------------
# EfficientViT + Lightweight T2T-ViT Token Construction
#
# It replaces the dense patch embed and both inter-stage PatchMerging
# modules with lightweight T2T-style token aggregation. Those modules are new,
# so they cannot inherit EfficientViT weights; pass a checkpoint path to load
# ImageNet weights for the ~90% of the backbone that is unchanged.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.backbone.efficientViT import (
    EfficientViT,
    EfficientViTBlock,
    EfficientViT_m0,
    EfficientViT_m1,
    EfficientViT_m2,
    EfficientViT_m3,
    EfficientViT_m4,
    EfficientViT_m5,
    PatchMerging,
    replace_batchnorm,
    update_weight,
)

__all__ = [
    'LightT2TStem',
    'T2TPatchMerging',
    'EfficientViT_T2TToken',
    'EfficientViT_T2TToken_M0',
    'EfficientViT_T2TToken_M1',
    'EfficientViT_T2TToken_M2',
    'EfficientViT_T2TToken_M3',
    'EfficientViT_T2TToken_M4',
    'EfficientViT_T2TToken_M5',
]


class T2TPatchMerging(nn.Module):
    """Rearrange 2x2 neighborhoods into channels, then aggregate with 1x1 conv."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim * 4)
        self.proj = nn.Conv2d(dim * 4, out_dim, 1, bias=False)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        return self.act(self.proj(self.norm(x)))


class LightT2TStem(nn.Module):
    """Lightweight progressive token stem with T2T-style restructure downsampling.

    The first two stages follow the dense 7x7 soft split and local DWConv
    design. Additional 2x2 neighborhood merges are appended until the output
    reaches the same H/8 scale as EfficientViT's original patch embed.
    """

    def __init__(self, in_chans=3, embed_dim=64, output_scale=4):
        super().__init__()
        if output_scale < 2 or (output_scale & (output_scale - 1)):
            raise ValueError('output_scale must be a power of two >= 2')

        self.output_scale = output_scale
        mid_dim = max(embed_dim // 2, 8)

        self.conv1 = nn.Conv2d(in_chans, mid_dim, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.dwconv = nn.Conv2d(
            mid_dim, mid_dim, 3, 1, 1, groups=mid_dim, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.downsample = nn.Conv2d(mid_dim, embed_dim, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.act = nn.Hardswish(inplace=True)

        self.extra_merges = nn.ModuleList()
        scale = 2
        while scale < output_scale:
            self.extra_merges.append(T2TPatchMerging(embed_dim, embed_dim))
            scale *= 2

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.dwconv(x)))
        x = self.act(self.bn3(self.downsample(x)))
        for merge in self.extra_merges:
            x = merge(x)
        return x


class EfficientViT_T2TToken(EfficientViT):
    """EfficientViT with T2T tokenization in the stem and downsample stages."""

    def __init__(
        self,
        stem_embed_dim=None,
        t2t_stem_scale=8,
        **kwargs,
    ):
        self.in_chans = kwargs.get('in_chans', 3)
        super().__init__(**kwargs)

        if stem_embed_dim is None:
            for child in self.blocks1:
                if isinstance(child, EfficientViTBlock):
                    stem_embed_dim = child.dw0.m.c.in_channels
                    break
        if stem_embed_dim is None:
            raise ValueError('Unable to infer EfficientViT stem embed dimension')

        self.patch_embed = LightT2TStem(
            in_chans=self.in_chans,
            embed_dim=stem_embed_dim,
            output_scale=t2t_stem_scale,
        )

        for stage_blocks in (self.blocks2, self.blocks3):
            for index, module in enumerate(stage_blocks):
                if isinstance(module, PatchMerging):
                    in_dim = module.conv1.c.in_channels
                    out_dim = module.conv3.c.out_channels
                    stage_blocks[index] = T2TPatchMerging(in_dim, out_dim)


def _build(
    model_cfg,
    pretrained='',
    frozen_stages=0,
    distillation=False,
    fuse=False,
    t2t_stem_scale=8,
):
    model = EfficientViT_T2TToken(
        frozen_stages=frozen_stages,
        distillation=distillation,
        stem_embed_dim=model_cfg['embed_dim'][0],
        t2t_stem_scale=t2t_stem_scale,
        **model_cfg,
    )
    if pretrained:
        # See EfficientViT_T2TToken_norm._build: the replaced modules simply do
        # not match by name, so ImageNet weights load for the rest of the backbone
        # (624/666 tensors, 90.2% of the parameters for EfficientViT_T2TToken_M0).
        ckpt = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(
            update_weight(model.state_dict(), ckpt.get('model', ckpt)))
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_T2TToken_M0(
    pretrained='', frozen_stages=0, distillation=False, fuse=False,
    t2t_stem_scale=8,
):
    return _build(EfficientViT_m0, pretrained, frozen_stages, distillation, fuse,
                  t2t_stem_scale)


def EfficientViT_T2TToken_M1(
    pretrained='', frozen_stages=0, distillation=False, fuse=False,
    t2t_stem_scale=8,
):
    return _build(EfficientViT_m1, pretrained, frozen_stages, distillation, fuse,
                  t2t_stem_scale)


def EfficientViT_T2TToken_M2(
    pretrained='', frozen_stages=0, distillation=False, fuse=False,
    t2t_stem_scale=8,
):
    return _build(EfficientViT_m2, pretrained, frozen_stages, distillation, fuse,
                  t2t_stem_scale)


def EfficientViT_T2TToken_M3(
    pretrained='', frozen_stages=0, distillation=False, fuse=False,
    t2t_stem_scale=8,
):
    return _build(EfficientViT_m3, pretrained, frozen_stages, distillation, fuse,
                  t2t_stem_scale)


def EfficientViT_T2TToken_M4(
    pretrained='', frozen_stages=0, distillation=False, fuse=False,
    t2t_stem_scale=8,
):
    return _build(EfficientViT_m4, pretrained, frozen_stages, distillation, fuse,
                  t2t_stem_scale)


def EfficientViT_T2TToken_M5(
    pretrained='', frozen_stages=0, distillation=False, fuse=False,
    t2t_stem_scale=8,
):
    return _build(EfficientViT_m5, pretrained, frozen_stages, distillation, fuse,
                  t2t_stem_scale)


if __name__ == '__main__':
    model = EfficientViT_T2TToken_M0()
    inputs = torch.randn(1, 3, 640, 640)
    outputs = model(inputs)
    print([tuple(x.shape) for x in outputs])

