#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import math
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.models.layers import lecun_normal_, Mlp
from timm.models.helpers import build_model_with_cfg, named_apply
from timm.models.registry import register_model
from torch import nn

from models.layers import Sequencer2DBlock, PatchEmbed, LSTM2D, GRU2D, RNN2D, Downsample2D


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': DEFAULT_CROP_PCT, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        stdv = 1.0 / math.sqrt(module.hidden_size)
        for weight in module.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_stage(index, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer, rnn_layer, mlp_layer,
              norm_layer, act_layer, num_layers, bidirectional, union,
              with_fc, drop=0., drop_path_rate=0., **kwargs):
    assert len(layers) == len(patch_sizes) == len(embed_dims) == len(hidden_sizes) == len(mlp_ratios)
    blocks = []
    for block_idx in range(layers[index]):
        drop_path = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_layer(embed_dims[index], hidden_sizes[index], mlp_ratio=mlp_ratios[index],
                                  rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer,
                                  act_layer=act_layer, num_layers=num_layers,
                                  bidirectional=bidirectional, union=union, with_fc=with_fc,
                                  drop=drop, drop_path=drop_path))

    if index < len(embed_dims) - 1:
        blocks.append(Downsample2D(embed_dims[index], embed_dims[index + 1], patch_sizes[index + 1]))

    blocks = nn.Sequential(*blocks)
    return blocks


class Sequencer2D(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            layers=[4, 3, 8, 3],
            patch_sizes=[7, 2, 1, 1],
            embed_dims=[192, 384, 384, 384],
            hidden_sizes=[48, 96, 96, 96],
            mlp_ratios=[3.0, 3.0, 3.0, 3.0],
            block_layer=Sequencer2DBlock,
            rnn_layer=LSTM2D,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_rnn_layers=1,
            bidirectional=True,
            union="cat",
            with_fc=True,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dims[0]  # num_features for consistency with other models
        self.embed_dims = embed_dims
        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_sizes[0], in_chans=in_chans,
            embed_dim=embed_dims[0], norm_layer=norm_layer if stem_norm else None,
            flatten=False)

        self.blocks = nn.Sequential(*[
            get_stage(
                i, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer=block_layer,
                rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer,
                num_layers=num_rnn_layers, bidirectional=bidirectional,
                union=union, with_fc=with_fc, drop=drop_rate, drop_path_rate=drop_path_rate,
            )
            for i, _ in enumerate(embed_dims)])

        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    return state_dict


default_cfgs = dict(
    sequencer2d_s=_cfg(url="https://github.com/okojoalg/sequencer/releases/download/weights/sequencer2d_s.pth"),
    sequencer2d_m=_cfg(url="https://github.com/okojoalg/sequencer/releases/download/weights/sequencer2d_m.pth"),
    sequencer2d_l=_cfg(url="https://github.com/okojoalg/sequencer/releases/download/weights/sequencer2d_l.pth"),
    sequencer2d_l_d4_3x=_cfg(),
    sequencer2d_s_unidirectional=_cfg(),
    sequencer2d_s_add=_cfg(),
    sequencer2d_s_h2x=_cfg(),
    sequencer2d_s_without_fc=_cfg(),
    sequencer2d_vertical=_cfg(),
    sequencer2d_s_horizontal=_cfg(),
    gru_sequencer2d_s=_cfg(),
    rnn_sequencer2d_s=_cfg(),
)


def _create_sequencer2d(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Sequencer2D models.')

    model = build_model_with_cfg(
        Sequencer2D, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


# main

@register_model
def sequencer2d_s(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_s', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_m(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 14, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_m', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_l(pretrained=False, **kwargs):
    model_args = dict(
        layers=[8, 8, 16, 4],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_l', pretrained=pretrained, **model_args)
    return model


# high resolution

@register_model
def sequencer2d_s_392(pretrained=False, **kwargs):
    model_args = dict(
        img_size=392,
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_s', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_m_392(pretrained=False, **kwargs):
    model_args = dict(
        img_size=392,
        layers=[4, 3, 14, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_m', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_l_392(pretrained=False, **kwargs):
    model_args = dict(
        img_size=392,
        layers=[8, 8, 16, 4],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_l', pretrained=pretrained, **model_args)
    return model


# ablation

@register_model
def sequencer2d_s_unidirectional(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=False,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_s_unidirectional', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_s_add(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="add",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_s_add', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_s_h2x(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[96, 192, 192, 192],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_s_h2x', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_s_without_fc(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=False,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_s_without_fc', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_vertical(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[96, 192, 192, 192],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="vertical",
        with_fc=False,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_vertical', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_s_horizontal(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[96, 192, 192, 192],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="horizontal",
        with_fc=False,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_s_horizontal', pretrained=pretrained, **model_args)
    return model


# option

@register_model
def gru_sequencer2d_s(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=GRU2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('gru_sequencer2d_s', pretrained=pretrained, **model_args)
    return model


@register_model
def rnn_sequencer2d_s(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[48, 96, 96, 96],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=RNN2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('rnn_sequencer2d_s', pretrained=pretrained, **model_args)
    return model


@register_model
def sequencer2d_l_d4_3x(pretrained=False, **kwargs):
    model_args = dict(
        layers=[8, 8, 16, 4],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[256, 512, 512, 512],
        hidden_sizes=[64, 128, 128, 128],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM2D,
        bidirectional=True,
        union="cat",
        with_fc=True,
        **kwargs)
    model = _create_sequencer2d('sequencer2d_l_d4_3x', pretrained=pretrained, **model_args)
    return model
