# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
from iopath.common.file_io import g_pathmgr
from mmengine.logging import print_log

from sam3.device_utils import get_default_device
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.vitdet import ViT


def _create_vision_backbone(compile_mode=None, use_rope_real=False):
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    vit_backbone = ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_rope_real=use_rope_real,
    )
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=False,
    )


def build_sam3_vision_encoder(
    checkpoint_path,
    device=None,
    eval_mode=True,
    compile=False,
    use_rope_real=None,
):
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for the SAM3 vision encoder")
    if not g_pathmgr.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM3 checkpoint does not exist: {checkpoint_path}")

    device = get_default_device() if device is None else torch.device(device)
    if use_rope_real is None:
        use_rope_real = device.type == "npu"
    print_log(
        f"[SAM3] Loading vision encoder checkpoint: {checkpoint_path}",
        logger="current",
    )

    encoder = _create_vision_backbone(
        compile_mode="default" if compile else None,
        use_rope_real=use_rope_real,
    )
    with g_pathmgr.open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=True)
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]

    encoder_keys = set(encoder.state_dict())
    encoder_checkpoint = {}
    prefixes = (
        "detector.backbone.vision_backbone.",
        "backbone.vision_backbone.",
        "vision_backbone.",
    )
    for key, value in checkpoint.items():
        if key in encoder_keys:
            encoder_checkpoint[key] = value
            continue
        for prefix in prefixes:
            if prefix in key:
                encoder_key = key.split(prefix, 1)[1]
                if encoder_key in encoder_keys:
                    encoder_checkpoint[encoder_key] = value
                break

    if not encoder_checkpoint:
        raise RuntimeError(
            "The checkpoint does not contain SAM3 vision encoder weights")
    missing_keys, unexpected_keys = encoder.load_state_dict(
        encoder_checkpoint, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "SAM3 vision checkpoint does not match the encoder: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")

    loaded_numel = sum(tensor.numel() for tensor in encoder_checkpoint.values())
    total_numel = sum(parameter.numel() for parameter in encoder.parameters())
    print_log(
        "[SAM3] Vision checkpoint loaded successfully: "
        f"matched_tensors={len(encoder_checkpoint)}, "
        f"loaded_parameters={loaded_numel:,}, "
        f"encoder_parameters={total_numel:,}",
        logger="current",
    )

    encoder = encoder.to(device)
    if eval_mode:
        encoder.eval()
    print_log(
        f"[SAM3] Vision encoder ready: device={device}, "
        f"eval_mode={eval_mode}, compile={compile}, "
        f"rope={'real' if use_rope_real else 'complex'}",
        logger="current",
    )
    return encoder
