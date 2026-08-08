# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from .vision_builder import build_sam3_vision_encoder

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "build_sam3_vision_encoder"]


def __getattr__(name):
    if name == "build_sam3_image_model":
        from .model_builder import build_sam3_image_model

        return build_sam3_image_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
