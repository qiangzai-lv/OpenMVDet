# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from .vision_builder import build_sam3_vision_encoder

__version__ = "0.1.0"

__all__ = [
    "build_sam3_image_model",
    "build_sam3_predictor",
    "build_sam3_vision_encoder",
]


def __getattr__(name):
    if name in {"build_sam3_image_model", "build_sam3_predictor"}:
        from . import model_builder

        return getattr(model_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
