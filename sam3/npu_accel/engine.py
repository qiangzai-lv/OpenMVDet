"""
AscendTurbo Engine v2 — 以编译级算子替换为核心的加速引擎

vs v1 的根本区别:
  v1: 包装式优化 (Python 层 wrapper, DtypeCaster 弄巧成拙)
  v2: 编译级优化 (CANN 融合 kernel, 直接替换执行路径)

设计原则:
  1. 简单 .half() 是基线, 框架只做加分项, 不做减分项
  2. 每个优化都有 micro-benchmark 证实的增量
  3. FP32 vs FP16 在框架内部做控制变量对比
"""

import time, logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
from PIL import Image

from .config import TurboConfig
from .compiled_ops import (
    install_compiled_ops, uninstall_compiled_ops,
    replace_layernorms, replace_gelus,
)
from .vit_patcher import patch_vit_blocks
from .fused_preprocess import FusedPreprocessor

logger = logging.getLogger(__name__)


class AscendTurbo:
    """
    昇腾推理加速引擎 v2

    用法:
        config = TurboConfig.fp16()
        engine = AscendTurbo(model, config)
        result = engine.infer(image, "cat")
        engine.benchmark(image, "cat")
    """

    def __init__(self, model: nn.Module, config: Optional[TurboConfig] = None):
        self.config = config or TurboConfig.fp16()
        self.config.validate()
        self.device = torch.device("npu:0")
        self.dtype = self.config.dtype

        self._setup_npu_globals()

        # ── 编译级算子安装 ──
        self._installed_ops = install_compiled_ops(
            fuse_add_layernorm=self.config.fuse_add_layernorm,
            fuse_gelu=self.config.fuse_gelu,
            fix_grid_sample_dtype=self.config.fix_grid_sample_dtype,
            fuse_roi_align=self.config.fuse_roi_align,
        )

        # ── 模型精度转换 (直接 .half(), 不做 LayerNorm 例外) ──
        if self.dtype == torch.float16:
            model = model.half()
        elif self.dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)
        self.model = model.to(self.device).eval()

        # ── ViT Block forward 融合 patch ──
        self._n_patched_blocks = 0
        if self.config.fuse_add_layernorm:
            self._n_patched_blocks = patch_vit_blocks(self.model)
            n_gelus = replace_gelus(self.model)
            logger.info(f"Replaced {n_gelus} GELU → FusedGELU")

        # ── 融合预处理器 ──
        self._preprocessor = FusedPreprocessor(
            self.config.resolution, self.dtype, self.device
        )

        # ── 文本缓存 ──
        self._text_cache = {} if self.config.cache_text_features else None

        # ── 预热 ──
        self._warmup()

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"AscendTurbo v2 ready: dtype={self.dtype}, "
                    f"params={param_count:.0f}M, "
                    f"patched_blocks={self._n_patched_blocks}, "
                    f"compiled_ops={self._installed_ops}")

    def _setup_npu_globals(self):
        import torch_npu
        torch_npu.npu.set_compile_mode(jit_compile=self.config.jit_compile)
        torch.npu.config.allow_internal_format = self.config.allow_internal_format

    def _warmup(self):
        dummy = torch.randn(1, 3, self.config.resolution, self.config.resolution,
                            device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            for _ in range(self.config.warmup_iters):
                _ = self.model.backbone.forward_image(dummy)
        torch.npu.synchronize()

    def _get_size(self, image):
        if isinstance(image, Image.Image):
            return image.size[0], image.size[1]  # w, h
        elif isinstance(image, np.ndarray):
            return image.shape[1], image.shape[0]  # w, h
        elif isinstance(image, torch.Tensor):
            return image.shape[-1], image.shape[-2]
        raise ValueError(f"Unsupported: {type(image)}")

    def _text_encode(self, prompt: str) -> dict:
        if self._text_cache is not None and prompt in self._text_cache:
            return self._text_cache[prompt]

        with torch.inference_mode(), torch.amp.autocast("npu", dtype=self.dtype):
            features = self.model.backbone.forward_text([prompt], device=str(self.device))

        if self._text_cache is not None:
            self._text_cache[prompt] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }
        return features

    @torch.inference_mode()
    def infer(self, image, prompt: str, threshold: float = None):
        """
        单帧推理

        Returns: dict with masks, boxes, scores, latency_ms
        """
        threshold = threshold or self.config.confidence_threshold
        orig_w, orig_h = self._get_size(image)

        torch.npu.synchronize()
        t0 = time.perf_counter()

        # 预处理
        img_tensor = self._preprocessor(image)

        # Backbone
        with torch.amp.autocast("npu", dtype=self.dtype):
            backbone_out = self.model.backbone.forward_image(img_tensor)

        # inst interactivity
        if self.model.inst_interactive_predictor is not None:
            if "sam2_backbone_out" in backbone_out:
                sam2_bo = backbone_out["sam2_backbone_out"]
                sam2_bo["backbone_fpn"][0] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                        sam2_bo["backbone_fpn"][0]))
                sam2_bo["backbone_fpn"][1] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                        sam2_bo["backbone_fpn"][1]))

        # Text
        text_out = self._text_encode(prompt)
        backbone_out.update(text_out)

        # Detection + Segmentation
        from sam3.model.data_misc import FindStage
        find_input = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None,
            input_boxes_label=None, input_points=None, input_points_mask=None,
        )
        geo_prompt = self.model._get_dummy_prompt()

        with torch.amp.autocast("npu", dtype=self.dtype):
            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=geo_prompt,
                find_target=None,
            )

        # Postprocess
        from sam3.model import box_ops
        from sam3.model.data_misc import interpolate

        out_probs = outputs["pred_logits"].sigmoid()
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence).squeeze(-1)
        keep = out_probs > threshold

        masks_out = boxes_out = scores_out = None
        n_det = keep.sum().item()
        if n_det > 0:
            scores_out = out_probs[keep]
            masks_raw = outputs["pred_masks"][keep]
            boxes_raw = outputs["pred_boxes"][keep]

            boxes_out = box_ops.box_cxcywh_to_xyxy(boxes_raw)
            scale = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                                 device=self.device, dtype=boxes_out.dtype)
            boxes_out = boxes_out * scale[None, :]

            masks_out = interpolate(
                masks_raw.unsqueeze(1).float(),
                (orig_h, orig_w),
                mode="bilinear", align_corners=False
            ).sigmoid() > 0.5

        torch.npu.synchronize()
        latency = (time.perf_counter() - t0) * 1000

        return {
            "masks": masks_out,
            "boxes": boxes_out,
            "scores": scores_out,
            "n_detections": n_det,
            "latency_ms": latency,
        }

    def benchmark(self, image, prompt: str = "cat",
                  n_warmup: int = 10, n_runs: int = 30):
        """
        全面基准测试 — 各阶段 + 端到端 + 视频流模拟
        """
        orig_w, orig_h = self._get_size(image)

        # Warmup
        for _ in range(n_warmup):
            self.infer(image, prompt)
        torch.npu.synchronize()

        # ── Stage profiling ──
        img_tensor = self._preprocessor(image)

        # Preprocess
        times_pre = []
        for _ in range(n_runs):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _ = self._preprocessor(image)
            torch.npu.synchronize()
            times_pre.append((time.perf_counter() - t0) * 1000)

        # Backbone
        times_bb = []
        for _ in range(n_runs):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("npu", dtype=self.dtype):
                _ = self.model.backbone.forward_image(img_tensor)
            torch.npu.synchronize()
            times_bb.append((time.perf_counter() - t0) * 1000)

        # Text (no cache)
        self._text_cache = {} if self._text_cache is not None else None
        times_txt = []
        for _ in range(n_runs):
            if self._text_cache is not None:
                self._text_cache.clear()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _ = self._text_encode(prompt)
            torch.npu.synchronize()
            times_txt.append((time.perf_counter() - t0) * 1000)

        # E2E
        times_e2e = []
        for _ in range(n_runs):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            self.infer(image, prompt)
            torch.npu.synchronize()
            times_e2e.append((time.perf_counter() - t0) * 1000)

        # Video stream
        n_stream = 100
        times_stream = []
        for _ in range(n_stream):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            self.infer(image, prompt)
            torch.npu.synchronize()
            times_stream.append((time.perf_counter() - t0) * 1000)

        def stats(arr):
            a = np.array(arr)
            return {"mean": round(float(np.mean(a)), 1),
                    "median": round(float(np.median(a)), 1),
                    "std": round(float(np.std(a)), 1),
                    "min": round(float(np.min(a)), 1),
                    "max": round(float(np.max(a)), 1)}

        arr_e2e = np.array(times_e2e)
        arr_st = np.array(times_stream)

        return {
            "config": {
                "dtype": str(self.dtype),
                "resolution": self.config.resolution,
                "fuse_add_layernorm": self.config.fuse_add_layernorm,
                "fuse_gelu": self.config.fuse_gelu,
                "fuse_roi_align": self.config.fuse_roi_align,
                "fix_grid_sample": self.config.fix_grid_sample_dtype,
                "patched_vit_blocks": self._n_patched_blocks,
            },
            "stages": {
                "preprocess_ms": stats(times_pre),
                "backbone_ms": stats(times_bb),
                "text_encoder_ms": stats(times_txt),
            },
            "e2e": {
                **stats(times_e2e),
                "p95": round(float(np.percentile(arr_e2e, 95)), 1),
                "fps": round(1000 / float(np.mean(arr_e2e)), 2),
            },
            "video_stream": {
                "frames": n_stream,
                "mean_ms": round(float(np.mean(arr_st)), 1),
                "p95_ms": round(float(np.percentile(arr_st, 95)), 1),
                "mean_fps": round(1000 / float(np.mean(arr_st)), 2),
                "p95_fps": round(1000 / float(np.percentile(arr_st, 95)), 2),
            },
            "memory_gb": round(torch.npu.max_memory_allocated() / 1024**3, 2),
        }

    def cleanup(self):
        uninstall_compiled_ops()
        if self._text_cache:
            self._text_cache.clear()
        torch.npu.empty_cache()
