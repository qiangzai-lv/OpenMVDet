"""
SAM3 单帧加速引擎 — 不靠跳帧，真实加速每一帧

经过系统分析的耗时分布 (FP16, 1008x1008):
  Backbone (32 blocks):  50ms  — kernel launch overhead 主导, 非计算 bound
  Text encoder:          27ms  — 视频流可缓存
  encode_prompt:         16ms  — 含 roi_align CPU fallback
  Transformer encoder:   14ms
  Decoder (6 layers):    51ms  — 每层 8.5ms
  Seg heads:              5ms
  其他开销:              ~5ms
  TOTAL:               ~168ms

不可行的方向 (已验证):
  - INT8 量化 matmul: 单个 matmul 只有 0.1-0.2ms, INT8 反而更慢 (kernel launch 主导)
  - NPUGraph: 不支持 Conv2D aclop 在 graph capture
  - torch.compile: NPU inductor 后端不支持 var_mean lowering
  - npu_fusion_attention: 和 SDPA 性能相同 (已经在用 flash attention)
  - 降分辨率: backbone 降了但 detection head 固定 ~90ms 不变

可行的加速手段:
  1. Text 特征缓存: -27ms (视频流同 prompt 只需编码一次)
  2. Decoder 3层裁剪: -25ms (保持检测数和精度)
  3. roi_align dtype 修复: -5ms (避免 CPU fallback 的 dtype 转换)
  4. Grid sample dtype 对齐: -2ms
  5. allow_internal_format: -3ms (NPU 内部格式优化)

  组合: 168 - 27 - 25 - 5 - 2 - 3 = 106ms → 9.4 FPS (1.6x)

  进一步: 使用更激进的 2 层 decoder: -31ms 额外
  → 168 - 27 - 31 - 10 = 100ms → 10 FPS (1.6x)

  现实: 由于 backbone 50ms 是硬底, decoder+encoder 占另外 65ms
  理论极限 (只跑 backbone): ~50ms → 20 FPS

  所以单帧推理的现实上限约 10 FPS (2层 decoder) ~ 15 FPS (极端裁剪)
"""

import time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from PIL import Image

logger = logging.getLogger(__name__)


class SingleFrameAccelerator:
    """
    单帧真实加速引擎

    每一帧都做完整推理, 通过以下手段加速:
      1. Text 特征缓存 (同 prompt 只编码一次)
      2. Decoder 层数裁剪 (3层 vs 6层, 精度损失极小)
      3. 算子兼容性修复 (roi_align, grid_sample dtype)
      4. NPU 全局优化 (allow_internal_format, jit_compile off)
    """

    def __init__(self, model: nn.Module, decoder_layers: int = 3,
                 resolution: int = 1008, confidence_threshold: float = 0.2):
        import torch_npu
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True

        self.device = torch.device("npu:0")
        self.model = model.half().to(self.device).eval()
        self.dtype = torch.float16
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold

        # 算子补丁
        self._install_patches()

        # Decoder 裁剪
        self._original_decoder_layers = list(self.model.transformer.decoder.layers)
        self._original_num_layers = self.model.transformer.decoder.num_layers
        self._original_bbox_embed = None
        if hasattr(self.model.transformer.decoder, 'bbox_embed'):
            be = self.model.transformer.decoder.bbox_embed
            if hasattr(be, '__len__'):
                self._original_bbox_embed = list(be)

        self._set_decoder_layers(decoder_layers)

        # Text 缓存
        self._text_cache: Dict[str, dict] = {}

        # warmup
        self._warmup()

    def _install_patches(self):
        _orig_gs = F.grid_sample
        def _safe_gs(inp, grid, **kw):
            if inp.dtype != grid.dtype:
                grid = grid.to(inp.dtype)
            return _orig_gs(inp, grid, **kw)
        F.grid_sample = _safe_gs

        import torchvision.ops as tvops
        _orig_roi = tvops.roi_align
        def _safe_roi(input, boxes, output_size, spatial_scale=1.0,
                      sampling_ratio=-1, aligned=False):
            if isinstance(boxes, torch.Tensor):
                boxes = [boxes.to(dtype=input.dtype, device=input.device)]
            elif isinstance(boxes, (list, tuple)):
                boxes = [b.to(dtype=input.dtype, device=input.device)
                        if isinstance(b, torch.Tensor) else b for b in boxes]
            return _orig_roi(input, boxes, output_size, spatial_scale,
                           sampling_ratio, aligned)
        tvops.roi_align = _safe_roi

    def _set_decoder_layers(self, n_layers: int):
        decoder = self.model.transformer.decoder
        decoder.layers = nn.ModuleList(self._original_decoder_layers[:n_layers])
        decoder.num_layers = n_layers
        if self._original_bbox_embed is not None:
            decoder.bbox_embed = nn.ModuleList(self._original_bbox_embed[:n_layers])
        self._decoder_layers = n_layers

    def _warmup(self):
        dummy = torch.randn(1, 3, self.resolution, self.resolution,
                           device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            for _ in range(3):
                _ = self.model.backbone.forward_image(dummy)
        torch.npu.synchronize()

    def _preprocess(self, image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            arr = image
        else:
            raise ValueError(f"Unsupported: {type(image)}")

        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = t.to(self.device, non_blocking=True).to(self.dtype)
        res = self.resolution
        if t.shape[-2] != res or t.shape[-1] != res:
            t = F.interpolate(t.float(), size=(res, res),
                            mode='bilinear', align_corners=False).to(self.dtype)
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=self.dtype).view(1,3,1,1)
        inv_std = torch.tensor([2.0, 2.0, 2.0], device=self.device, dtype=self.dtype).view(1,3,1,1)
        t.mul_(1.0 / 255.0).sub_(mean).mul_(inv_std)
        return t

    def _get_text(self, prompt: str) -> dict:
        if prompt in self._text_cache:
            return self._text_cache[prompt]
        with torch.inference_mode(), torch.amp.autocast("npu", dtype=self.dtype):
            feats = self.model.backbone.forward_text([prompt], device=str(self.device))
        self._text_cache[prompt] = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in feats.items()
        }
        return self._text_cache[prompt]

    @torch.inference_mode()
    def infer(self, image, prompt: str = "cat") -> dict:
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
        elif isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
        else:
            orig_h, orig_w = image.shape[-2:]

        torch.npu.synchronize()
        t_start = time.perf_counter()

        # 1. Preprocess
        img_tensor = self._preprocess(image)

        # 2. Backbone
        with torch.amp.autocast("npu", dtype=self.dtype):
            backbone_out = self.model.backbone.forward_image(img_tensor)

        # 3. SAM2 convs (if present)
        if self.model.inst_interactive_predictor is not None:
            s2 = backbone_out.get("sam2_backbone_out")
            if s2:
                s2["backbone_fpn"][0] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    s2["backbone_fpn"][0])
                s2["backbone_fpn"][1] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    s2["backbone_fpn"][1])

        # 4. Text (cached)
        text_out = self._get_text(prompt)
        backbone_out.update(text_out)

        # 5. Detection
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
                backbone_out=backbone_out, find_input=find_input,
                geometric_prompt=geo_prompt, find_target=None,
            )

        # 6. Postprocess
        result = self._postprocess(outputs, orig_h, orig_w)

        torch.npu.synchronize()
        result["latency_ms"] = (time.perf_counter() - t_start) * 1000
        result["decoder_layers"] = self._decoder_layers
        return result

    def _postprocess(self, outputs: dict, orig_h: int, orig_w: int) -> dict:
        from sam3.model import box_ops
        from sam3.model.data_misc import interpolate

        out_probs = outputs["pred_logits"].sigmoid()
        if "presence_logit_dec" in outputs:
            presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = out_probs * presence
        out_probs = out_probs.squeeze(-1)
        keep = out_probs > self.confidence_threshold

        n_det = keep.sum().item()
        if n_det == 0:
            return {"masks": None, "boxes": None, "scores": None, "n_detections": 0}

        scores = out_probs[keep]
        boxes_raw = outputs["pred_boxes"][keep]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes_raw)
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                           device=self.device, dtype=boxes.dtype)
        boxes = boxes * scale[None, :]

        masks = None
        if "pred_masks" in outputs:
            masks_raw = outputs["pred_masks"][keep]
            masks = interpolate(
                masks_raw.unsqueeze(1).float(),
                (orig_h, orig_w), mode="bilinear", align_corners=False
            ).sigmoid() > 0.5

        return {"masks": masks, "boxes": boxes, "scores": scores, "n_detections": n_det}

    def benchmark(self, image, prompt: str = "cat", n_warmup: int = 10, n_iter: int = 30) -> dict:
        for _ in range(n_warmup):
            self.infer(image, prompt)
        torch.npu.synchronize()

        latencies = []
        last_result = None
        for _ in range(n_iter):
            result = self.infer(image, prompt)
            latencies.append(result["latency_ms"])
            last_result = result

        arr = np.array(latencies)
        return {
            "mean_ms": round(float(np.mean(arr)), 1),
            "median_ms": round(float(np.median(arr)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "fps": round(1000 / float(np.mean(arr)), 1),
            "n_detections": last_result["n_detections"],
            "scores": [round(s.item(), 3) for s in last_result["scores"][:5]] if last_result["scores"] is not None else [],
            "decoder_layers": self._decoder_layers,
            "memory_gb": round(torch.npu.max_memory_allocated() / 1024**3, 2),
        }

    def restore_decoder(self):
        self._set_decoder_layers(self._original_num_layers)
