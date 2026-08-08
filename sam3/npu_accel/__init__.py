"""
AscendTurbo — 昇腾 NPU 第一性原理推理加速框架

核心设计思想:
  不做"包装式"优化 (纯 Python 包一层), 而是通过编译级算子替换
  让每一个 kernel 都跑在 NPU 原生最优路径上。

  本框架做了三件事:
    1. 编译级算子替换 — 用 CANN 原生融合算子替换 PyTorch 分解算子
    2. dtype 全链路对齐 — 消除 FP16↔FP32 隐式转换和 CPU 回退
    3. 运行时优化     — 文本缓存 / stream 并行 / 预热

  已验证的原生融合算子及其收益:
    torch.ops.npu.npu_add_layer_norm  → Add+LayerNorm 单 kernel, 1.37x
    torch.ops.npu.npu_fast_gelu       → GELU 融合, 减少 1 次 kernel launch
    torch.ops.npu.npu_roi_align       → 消除 CPU 回退 (原生 NPU kernel)
    torch.ops.npu.npu_silu            → SiLU 融合

Author: AscendTurbo Framework
Target: Huawei Ascend 910B2 + CANN 8.x + torch_npu 2.5
"""

__version__ = "2.0.0"

from .engine import AscendTurbo
from .config import TurboConfig

__all__ = ["AscendTurbo", "TurboConfig"]
