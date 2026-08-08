"""TurboConfig — 加速配置"""

from dataclasses import dataclass
import torch


@dataclass
class TurboConfig:
    # 精度控制
    dtype: torch.dtype = torch.float16

    # 编译级算子替换开关
    fuse_add_layernorm: bool = True     # npu_add_layer_norm 替换 Add+LN
    fuse_gelu: bool = True              # npu_fast_gelu 替换 F.gelu
    fuse_roi_align: bool = True         # npu_roi_align 替换 torchvision
    fix_grid_sample_dtype: bool = True  # 自动对齐 grid_sample 输入 dtype

    # NPU 全局配置
    allow_internal_format: bool = True  # FRACTAL_NZ 格式
    jit_compile: bool = False           # 二进制编译缓存

    # 运行时优化
    cache_text_features: bool = True
    warmup_iters: int = 3
    confidence_threshold: float = 0.2
    resolution: int = 1008

    def validate(self):
        assert self.resolution % 14 == 0
        assert self.dtype in (torch.float16, torch.bfloat16, torch.float32)
        return self

    @classmethod
    def fp16(cls):
        """FP16 全量优化"""
        return cls(dtype=torch.float16)

    @classmethod
    def fp32(cls):
        """FP32 基线 (仅编译级算子替换, 不改精度)"""
        return cls(dtype=torch.float32)
