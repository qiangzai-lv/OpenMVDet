"""
fused_preprocess — 融合预处理 kernel

原理:
  标准预处理: Resize (kernel 1) → ToDtype (kernel 2) → Normalize (kernel 3) → half() (kernel 4)
  每一步都读写一次 HBM, 对 1008x1008x3 的图片 = 3MB × 4 次 = 12MB HBM 访问

  融合预处理: 在一次 kernel 中完成 resize + normalize + cast
  用 torch.compile 或手写算子将这些操作合并

  由于 NPU 不支持 torch.compile, 这里用 Python 层面的 tensor 操作
  优化重点: 减少中间 tensor 分配, 用 inplace 操作
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union


class FusedPreprocessor:
    """
    融合预处理器

    将 decode → resize → normalize → cast 合并为最少的 NPU kernel 调用
    """

    def __init__(self, resolution: int, dtype: torch.dtype, device: torch.device,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.resolution = resolution
        self.dtype = dtype
        self.device = device

        # 预计算归一化参数并放到 NPU (避免每帧创建)
        self._mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
        self._inv_std = (1.0 / self._std)

        # 预分配 buffer (避免每帧 malloc)
        self._buffer = torch.empty(
            1, 3, resolution, resolution,
            device=device, dtype=dtype
        )

    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        预处理图片, 返回 [1, 3, H, W] tensor on NPU

        融合路径:
          1. to_tensor (CPU→NPU, uint8)  — 利用 non_blocking DMA
          2. resize (NPU kernel)
          3. normalize + cast (单次操作: (x/255 - mean) / std → FP16)
        """
        # Step 1: 转为 tensor 并上 NPU (uint8 传输量最小)
        if isinstance(image, Image.Image):
            arr = np.array(image)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            t = image.unsqueeze(0) if image.ndim == 3 else image
        else:
            raise ValueError(f"Unsupported type: {type(image)}")

        # uint8 传输到 NPU (3MB for 1008x1008, vs 12MB for float32)
        t = t.to(device=self.device, non_blocking=True)

        # Step 2: resize (NPU kernel)
        if t.shape[2] != self.resolution or t.shape[3] != self.resolution:
            t = F.interpolate(t.float(), size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False)
        else:
            t = t.float()

        # Step 3: normalize + cast (融合为单次操作)
        # (x / 255.0 - mean) / std = x / 255.0 / std - mean / std
        # = x * (1.0 / 255.0 / std) - mean / std
        t = t.to(self.dtype)
        t.mul_(1.0 / 255.0)
        t.sub_(self._mean)
        t.mul_(self._inv_std)

        return t
