"""
vit_patcher — ViT Block 级别的编译算子注入

通过 patch Block.forward, 将 Pre-Norm ViT 中的
  x = residual + dropout(attn(norm(x)))
替换为
  x, norm_out = npu_add_layer_norm(residual, dropout(attn_out))
实现 Add + LayerNorm 的 kernel 融合。
"""

import torch
import torch.nn as nn
import types
import logging

logger = logging.getLogger(__name__)

_HAS_NPU = False
try:
    import torch_npu
    _HAS_NPU = hasattr(torch.ops, "npu") and hasattr(torch.ops.npu, "npu_add_layer_norm")
except ImportError:
    pass


def _fused_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    ViT Block 的融合 forward

    原始:
      shortcut = x
      x = self.norm1(x)                           # LN kernel
      x = self.ls1(self.attn(x))
      x = shortcut + self.dropout(self.drop_path(x))  # Add kernel
      x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

    融合:
      把 "residual add → 下一步的 LayerNorm" 合并为一次 npu_add_layer_norm 调用
      实际操作: 第二个 add + norm2 合并, 减少一次 kernel launch + 一次 HBM 往返
    """
    shortcut = x
    x = self.norm1(x)

    if self.window_size > 0:
        from sam3.model.vitdet import window_partition, window_unpartition
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)

    x = self.ls1(self.attn(x))

    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))

    attn_out = self.dropout(self.drop_path(x))

    # 融合: residual_add + norm2
    if _HAS_NPU and shortcut.device.type == "npu":
        # 确保 dtype 一致 (attn_out 可能因 drop_path 或 autocast 产生不同 dtype)
        if shortcut.dtype != attn_out.dtype:
            attn_out = attn_out.to(shortcut.dtype)

        gamma = self.norm2.weight
        beta = self.norm2.bias
        # npu_add_layer_norm 的 SupportInfo 要求 gamma/beta 的 dtype 匹配
        if gamma.dtype != shortcut.dtype:
            gamma = gamma.to(shortcut.dtype)
            beta = beta.to(shortcut.dtype)

        # npu_add_layer_norm 返回:
        #   [0] = LayerNorm(x1+x2) — normalized 输出
        #   [1] = mean
        #   [2] = rstd
        #   [3] = x1 + x2 — 原始求和
        norm2_out, _, _, fused_sum = torch.ops.npu.npu_add_layer_norm(
            shortcut, attn_out, gamma, beta, self.norm2.eps, True
        )
        mlp_out = self.dropout(self.drop_path(self.ls2(self.mlp(norm2_out))))
        x = fused_sum + mlp_out
    else:
        x = shortcut + attn_out
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

    return x


def patch_vit_blocks(model: nn.Module) -> int:
    """
    查找所有 ViT Block 并替换其 forward 为融合版本

    Returns: 被 patch 的 block 数量
    """
    from sam3.model.vitdet import Block

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Block):
            module.forward = types.MethodType(_fused_block_forward, module)
            count += 1

    if count > 0:
        logger.info(f"Patched {count} ViT Blocks with fused Add+LayerNorm forward")
    return count
