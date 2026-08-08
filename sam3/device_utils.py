"""
SAM3 国产算力适配层 - 设备抽象工具
支持：华为昇腾 (NPU), 寒武纪 (MLU), 海光 DCU (ROCm), NVIDIA CUDA
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)

_device_type = None
_device_module = None

BACKEND_PRIORITY = ["cuda", "npu", "mlu"]


def _detect_device_type():
    """自动检测可用的加速设备类型"""
    global _device_type, _device_module

    if _device_type is not None:
        return _device_type

    env_override = os.environ.get("SAM3_DEVICE_TYPE", "").lower()
    if env_override in ("cuda", "npu", "mlu", "cpu"):
        _device_type = env_override
        _device_module = _get_device_module(_device_type)
        logger.info(f"SAM3 device type overridden by env: {_device_type}")
        return _device_type

    if torch.cuda.is_available():
        _device_type = "cuda"
        _device_module = torch.cuda
        logger.info("SAM3 using CUDA (NVIDIA GPU / ROCm DCU)")
        return _device_type

    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            _device_type = "npu"
            _device_module = torch.npu
            logger.info("SAM3 using NPU (Huawei Ascend)")
            return _device_type
    except ImportError:
        pass

    try:
        import torch_mlu  # noqa: F401
        if hasattr(torch, "mlu") and torch.mlu.is_available():
            _device_type = "mlu"
            _device_module = torch.mlu
            logger.info("SAM3 using MLU (Cambricon)")
            return _device_type
    except ImportError:
        pass

    _device_type = "cpu"
    _device_module = None
    logger.warning("SAM3: No accelerator found, falling back to CPU")
    return _device_type


def _get_device_module(dtype):
    if dtype == "cuda":
        return torch.cuda
    elif dtype == "npu":
        import torch_npu  # noqa: F401
        return torch.npu
    elif dtype == "mlu":
        import torch_mlu  # noqa: F401
        return torch.mlu
    return None


def get_device_type():
    """获取当前设备类型字符串: 'cuda', 'npu', 'mlu', 'cpu'"""
    return _detect_device_type()


def get_device_module():
    """获取设备模块 (torch.cuda / torch.npu / torch.mlu / None)"""
    _detect_device_type()
    return _device_module


def get_device(index=0):
    """获取 torch.device 对象"""
    dt = get_device_type()
    if dt == "cpu":
        return torch.device("cpu")
    return torch.device(dt, index)


def get_default_device():
    """获取默认设备，可用于替代 'cuda'"""
    return get_device(0)


def is_accelerator_available():
    """检查是否有加速器可用"""
    return get_device_type() != "cpu"


def device_count():
    """返回加速器数量"""
    mod = get_device_module()
    if mod is None:
        return 0
    return mod.device_count()


def current_device():
    """返回当前设备索引"""
    mod = get_device_module()
    if mod is None:
        return 0
    return mod.current_device()


def set_device(device):
    """设置当前设备"""
    mod = get_device_module()
    if mod is not None:
        mod.set_device(device)


def empty_cache():
    """清空设备缓存"""
    mod = get_device_module()
    if mod is not None:
        mod.empty_cache()


def memory_allocated(device=None):
    """获取已分配内存"""
    mod = get_device_module()
    if mod is None:
        return 0
    return mod.memory_allocated(device)


def memory_reserved(device=None):
    """获取已保留内存"""
    mod = get_device_module()
    if mod is None:
        return 0
    return mod.memory_reserved(device)


def max_memory_allocated(device=None):
    """获取最大已分配内存"""
    mod = get_device_module()
    if mod is None:
        return 0
    return mod.max_memory_allocated(device)


def max_memory_reserved(device=None):
    """获取最大已保留内存"""
    mod = get_device_module()
    if mod is None:
        return 0
    return mod.max_memory_reserved(device)


def reset_peak_memory_stats(device=None):
    """重置峰值内存统计"""
    mod = get_device_module()
    if mod is not None and hasattr(mod, "reset_peak_memory_stats"):
        mod.reset_peak_memory_stats(device)


def manual_seed_all(seed):
    """设置所有设备的随机种子"""
    mod = get_device_module()
    if mod is not None and hasattr(mod, "manual_seed_all"):
        mod.manual_seed_all(seed)


def synchronize(device=None):
    """同步设备"""
    mod = get_device_module()
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize(device)


def to_device(tensor_or_model, device=None, non_blocking=False):
    """将张量或模型移动到加速设备"""
    if device is None:
        device = get_default_device()
    return tensor_or_model.to(device, non_blocking=non_blocking)


def get_autocast_device_type():
    """获取 autocast 使用的设备类型字符串"""
    dt = get_device_type()
    if dt == "npu":
        return "npu"
    elif dt == "mlu":
        return "mlu"
    return "cuda"


def get_dist_backend():
    """获取分布式训练后端"""
    dt = get_device_type()
    if dt == "npu":
        return "hccl"
    elif dt == "mlu":
        return "cncl"
    return "nccl"


def setup_tf32():
    """安全地配置 TF32（仅在支持的平台上）"""
    dt = get_device_type()
    if dt == "cuda":
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap is not None:
                major, _ = cap
                if major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
    elif dt == "npu":
        pass


def supports_torch_compile():
    """检查当前平台是否支持 torch.compile"""
    dt = get_device_type()
    if dt == "cuda":
        return True
    if dt == "npu":
        return hasattr(torch, "compile")
    return False


def safe_compile(fn, **kwargs):
    """安全的 torch.compile 封装，不支持时直接返回原函数"""
    if not supports_torch_compile():
        return fn
    try:
        return torch.compile(fn, **kwargs)
    except Exception as e:
        logger.warning(f"torch.compile failed, using eager mode: {e}")
        return fn


def get_sdpa_backend_context():
    """获取 SDPA 后端上下文管理器（仅 CUDA 平台有效）"""
    dt = get_device_type()
    if dt == "cuda":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    from contextlib import nullcontext
    return nullcontext()


def is_on_device(tensor):
    """检查张量是否在加速设备上"""
    dt = get_device_type()
    if dt == "cuda":
        return tensor.is_cuda
    return tensor.device.type == dt
