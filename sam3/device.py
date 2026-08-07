import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def is_npu_available():
    npu = getattr(torch, "npu", None)
    return torch_npu is not None and npu is not None and npu.is_available()


def get_default_device():
    if is_npu_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
