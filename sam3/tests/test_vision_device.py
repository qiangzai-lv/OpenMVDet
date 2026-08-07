import os
import subprocess
import sys


def _run_without_cuda(code):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    return subprocess.run(
        [sys.executable, '-W', 'error::UserWarning', '-c', code],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True)


def test_vision_builder_imports_without_cuda_warning():
    result = _run_without_cuda(
        'from sam3.vision_builder import build_sam3_vision_encoder')
    assert result.returncode == 0, result.stderr


def test_position_encoding_precompute_is_cpu_safe():
    result = _run_without_cuda(
        'import torch\n'
        'from sam3.model.position_encoding import PositionEmbeddingSine\n'
        'module = PositionEmbeddingSine(32, precompute_resolution=28)\n'
        'output = module(torch.zeros(2, 1, 8, 10))\n'
        'assert output.device.type == "cpu"\n'
        'assert output.shape == (2, 32, 8, 10)')
    assert result.returncode == 0, result.stderr


def test_default_device_falls_back_to_cpu_without_accelerator():
    result = _run_without_cuda(
        'from sam3.device import get_default_device\n'
        'assert get_default_device().type == "cpu"')
    assert result.returncode == 0, result.stderr
