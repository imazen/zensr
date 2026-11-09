"""Two-call CUDA SR benchmark: resident inference and separate pinned transfers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from cpu_plan import sha
from sr_models import load


def spanf(a):
    spec = importlib.util.spec_from_file_location('zensr_author_spanf', a.spanf_root / 'team24_SPANF.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SPANF(3, 3, feature_channels=32, upscale=4)
    model.load_state_dict(torch.load(a.spanf_root / 'team24_spanf.pth',
                                    map_location='cpu', weights_only=True), strict=True)
    order = ['conv_near.weight']
    for b in range(1, 6):
        for c in ('c1_r', 'c2_r', 'c3_r'):
            order.extend([f'block_{b}.{c}.eval_conv.weight', f'block_{b}.{c}.eval_conv.bias'])
    order.extend(['conv_cat.weight', 'conv_cat.bias', 'conv_2.eval_conv.weight', 'conv_2.eval_conv.bias'])
    state = model.state_dict()
    original = b''.join(state[k].cpu().numpy().astype('<f4').tobytes() for k in order)
    if original != a.spanf_f32.read_bytes():
        raise ValueError('author checkpoint differs from retained SPANF f32 weights')
    mixed = a.spanf_mixed.read_bytes()
    offset = 0
    for key in order:
        dtype = '<f2' if key.endswith('.weight') else '<f4'
        count = state[key].numel()
        values = np.frombuffer(mixed, dtype=dtype, count=count, offset=offset).astype('<f4')
        offset += count * np.dtype(dtype).itemsize
        state[key].copy_(torch.from_numpy(values.reshape(tuple(state[key].shape))))
    if offset != len(mixed):
        raise ValueError('native SPANF mixed-weight layout differs')
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def main(a):
    if not torch.cuda.is_available() or a.samples != 2:
        raise ValueError('requires CUDA and exactly two timed calls')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    a.output.mkdir(exist_ok=True)

    def progress(stage, completed):
        (a.output / 'PROGRESS.json').write_text(json.dumps(dict(
            model=a.name, stage=stage, completed=completed, required=2, pid=os.getpid())) + '\n')

    progress('loading', 0)
    model = spanf(a) if a.name == 'spanf' else load(a.dscf_root, 'dscf').model
    model = model.cuda().eval().requires_grad_(False)
    if any(p.dtype != torch.float32 or p.device.type != 'cuda' for p in model.parameters()):
        raise ValueError('expected CUDA FP32 parameters')
    data = np.fromfile(a.input, dtype='<f4')
    if data.size != 3 * 2048 * 2048 or not np.isfinite(data).all():
        raise ValueError('input geometry or values differ')
    cpu_input = torch.from_numpy(data.reshape(1, 3, 2048, 2048)).pin_memory()
    gpu_input = torch.empty_like(cpu_input, device='cuda')
    cpu_output = torch.empty((1, 3, 8192, 8192), dtype=torch.float32, pin_memory=True)

    def event_call(call):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        begin.record()
        result = call()
        end.record()
        end.synchronize()
        return result, begin.elapsed_time(end), (time.perf_counter_ns() - started) / 1e6

    def checksum(output):
        if tuple(output.shape) != (1, 3, 8192, 8192) or not torch.isfinite(cpu_output).all():
            raise ValueError('invalid full-resolution output')
        return hashlib.sha256(memoryview(cpu_output.numpy()).cast('B')).hexdigest()

    gpu_input.copy_(cpu_input, non_blocking=True)
    torch.cuda.synchronize()
    progress('warming', 0)
    warm = model(gpu_input)
    cpu_output.copy_(warm, non_blocking=True)
    torch.cuda.synchronize()
    expected = checksum(warm)
    del warm
    samples = []
    for iteration in range(a.samples):
        progress('timing', iteration)
        _, h2d_ms, _ = event_call(lambda: gpu_input.copy_(cpu_input, non_blocking=True))
        torch.cuda.reset_peak_memory_stats()
        output, inference_ms, synchronized_host_ms = event_call(lambda: model(gpu_input))
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        _, d2h_ms, _ = event_call(lambda: cpu_output.copy_(output, non_blocking=True))
        if checksum(output) != expected:
            raise ValueError('timed output differs from warm control')
        sample = dict(iteration=iteration + 1, inference_cuda_ms=inference_ms,
                      inference_synchronized_host_ms=synchronized_host_ms,
                      pinned_h2d_cuda_ms=h2d_ms, pinned_d2h_cuda_ms=d2h_ms,
                      peak_allocated_bytes=allocated, peak_reserved_bytes=reserved)
        samples.append(sample)
        print('SAMPLE ' + json.dumps(sample), flush=True)
        print(f'latency_ns={inference_ms * 1e6:.0f}', flush=True)
        del output
    identity = dict(name=a.name, input_sha256=sha(a.input), output_sha256=expected,
                    shape=[1, 3, 8192, 8192], device=torch.cuda.get_device_name(0),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), tf32=False, precision='FP32',
                    spanf_weights='native mixed f16/f32 expanded to FP32' if a.name == 'spanf' else None,
                    parameters=sum(p.numel() for p in model.parameters()),
                    transfers='preallocated pinned buffers, separate CUDA events',
                    memory='PyTorch inference high-water allocated and reserved bytes; model/input/output included')
    print('IDENTITY ' + json.dumps(identity), flush=True)
    (a.output / 'RESULT.json').write_text(json.dumps(dict(identity=identity, samples=samples), indent=2) + '\n')
    progress('completed', 2)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', choices=['spanf', 'dscf'], required=True)
    for name in ['spanf-root', 'spanf-f32', 'spanf-mixed', 'dscf-root', 'input', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--samples', type=int, default=2)
    main(p.parse_args())
