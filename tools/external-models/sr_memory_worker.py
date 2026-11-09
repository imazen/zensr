"""Measure final integer outputs and shorter DSCF activation lifetimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from cpu_plan import sha
from sr_models import load


def compact_dscf(model, image):
    """Same author ops/order; retain only the inference outputs used later."""
    # Derived from pinned DSCF.forward; source/checkpoint identities are bound
    # by the plan. Do not change convolution, activation or normalization math.
    model.mean = model.mean.type_as(image)
    image = (image - model.mean) * model.img_range
    feature = model.conv_1(image)
    del image
    first = model.block_1(feature)[0]
    value = model.block_2(first)[0]
    value = model.block_3(value)[0]
    value = model.block_4(value)[0]
    value = model.block_5(value)[0]
    last = model.block_6(value)
    del value
    value, branch = last[0], last[1]
    del last
    value = model.conv_2(value)
    joined = torch.cat([feature, value, first, branch], 1)
    del feature, value, first, branch
    value = model.conv_cat(joined)
    del joined
    return model.upsampler(value)


def quantize(output, bits):
    if bits == 0:
        return output
    maximum = 255 if bits == 8 else 65535
    # Output has no future model consumer. No intermediate activation changes.
    return output.mul_(maximum).add_(0.5).clamp_(0, maximum).to(
        torch.uint8 if bits == 8 else torch.uint16)


def digest(tensor):
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.numpy()).cast('B')).hexdigest()


@torch.inference_mode()
def main(a):
    if a.samples != 2 or (a.device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('requires explicit device and two calls')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261109)
    model = load(a.root, 'dscf', device=a.device).model
    image = torch.from_numpy(np.fromfile(a.input, dtype='<f4').reshape(1, 3, 2048, 2048)).to(a.device)
    forward = (lambda: compact_dscf(model, image)) if a.lifetime == 'compact' else (lambda: model(image))
    # Validate all float values against the already measured backend before
    # quantization; keep a digest instead of another full reference image.
    control = forward()
    if tuple(control.shape) != (1, 3, 8192, 8192) or not torch.isfinite(control).all():
        raise ValueError('nonfinite output or wrong geometry')
    float_hash = digest(control)
    if float_hash != a.expected_float:
        raise ValueError('float computation differs from frozen backend output')
    final_control = quantize(control, a.bits)
    expected = digest(final_control)
    del control, final_control
    cpu_output = None
    if a.device == 'cuda':
        dtype = torch.float32 if a.bits == 0 else (torch.uint8 if a.bits == 8 else torch.uint16)
        cpu_output = torch.empty((1, 3, 8192, 8192), dtype=dtype, pin_memory=True)
    for iteration in range(a.samples):
        if a.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        start = time.perf_counter_ns()
        output = quantize(forward(), a.bits)
        host_ns = time.perf_counter_ns() - start
        allocated = reserved = 0
        d2h_ms = None
        if a.device == 'cuda':
            end.record()
            end.synchronize()
            elapsed = round(begin.elapsed_time(end) * 1e6)
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            begin.record()
            cpu_output.copy_(output, non_blocking=True)
            end.record()
            end.synchronize()
            d2h_ms = begin.elapsed_time(end)
            actual = digest(cpu_output)
        else:
            elapsed = host_ns
            actual = digest(output)
        if actual != expected:
            raise ValueError('timed final pixels differ from float-derived control')
        print('SAMPLE ' + json.dumps(dict(iteration=iteration + 1, latency_ns=elapsed,
              allocated_bytes=allocated, reserved_bytes=reserved, d2h_ms=d2h_ms)), flush=True)
        print(f'latency_ns={elapsed}', flush=True)
        del output
    print('IDENTITY ' + json.dumps(dict(name='dscf', device=a.device, bits=a.bits,
          lifetime=a.lifetime, shape=[1, 3, 8192, 8192], input_sha256=sha(a.input),
          output_sha256=expected, float_output_sha256=float_hash, torch=torch.__version__,
          compute='FP32', output_layout='NCHW')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--lifetime', choices=['original', 'compact'], required=True)
    p.add_argument('--bits', type=int, choices=[0, 8, 16], required=True)
    p.add_argument('--expected-float', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--samples', type=int, default=2)
    main(p.parse_args())
