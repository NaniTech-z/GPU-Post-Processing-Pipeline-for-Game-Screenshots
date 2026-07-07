import os
import time
import numpy as np
import torch

from image_utils import load_image_to_array, save_image
from cpu import (
    change_brightness_cpu,
    change_contrast_cpu,
    change_gamma_contrast_cpu,
)
from gpu import (
    change_brightness_gpu_wrapper,
    change_contrast_gpu_wrapper,
    change_gamma_contrast_gpu_wrapper,
)

def _time_function(fn, image_array, *args, repeats=3, use_gpu=False):
    fn(image_array.copy(), *args)
    if use_gpu:
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(image_array.copy(), *args)
        if use_gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.mean(times))


def benchmark_cpu_vs_gpu(input_path, output_dir="images/output", repeats=3, print_results=True):
    image = load_image_to_array(input_path)
    os.makedirs(output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please make sure your GPU is configured correctly.")

    operations = [
        ("brightness", change_brightness_cpu, change_brightness_gpu_wrapper, 100),
        ("contrast", change_contrast_cpu, change_contrast_gpu_wrapper, 1000),
        ("gamma", change_gamma_contrast_cpu, change_gamma_contrast_gpu_wrapper, 0.5),
    ]

    results = []

    if print_results:
        print("Benchmarking CPU vs GPU")
        print("-" * 50)

    for name, cpu_fn, gpu_fn, value in operations:
        cpu_time = _time_function(cpu_fn, image, value, repeats=repeats, use_gpu=False)
        gpu_time = _time_function(gpu_fn, image, value, repeats=repeats, use_gpu=True)

        cpu_output_path = os.path.join(output_dir, f"cpu_{name}.png")
        gpu_output_path = os.path.join(output_dir, f"gpu_{name}.png")

        cpu_processed = cpu_fn(image.copy(), value)
        gpu_processed = gpu_fn(image.copy(), value)

        save_image(cpu_output_path, cpu_processed)
        save_image(gpu_output_path, gpu_processed)

        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            summary = f"{name}: CPU {cpu_time * 1000:.3f} ms, GPU {gpu_time * 1000:.3f} ms, GPU is {speedup:.2f}x faster"
        else:
            slowdown = gpu_time / cpu_time
            summary = f"{name}: CPU {cpu_time * 1000:.3f} ms, GPU {gpu_time * 1000:.3f} ms, CPU is {slowdown:.2f}x faster"

        if print_results:
            print(summary)

        results.append({
            "name": name,
            "cpu_time_ms": cpu_time * 1000,
            "gpu_time_ms": gpu_time * 1000,
            "cpu_output": cpu_output_path,
            "gpu_output": gpu_output_path,
        })

    return results