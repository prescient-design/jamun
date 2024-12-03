#!/usr/bin/env python

import copy

import e3nn
import e3nn.util
import torch
import torch._dynamo.config
import torch._inductor.config
from e3nn import o3
from e3tools.nn import ExperimentalTensorProduct
from torch._inductor.utils import timed

e3nn.set_optimization_defaults(jit_script_fx=False)

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

device = "cuda"

torch.set_float32_matmul_precision("high")

LMAX = 1
CHANNEL = 128
BATCH = 64


def benchmark(fn, args=(), times=10, repeat=10, device="cuda"):
    timings = torch.tensor([timed(fn, args, times, device) for _ in range(repeat)])
    took = torch.median(timings) / times
    return took * 1000


def main():
    irreps = o3.Irreps.spherical_harmonics(LMAX)
    irreps_x = (CHANNEL * irreps).regroup()
    irreps_y = (irreps).regroup()
    irreps_out = (CHANNEL * irreps).regroup()

    tp = o3.FullyConnectedTensorProduct(
        irreps_x, irreps_y, irreps_out, internal_weights=False, shared_weights=False
    ).to(device=device)
    tp_jit_compile = e3nn.util.jit.compile(copy.deepcopy(tp)).to(device=device)

    tp_torch_compiled = torch.compile(
        tp, mode="max-autotune", fullgraph=True, dynamic=True
    )

    tp_experimental = ExperimentalTensorProduct(irreps_x, irreps_y, irreps_out).to(
        device=device
    )

    tp_experimental_compiled = torch.compile(
        tp_experimental,
        mode="max-autotune",
        fullgraph=True,
        dynamic=True,
    )

    print(f"{tp.weight_numel=}")
    print(f"{tp_experimental.weight_numel=}")

    x = irreps_x.randn(BATCH, -1).to(device=device)
    y = irreps_y.randn(BATCH, -1).to(device=device)
    weight = torch.randn(BATCH, tp.weight_numel, device=device)

    _ = tp(x, y, weight)
    _ = tp_jit_compile(x, y, weight)
    _ = tp_torch_compiled(x, y, weight)
    _ = tp_experimental(x, y, weight)
    _ = tp_experimental_compiled(x, y, weight)

    print(
        f"{irreps_x} x {irreps_y} -> {irreps_out}: {benchmark(lambda: tp(x, y, weight), times=100, repeat=10)=:.3f}ms"
    )

    print(
        f"{irreps_x} x {irreps_y} -> {irreps_out}: {benchmark(lambda: tp_jit_compile(x, y, weight), times=100, repeat=10)=:.3f}ms"
    )

    print(
        f"{irreps_x} x {irreps_y} -> {irreps_out}: {benchmark(lambda: tp_torch_compiled(x, y, weight), times=100, repeat=10)=:.3f}ms"
    )

    print(
        f"{irreps_x} x {irreps_y} -> {irreps_out}: {benchmark(lambda: tp_experimental(x, y, weight), times=100, repeat=10)=:.3f}ms"
    )

    print(
        f"{irreps_x} x {irreps_y} -> {irreps_out}: {benchmark(lambda: tp_experimental_compiled(x, y, weight), times=100, repeat=10)=:.3f}ms"
    )


if __name__ == "__main__":
    main()
