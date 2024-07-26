"""Build the C++/CUDA extensions for torchmatch."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

CSRC = Path("sources/torchmatch")
ASSIGNMENT_CUDA_CSRC = CSRC / "assignment" / "cuda"
ASSIGNMENT_CPU_CSRC = CSRC / "assignment" / "cpu"
TRANSPORT_CSRC = CSRC / "transport"

# By default no extensions are compiled — the wheel ships C++ sources and
# compiles them at first import via torch.utils.cpp_extension.load (JIT).
# The JIT path detects the host CPU at runtime (e.g. AVX2/FMA) so users
# get locally-optimal code without a platform-specific wheel.
#
# Set the BUILD_* variables to produce a wheel with precompiled extensions:
#   TORCHMATCH_BUILD_CPU=1        — Jonker-Volgenant CPU kernels
#   TORCHMATCH_BUILD_CUDA=1       — CUDA kernels (also needs CUDA_HOME)
#   TORCHMATCH_BUILD_TRANSPORT=1  — network-simplex exact-EMD CPU kernel
#   TORCHMATCH_BUILD_STABLE_ABI=1 — use LibTorch stable ABI + Py_LIMITED_API
#                                   to produce a cp313-abi3 wheel (CPU only)
BUILD_CPU = os.environ.get("TORCHMATCH_BUILD_CPU") == "1"
BUILD_CUDA = os.environ.get("TORCHMATCH_BUILD_CUDA") == "1"
BUILD_TRANSPORT = os.environ.get("TORCHMATCH_BUILD_TRANSPORT") == "1"
# Compile against PyTorch's stable LibTorch ABI (≥2.10) and Python's
# limited ABI (≥3.13) so the resulting wheel works with any future torch
# and Python version without recompilation. Only valid for CPU extensions;
# nvcc does not support Py_LIMITED_API.
BUILD_STABLE_ABI = os.environ.get("TORCHMATCH_BUILD_STABLE_ABI") == "1"
if BUILD_STABLE_ABI and BUILD_CUDA:
    raise ValueError(
        "TORCHMATCH_BUILD_STABLE_ABI=1 is only valid for CPU-only builds; "
        "nvcc does not support Py_LIMITED_API. Unset TORCHMATCH_BUILD_CUDA."
    )


def _ext_modules():
    if not BUILD_CPU and not BUILD_CUDA and not BUILD_TRANSPORT:
        return []

    import platform

    from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

    # -mavx2 / -mfma are x86-64-only; skip on aarch64 / other ISAs.
    x86_flags = ["-mavx2", "-mfma"] if platform.machine() in ("x86_64", "AMD64") else []
    # Targeting the LibTorch stable API floor (2.10) plus Python's limited ABI
    # (3.13) produces a cp313-abi3 wheel that is forward-compatible with any
    # future torch ≥ 2.10 and Python ≥ 3.13. Only applies to CppExtension;
    # CUDAExtension (nvcc) does not support Py_LIMITED_API.
    stable_abi_flags = (
        [
            "-DPy_LIMITED_API=0x030d0000",
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
        ]
        if BUILD_STABLE_ABI
        else []
    )

    mods = []
    assignment_ops = "ops_stable.cpp" if BUILD_STABLE_ABI else "ops.cpp"
    transport_ops  = "ops_stable.cpp" if BUILD_STABLE_ABI else "ops.cpp"
    transport_emd  = "exact_emd_op_stable.cpp" if BUILD_STABLE_ABI else "exact_emd_op.cpp"

    if BUILD_CPU:
        mods.append(
            CppExtension(
                name="torchmatch._assignment_cpu_impl",
                sources=[
                    str(ASSIGNMENT_CPU_CSRC / "jonker_scalar.cpp"),
                    str(ASSIGNMENT_CPU_CSRC / "jonker_dense_core.cpp"),
                    str(ASSIGNMENT_CPU_CSRC / assignment_ops),
                ],
                include_dirs=[str(ASSIGNMENT_CPU_CSRC)],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17", *x86_flags, *stable_abi_flags],
                },
                py_limited_api=BUILD_STABLE_ABI,
            )
        )

    if BUILD_CUDA and CUDA_HOME is not None:
        mods.append(
            CUDAExtension(
                name="torchmatch._assignment_cuda_impl",
                sources=[
                    str(ASSIGNMENT_CUDA_CSRC / "ops.cpp"),
                    str(ASSIGNMENT_CUDA_CSRC / "dispatch.cu"),
                    str(ASSIGNMENT_CUDA_CSRC / "munkres.cu"),
                    str(ASSIGNMENT_CUDA_CSRC / "hybrid.cu"),
                    str(ASSIGNMENT_CUDA_CSRC / "lawler.cu"),
                ],
                include_dirs=[str(ASSIGNMENT_CUDA_CSRC)],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "-std=c++17"],
                },
            )
        )

    if BUILD_TRANSPORT:
        mods.append(
            CppExtension(
                name="torchmatch._transport_cpu_impl",
                sources=[
                    str(TRANSPORT_CSRC / "matrix" / "cpu" / transport_ops),
                    str(TRANSPORT_CSRC / "matrix" / "cpu" / transport_emd),
                    str(
                        TRANSPORT_CSRC / "matrix" / "cpu" / "exact" / "EMD_wrapper.cpp"
                    ),
                ],
                include_dirs=[
                    str(TRANSPORT_CSRC / "matrix" / "cpu"),
                    str(TRANSPORT_CSRC / "matrix" / "cpu" / "exact"),
                ],
                # WHY -fno-fast-math: the network-simplex pivot selection
                # is numerically sensitive; -ffast-math can break the
                # tie-breaking comparisons and produce wrong pivots.
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17", "-fno-fast-math", *stable_abi_flags],
                },
                py_limited_api=BUILD_STABLE_ABI,
            )
        )

    if BUILD_CUDA and BUILD_TRANSPORT and CUDA_HOME is not None:
        mods.append(
            CUDAExtension(
                name="torchmatch._transport_cuda_impl",
                sources=[
                    str(TRANSPORT_CSRC / "matrix" / "cuda" / "ops.cpp"),
                ],
                include_dirs=[str(TRANSPORT_CSRC / "matrix" / "cuda")],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "-std=c++17"],
                },
            )
        )

    return mods


mods = _ext_modules()
if mods:
    from torch.utils.cpp_extension import BuildExtension

    setup_kwargs: dict = {"ext_modules": mods, "cmdclass": {"build_ext": BuildExtension}}
    if BUILD_STABLE_ABI:
        setup_kwargs["options"] = {"bdist_wheel": {"py_limited_api": "cp313"}}
    setup(**setup_kwargs)
else:
    setup()
