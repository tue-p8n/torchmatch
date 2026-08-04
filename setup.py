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
#   TORCHMATCH_BUILD_STABLE_ABI=1 — use LibTorch stable ABI; adds
#                                   Py_LIMITED_API for CPU extensions,
#                                   producing a cp313-abi3 wheel. For CUDA,
#                                   only the assignment extension is ported
#                                   so far (see below) — combining with
#                                   TORCHMATCH_BUILD_TRANSPORT=1 is still
#                                   rejected.
BUILD_CPU = os.environ.get("TORCHMATCH_BUILD_CPU") == "1"
BUILD_CUDA = os.environ.get("TORCHMATCH_BUILD_CUDA") == "1"
BUILD_TRANSPORT = os.environ.get("TORCHMATCH_BUILD_TRANSPORT") == "1"
# Compile against PyTorch's stable LibTorch ABI (≥2.10) so the resulting
# wheel works with any future torch version without recompilation. For CPU
# extensions this additionally targets Python's limited ABI (≥3.13) via
# -DPy_LIMITED_API, producing a cp313-abi3 wheel. CUDA extensions can't use
# -DPy_LIMITED_API (nvcc doesn't support it); torchmatch._assignment_cuda_impl
# doesn't need it either, since it links no CPython symbols at all — so
# BUILD_CUDA + BUILD_STABLE_ABI together builds and produces a valid
# cp313-abi3 wheel for the assignment extension alone. That extension is
# stable-ABI end to end: op registration (ops.cpp), tensor manipulation
# (dispatch.cu) and the workspace cache (workspace.cuh, which allocates its
# device buffers as stable tensors through the AOTI C shim) all compile
# under -DTORCH_TARGET_VERSION, which the CUDAExtension block below applies.
# torchmatch._transport_cuda_impl (below) is NOT stable-ABI-ported at all
# yet — it still uses plain ATen (torch/library.h, TORCH_LIBRARY_FRAGMENT) —
# so that specific combination stays blocked until it is.
BUILD_STABLE_ABI = os.environ.get("TORCHMATCH_BUILD_STABLE_ABI") == "1"
if BUILD_CUDA and BUILD_TRANSPORT and BUILD_STABLE_ABI:
    raise ValueError(
        "TORCHMATCH_BUILD_CUDA=1 + TORCHMATCH_BUILD_TRANSPORT=1 + "
        "TORCHMATCH_BUILD_STABLE_ABI=1 is not yet supported: "
        "torchmatch._transport_cuda_impl has not been ported to the "
        "LibTorch stable ABI (unlike torchmatch._assignment_cuda_impl, "
        "which has). Unset TORCHMATCH_BUILD_TRANSPORT, or drop "
        "TORCHMATCH_BUILD_STABLE_ABI, to proceed."
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
    transport_ops = "ops_stable.cpp" if BUILD_STABLE_ABI else "ops.cpp"
    transport_emd = (
        "exact_emd_op_stable.cpp" if BUILD_STABLE_ABI else "exact_emd_op.cpp"
    )

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
        # -DTORCH_TARGET_VERSION goes to both compilers: ops.cpp is built
        # by the host compiler ("cxx") and the .cu files by nvcc, and the
        # flag only enforces the stable-ABI floor on the translation units
        # it reaches. -DPy_LIMITED_API is deliberately not passed (nvcc
        # doesn't support it, and the extension carries no CPython symbols
        # anyway). Every source in this extension — including
        # workspace.cuh's device-memory pool, which allocates stable
        # tensors through the AOTI C shim rather than calling
        # c10::cuda::CUDACachingAllocator — is free of unstable ATen/c10
        # headers, so the flag compiles clean and the resulting .so really
        # does run against any torch >= 2.10.
        torch_target_flag = (
            ["-DTORCH_TARGET_VERSION=0x020a000000000000"] if BUILD_STABLE_ABI else []
        )
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
                    "cxx": ["-O3", *torch_target_flag],
                    "nvcc": ["-O3", "-std=c++17", *torch_target_flag],
                },
            )
        )
        # NOTE: BUILD_CUDA=1 + BUILD_STABLE_ABI=1 together is not exercised
        # by any CI workflow (release.yml only ever sets them separately)
        # or by nix/variants.nix — verified only with a manual local build
        # (see the stable-ABI refactor's task reports under
        # .superpowers/sdd/). Treat changes near this block with extra
        # care until a CI job covers it.

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

    setup_kwargs: dict = {
        "ext_modules": mods,
        "cmdclass": {"build_ext": BuildExtension},
    }
    if BUILD_STABLE_ABI:
        setup_kwargs["options"] = {"bdist_wheel": {"py_limited_api": "cp313"}}
    setup(**setup_kwargs)
else:
    setup()
