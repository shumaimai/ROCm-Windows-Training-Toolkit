from __future__ import annotations

# Copyright 2026 Shuhei
# Licensed under the Apache License, Version 2.0.

import contextlib
import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "extensions" / "rocm_windows_kernels.cpp",
    ROOT / "extensions" / "rocm_windows_kernels.cu",
)
_loaded_info: dict[str, str] = {}


def _find_rocm_root() -> Path:
    configured = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    candidates = [Path(configured)] if configured else []
    spec = importlib.util.find_spec("_rocm_sdk_devel")
    if spec and spec.submodule_search_locations:
        candidates.extend(Path(path) for path in spec.submodule_search_locations)
    if sys.platform == "win32":
        candidates.append(Path(sys.executable).parents[1] / "Lib" / "site-packages" / "_rocm_sdk_devel")
    else:
        rocm_sdk = shutil.which("rocm-sdk")
        if rocm_sdk:
            result = subprocess.run(
                [rocm_sdk, "path", "--root"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                candidates.append(Path(result.stdout.strip()))
        try:
            from torch.utils.cpp_extension import ROCM_HOME

            if ROCM_HOME:
                candidates.append(Path(ROCM_HOME))
        except ImportError:
            pass
        candidates.extend((Path("/opt/rocm"), *sorted(Path("/opt").glob("rocm-*"), reverse=True)))
    for candidate in candidates:
        windows_compiler = candidate / "lib" / "llvm" / "bin" / "amdclang-cl.exe"
        linux_compiler = candidate / "bin" / "hipcc"
        linux_clang = candidate / "lib" / "llvm" / "bin" / "clang++"
        if windows_compiler.is_file() or linux_compiler.is_file() or linux_clang.is_file():
            return candidate.resolve()
    raise RuntimeError("ROCm development SDK was not found in this Python environment")


def _gpu_arch() -> str:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("An available ROCm GPU is required")
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


def _build_identity(arch: str) -> str:
    digest = hashlib.sha256()
    for source in SOURCES:
        digest.update(source.read_bytes())
    digest.update(torch.__version__.encode())
    digest.update(str(torch.version.hip).encode())
    digest.update(arch.encode())
    digest.update(sys.platform.encode())
    return digest.hexdigest()[:16]


@contextlib.contextmanager
def _build_environment(rocm_root: Path, arch: str):
    llvm_bin = rocm_root / "lib" / "llvm" / "bin"
    updates = {
        "ROCM_HOME": str(rocm_root),
        "ROCM_PATH": str(rocm_root),
        "HIP_HOME": str(rocm_root),
        "PYTORCH_ROCM_ARCH": arch,
        "PATH": f"{rocm_root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    if sys.platform == "win32":
        updates.update(
            CC=str(llvm_bin / "amdclang-cl.exe"),
            CXX=str(llvm_bin / "amdclang-cl.exe"),
        )
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_extension():
    arch = _gpu_arch()
    rocm_root = _find_rocm_root()
    identity = _build_identity(arch)
    module_name = f"rocm_windows_training_{identity}"
    build_dir = ROOT / "artifacts" / "hip-extension" / identity
    build_dir.mkdir(parents=True, exist_ok=True)
    binaries = [*build_dir.glob(f"{module_name}*.pyd"), *build_dir.glob(f"{module_name}*.so")]
    binary = binaries[0] if binaries else None

    if binary is not None and binary.is_file():
        spec = importlib.util.spec_from_file_location(module_name, binary)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load extension binary: {binary}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        with _build_environment(rocm_root, arch):
            from torch.utils.cpp_extension import load

            device_libs = rocm_root / "lib" / "llvm" / "amdgcn" / "bitcode"
            extra_cuda_cflags = ["-O3"]
            if device_libs.is_dir():
                extra_cuda_cflags.append(f"--rocm-device-lib-path={device_libs}")
            extra_ldflags: list[str] = []
            if sys.platform != "win32":
                rocm_lib = rocm_root / "lib"
                extra_ldflags.extend((f"-L{rocm_lib}", f"-Wl,-rpath,{rocm_lib}"))
            module = load(
                name=module_name,
                sources=[str(source) for source in SOURCES],
                build_directory=str(build_dir),
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                verbose=False,
            )
            binary = Path(module.__file__).resolve()

    _loaded_info.update(
        backend="hip",
        architecture=arch,
        platform=platform.system().lower(),
        build_identity=identity,
        binary=str(binary) if binary is not None else "",
    )
    return module


def extension_info() -> dict[str, str]:
    return dict(_loaded_info)


def make_autograd_function():
    extension = load_extension()

    class SsdHip(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dt, a, b, c, d, z, dt_bias, seq_idx):
            empty_seq = torch.empty(0, device=x.device, dtype=torch.int32)
            output, states = extension.forward(
                x.contiguous(), dt.contiguous(), a.float().contiguous(),
                b.contiguous(), c.contiguous(), d.float().contiguous(),
                z.contiguous(), dt_bias.float().contiguous(),
                seq_idx.contiguous() if seq_idx.numel() else empty_seq,
            )
            ctx.save_for_backward(x, dt, a, b, c, d, z, dt_bias, seq_idx, states)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            x, dt, a, b, c, d, z, dt_bias, seq_idx, states = ctx.saved_tensors
            gradients = extension.backward(
                grad_output.contiguous(), states, x, dt, a.float().contiguous(),
                b, c, d.float().contiguous(), z, dt_bias.float().contiguous(), seq_idx,
            )
            return (*gradients, None)

    return SsdHip.apply


def make_conv_autograd_function():
    extension = load_extension()

    class CausalConvHip(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, seq_idx):
            empty_seq = torch.empty(0, device=x.device, dtype=torch.int32)
            ctx.save_for_backward(x, weight, seq_idx)
            return extension.conv_forward(
                x.contiguous(), weight.contiguous(),
                seq_idx.contiguous() if seq_idx.numel() else empty_seq,
            )

        @staticmethod
        def backward(ctx, grad_output):
            x, weight, seq_idx = ctx.saved_tensors
            grad_x, grad_weight = extension.conv_backward(
                grad_output, x, weight, seq_idx, ctx.needs_input_grad[1]
            )
            return grad_x, grad_weight if ctx.needs_input_grad[1] else None, None

    return CausalConvHip.apply
