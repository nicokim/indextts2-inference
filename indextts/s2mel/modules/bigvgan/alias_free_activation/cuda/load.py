# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib
import subprocess

from torch.utils import cpp_extension

"""
Setting this param to a list has a problem of generating different compilation commands (with diferent order of architectures) and leading to recompilation of fused kernels. 
Set it to empty stringo avoid recompilation and assign arch flags explicity in extra_cuda_cflags below
"""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load():
    import torch

    # Detect GPU compute capability and build for it
    cc_flags = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            arch = f"arch=compute_{major}{minor},code=sm_{major}{minor}"
            flag_pair = ["-gencode", arch]
            if flag_pair not in cc_flags:
                cc_flags.extend(flag_pair)

    # Fallback: add sm_70 and sm_80 if no GPU detected
    if not cc_flags:
        _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
        cc_flags.extend(["-gencode", "arch=compute_70,code=sm_70"])
        if int(bare_metal_major) >= 11:
            cc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ]
            + extra_cuda_flags
            + cc_flags,
            verbose=True,
        )

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    anti_alias_activation_cuda = _cpp_extention_load_helper(
        "anti_alias_activation_cuda", sources, extra_cuda_flags
    )

    return anti_alias_activation_cuda


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
