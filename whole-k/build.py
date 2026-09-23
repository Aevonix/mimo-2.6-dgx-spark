"""Build a separately named operation; never replaces or patches installed vLLM."""
from pathlib import Path
import argparse
import hashlib
import json
import os


def build(build_dir, verbose=False):
    import torch
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["patched_sha256"].items():
        assert hashlib.sha256((root / "src" / name).read_bytes()).hexdigest() == expected, name
    assert torch.cuda.get_device_capability() == (12, 1), "This isolated build is qualified only for SM121"
    directory = Path(build_dir)
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
    sources = root / "src/csrc/libtorch_stable/moe/marlin_moe_wna16"
    output = load(name="mimo_marlin_whole_k_v1", sources=[str(sources / "ops.cu")] +
                  [str(p) for p in sorted(sources.glob("whole_k_m*.cu"))],
                  extra_include_paths=[str(root / "src/csrc")],
                  extra_cflags=["-O3", "-std=c++20", "-DUSE_CUDA", "-DTORCH_TARGET_VERSION=0x020B000000000000ULL"],
                  extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr",
                      "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
                      "-DUSE_CUDA", "-static-global-template-stub=false",
                      "-DMARLIN_NAMESPACE_NAME=mimo_marlin_whole_k_v1",
                      "-Dmoe_wna16_marlin_gemm=mimo_marlin_whole_k_gemm_v1"],
                  build_directory=str(directory), verbose=verbose,
                  is_python_module=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"library": str(build(args.build_dir, args.verbose))}), flush=True)
