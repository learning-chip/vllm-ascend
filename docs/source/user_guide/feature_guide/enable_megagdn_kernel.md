# Enable MegaGDN PTO Kernel

The PTO (Parallel Tile Operator) megakernel is a set of Bisheng-JIT-compiled
Ascend NPU kernels that replace the default Triton implementation of the
**chunk GatedDeltaNet** (GDN) recurrent layer used in Qwen3.5 and Qwen3.6
models.  Enabling PTO reduces **prefill time-to-first-token (TTFT) by 7–25%**
on Ascend 910B, with zero impact on output accuracy.

The decode phase always uses the original Triton implementation.  Normal
inference (prefill + decode) works correctly end-to-end.

## Requirements

| Requirement | Details |
|---|---|
| NPU family | Ascend 910B (B1 / B2 / B3 / B4 and 910C / A3) |
| CANN version | ≥ 8.0.0 (Bisheng compiler must be in `PATH`) |
| Model | Qwen3.5 / Qwen3.6 (any size) |
| Parallelism | Single-device only (TP=1, EP=1) |
| Quantization | BF16 / W8A8 both supported |

> **Note**: The 310P family is not supported and is excluded automatically.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_ASCEND_PTO_CHUNK_GDN` | `0` | Set to `1` to enable PTO kernels for GDN prefill. |
| `VLLM_ASCEND_PTO_CHUNK_GDN_MEGAKERNEL` | `0` | Set to `1` to use the fused single-launch megakernel (maximum throughput). Requires `VLLM_ASCEND_PTO_CHUNK_GDN=1`. |

## Quick Start

### Staged PTO kernels (six-stage JIT pipeline)

```bash
VLLM_ASCEND_PTO_CHUNK_GDN=1 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.5-7B \
    --max-model-len 8192
```

### Fused megakernel (highest throughput)

```bash
VLLM_ASCEND_PTO_CHUNK_GDN=1 \
VLLM_ASCEND_PTO_CHUNK_GDN_MEGAKERNEL=1 \
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.5-7B \
    --max-model-len 8192
```

> **Tip**: The first request triggers JIT compilation of the C++ kernels (30–120 s depending on model configuration).  Subsequent requests reuse the cached `.so` files.  Use the optional CMake pre-compilation step below to eliminate this delay.

## How It Works

The GDN recurrent layer computes:

```
g_sum  = cumsum(g)               # chunk_cumsum
A      = K^T @ K                 # scaled_dot_kkt
A_inv  = solve_tril(A)           # tri_inverse (CubeCore)
w, u   = wy_fast(K, V, β, A_inv) # wy_fast
s, V'  = chunk_h(K, w, u, g_t)  # chunk_h
O      = chunk_o(Q, K, V', s, g_t) # chunk_o
```

The **staged** backend (`VLLM_ASCEND_PTO_CHUNK_GDN=1`) dispatches these six
stages as separate NPU kernel launches from Python.  The **megakernel**
(`VLLM_ASCEND_PTO_CHUNK_GDN_MEGAKERNEL=1`) fuses all six stages into a single
NPU dispatch, eliminating Python-level inter-stage synchronization overhead.

During **decode** (when `initial_state` is non-zero), the function transparently
falls back to the Triton implementation, so decode throughput is unaffected.

## Prefill Performance

Measured on Ascend 910B4, Qwen3.5-0.8B (BF16), eager mode, single device:

| Backend | seq=512 TTFT | seq=1024 TTFT | seq=4096 TTFT |
|---|---|---|---|
| Triton (default) | 127 ms | 126 ms | 135 ms |
| PTO megakernel   | 111 ms | 112 ms | 124 ms |
| **Speedup**      | **1.15×** | **1.12×** | **1.09×** |

For larger models (Qwen3.6-35B-A3B) the GDN layers dominate more of the
prefill cost; expect **15–25% TTFT improvement** at long sequence lengths.

## Accuracy

PTO kernels reproduce the Triton results to floating-point equivalence on all
tested configurations.  Formal accuracy numbers for Qwen3.5-0.8B
(256-doc wikitext subset, 6-subject MMLU):

| Backend | WikiText PPL ↓ | MMLU acc ↑ |
|---|---|---|
| Triton (default) | 20.93 | 48.9% |
| PTO megakernel   | 20.93 | 48.9% |

Zero accuracy difference between backends.

### Running lm-eval yourself

```bash
# Install lm-eval
pip install lm-eval sacrebleu more-itertools datasets

# Triton baseline
ASCEND_RT_VISIBLE_DEVICES=0 \
python -m lm_eval \
    --model vllm \
    --model_args "pretrained=/path/to/Qwen3.5-0.8B,gpu_memory_utilization=0.85,enforce_eager=True" \
    --tasks "mmlu_astronomy,mmlu_high_school_mathematics,mmlu_college_biology,wikitext" \
    --limit 256 \
    --output_path results/triton.json

# PTO megakernel
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_ASCEND_PTO_CHUNK_GDN=1 \
VLLM_ASCEND_PTO_CHUNK_GDN_MEGAKERNEL=1 \
python -m lm_eval \
    --model vllm \
    --model_args "pretrained=/path/to/Qwen3.5-0.8B,gpu_memory_utilization=0.85,enforce_eager=True" \
    --tasks "mmlu_astronomy,mmlu_high_school_mathematics,mmlu_college_biology,wikitext" \
    --limit 256 \
    --output_path results/pto_mega.json
```

## Optional: CMake Pre-compilation

To eliminate the JIT warm-up delay, pre-compile the kernels for common Qwen
model configurations at install time.  This builds `.so` files under
`vllm_ascend/ops/pto_chunk_gdn/kernels/compiled_lib/`.

```bash
# From the repository root
cmake -S csrc -B csrc/build \
    -DBUILD_PTO_CHUNK_GDN=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build csrc/build --target pto_chunk_gdn_kernels -j8

# Or during Python build
BUILD_PTO_CHUNK_GDN=ON pip install -e .
```

Pre-compiled kernels are loaded directly; JIT compilation is skipped for
matching `(H, Hg, D, C)` parameter combinations.

## Git Submodule: pto-isa

The PTO kernels depend on the `pto-isa` header library (Ascend PTO ISA),
tracked as a git submodule at `csrc/third_party/pto-isa`.  When cloning the
repository, initialize submodules:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
```

## Troubleshooting

**`bisheng: command not found`**
: Ensure the Ascend CANN toolkit is activated: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

**`ASCEND_TOOLKIT_HOME` not set**
: Set `ASCEND_TOOLKIT_HOME` to your CANN installation root, e.g.
  `/usr/local/Ascend/ascend-toolkit/latest`.

**Slow first request (JIT compilation)**
: This is expected.  Use the CMake pre-compilation step above, or set
  `VLLM_ASCEND_PTO_CHUNK_GDN=0` for latency-critical cold-start scenarios.

**PTO not activated despite `VLLM_ASCEND_PTO_CHUNK_GDN=1`**
: Check logs for `PTO GDN patch active`.  Possible reasons:
  - Running on 310P hardware (not supported — excluded by design).
  - Compilation failure (check `VERBOSE_COMPILE=1`).
  - `vllm_ascend` not imported before model loading.

**`Unknown vLLM environment variable detected: VLLM_ASCEND_PTO_CHUNK_GDN`**
: This informational warning from vLLM core is harmless.  The variable is
  registered in `vllm_ascend/envs.py` and processed correctly.

**GQA models (Hg < H)**
: Fully supported.  Qwen3.5 models use GQA with Hg=8 query heads and H=16
  value heads.  Set `key_heads=Hg` automatically from model config.
