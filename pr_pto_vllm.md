# PR: Enable PTO MegaGDN kernel for Qwen3.5 / Qwen3.6 prefill

**Target**: `vllm-project/vllm-ascend` main branch

---

## Summary

This PR integrates Bisheng-JIT-compiled PTO (Parallel Tile Operator) kernels
for the **chunk GatedDeltaNet (GDN)** recurrent layer used in Qwen3.5 and
Qwen3.6 models, replacing the default Triton implementation during the prefill
phase.

Key outcomes:
- **7–25% TTFT reduction** on Ascend 910B, depending on model size and sequence length.
- **Zero accuracy regression**: WikiText PPL and MMLU scores match Triton to floating-point equivalence.
- **Decode unaffected**: the PTO path only activates for prefill (zero `initial_state`); decode transparently falls back to Triton.
- **Opt-in**: disabled by default; requires `VLLM_ASCEND_PTO_CHUNK_GDN=1`.

---

## Background

Qwen3.5 (0.8B – 9B) and Qwen3.6 (27B, 35B-A3B-MoE) use a **GatedDeltaNet**
recurrent layer that dominates prefill cost at long context lengths.  The
existing Triton-based implementation runs with `chunk_size=64` and BF16
arithmetic.  The new PTO kernels run with `chunk_size=128` in FP16 on the
Ascend AI-Core (vector unit), and optionally fuse all six pipeline stages into
a single kernel launch (**megakernel**).

### Pipeline stages

```
Q, K, V, g, β, cu_seqlens
│
├── chunk_cumsum       prefix cumulative sum of gate logits g
├── scaled_dot_kkt     intra-chunk K^T K attention matrix A
├── tri_inverse        lower-triangular solve A_inv = tril(A)^{-1} (CubeCore)
├── wy_fast            woodbury updates W, U from K, V, β, A_inv
├── chunk_h            inter-chunk recurrent state S and V'
└── chunk_o            output O = Q (K^T V' + S)
```

The **megakernel** fuses all six stages into a single NPU dispatch.

---

## Changes

### New: `csrc/pto_chunk_gdn/`

Eight C++ kernel sources compiled by Bisheng at runtime (or optionally via
CMake):

| File | Description |
|---|---|
| `chunk_cumsum.cpp` | Prefix cumulative sum of gate logits |
| `scaled_dot_kkt.cpp` | Intra-chunk attention matrix (K^T K with gating) |
| `wy_fast.cpp` | Woodbury W/U updates |
| `chunk_h.cpp` | Inter-chunk recurrent state |
| `chunk_o.cpp` | Final output projection |
| `tri_inverse.cpp` + `tri_inverse_impl.cpp` | Lower-triangular matrix inverse (CubeCore) |
| `mega_kernel.cpp` | Fused megakernel (all six stages) |
| `include/kernel_utils.h` | Shared AI-Core utility templates |

### New: `csrc/pto_chunk_gdn/CMakeLists.txt`

Optional cmake target `BUILD_PTO_CHUNK_GDN` (default `OFF`).  When enabled,
pre-compiles kernels for common Qwen model configurations at build time.

```bash
cmake -S csrc -B csrc/build -DBUILD_PTO_CHUNK_GDN=ON
cmake --build csrc/build --target pto_chunk_gdn_kernels -j8
```

### New: `csrc/third_party/pto-isa` (git submodule)

PTO ISA header library required by the C++ kernels.  Added as a submodule at
the same level as the existing `catlass` submodule:
```
csrc/third_party/pto-isa  ← new (gitcode.com/cann/pto-isa.git)
csrc/third_party/catlass   ← existing
```

### New: `vllm_ascend/ops/pto_chunk_gdn/`

Python package providing:

| Module | Description |
|---|---|
| `compile.py` | Bisheng JIT compilation; resolves `PTO_LIB_PATH`; caches `.so` under `kernels/compiled_lib/` |
| `kernel_libs.py` | `ctypes` wrappers for all six staged kernels |
| `fast_inverse.py` | `solve_tril` (lower-triangular matrix inverse) using CubeCore |
| `mega_kernel.py` | Fused megakernel launcher |
| `chunk_gated_delta_wrapper.py` | Drop-in replacement for `chunk_gated_delta_rule`; handles GQA, fallback to Triton for decode |
| `worker_hook.py` | `apply_pto_gdn_patch()` — monkey-patches `chunk_gated_delta_rule` at worker startup |

### Modified: `vllm_ascend/envs.py`

Two new environment variables:

```python
VLLM_ASCEND_PTO_CHUNK_GDN         # bool, default False; enable PTO kernels
VLLM_ASCEND_PTO_CHUNK_GDN_MEGAKERNEL  # bool, default False; use fused megakernel
```

### Modified: `vllm_ascend/patch/worker/__init__.py`

Native activation hook — runs immediately after Triton patches, before model
loading:

```python
if not is_310p():
    if envs.VLLM_ASCEND_PTO_CHUNK_GDN:
        from vllm_ascend.ops.pto_chunk_gdn.worker_hook import apply_pto_gdn_patch
        apply_pto_gdn_patch()
```

### Modified: `vllm_ascend/ops/gdn.py`

Changed static import to module-level access so the worker hook patch applies
correctly:

```python
# Before
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule

# After
import vllm_ascend.ops.triton.fla.chunk as _fla_chunk_mod
# ...
(out, state) = _fla_chunk_mod.chunk_gated_delta_rule(...)
```

### New: `docs/source/user_guide/feature_guide/enable_megagdn_kernel.md`

User guide covering quick-start, performance numbers, accuracy verification
(with lm-eval commands), CMake pre-compilation, and troubleshooting.

---

## Performance Results

Measured on Ascend 910B4 with Qwen3.5-0.8B (BF16, eager mode, single device):

| Backend | 512 TTFT | 1024 TTFT | 4096 TTFT |
|---|---|---|---|
| Triton (default)  | 127 ms | 126 ms | 135 ms |
| PTO megakernel    | 111 ms | 112 ms | 124 ms |
| **Speedup**       | **1.15×** | **1.12×** | **1.09×** |

For larger models (Qwen3.6-35B-A3B-MoE) where GDN layers dominate, speedups
reach **~25% at 4k+ tokens**.

---

## Accuracy Results (Qwen3.5-0.8B)

256-document wikitext subset + 6-subject MMLU:

| Backend | WikiText PPL ↓ | MMLU acc ↑ |
|---|---|---|
| Triton (default) | 20.93 | 48.9% |
| PTO megakernel   | 20.93 | 48.9% |

Zero accuracy difference.

---

## Testing

```bash
# Unit test: kernel correctness (H=16, C=128)
ASCEND_RT_VISIBLE_DEVICES=0 \
GDN_NPU_DEVICE=npu:0 \
python tests/ops/test_pto_chunk_gdn.py --quick

# E2E: all three backends produce identical outputs
ASCEND_RT_VISIBLE_DEVICES=0 \
python tests/ops/test_pto_chunk_gdn.py --e2e

# Prefill benchmark
ASCEND_RT_VISIBLE_DEVICES=0 \
python benchmarks/ops/bench_pto_chunk_gdn.py \
    --model /path/to/Qwen3.5-0.8B \
    --seq-len 512 1024 4096 \
    --cases triton ascend_pto ascend_pto_mega

# lm-eval accuracy
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_ASCEND_PTO_CHUNK_GDN=1 \
VLLM_ASCEND_PTO_CHUNK_GDN_MEGAKERNEL=1 \
python -m lm_eval --model vllm \
    --model_args "pretrained=/path/to/Qwen3.5-0.8B" \
    --tasks wikitext --limit 256
```

---

## Limitations

- Single-device only (TP=1, EP=1).  Tensor parallel and expert parallel will
  be addressed in follow-up PRs.
- Chunk size fixed at C=128 (matches optimal tiling for Ascend 910B AI-Core).
- 910C/A3 compiles with the same code but is not directly tested in CI.

---

## Notes for Reviewers

- The implementation is fully in-tree with no external dependencies beyond
  the `csrc/third_party/pto-isa` submodule (same hosting as `catlass`).
- PTO env vars starting with `VLLM_ASCEND_` are registered in
  `vllm_ascend/envs.py`; the "Unknown vLLM environment variable" warnings from
  vLLM core are harmless.
- The JIT cache directory (`kernels/compiled_lib/`) is listed in `.gitignore`
  except for the `.gitkeep` placeholder.
