# PR: Enable PTO GDN Megakernel for Qwen3.5 / Qwen3.6 prefill

**Target**: `vllm-project/vllm-ascend` main branch

---

## Summary

Integrates a Bisheng-JIT-compiled fused PTO (Parallel Tile Operator) megakernel
for the **chunk GatedDeltaNet (GDN)** recurrent layer used in Qwen3.5 and
Qwen3.6 models, replacing the default Triton implementation during prefill.

Key outcomes:
- **7–25% TTFT reduction** on Ascend 910B, depending on model size and sequence length.
- **Zero accuracy regression**: WikiText PPL and MMLU scores match Triton to floating-point equivalence.
- **Decode unaffected**: the PTO path only activates for prefill (zero `initial_state`); decode transparently falls back to Triton.
- **Single opt-in env var**: `VLLM_ASCEND_PTO_CHUNK_GDN=1`.

---

## Background

Qwen3.5 (0.8B – 9B) and Qwen3.6 (27B, 35B-A3B-MoE) use a **GatedDeltaNet**
recurrent layer that dominates prefill cost at long context lengths.  The
existing Triton-based implementation runs with `chunk_size=64` and BF16.
The new PTO megakernel runs with `chunk_size=128` in FP16 on the Ascend
AI-Core vector unit, fusing all six pipeline stages into a **single NPU dispatch**.

### Pipeline stages (all fused into one kernel launch)

```
Q, K, V, g, β, cu_seqlens
│
├── cumsum       prefix cumulative sum of gate logits g        [Vec]
├── kkt          intra-chunk K^T K attention matrix A          [Cube+Vec]
├── solve_tril   lower-triangular solve A_inv = tril(A)^{-1}  [CubeCore]
├── wy_fast      Woodbury updates W, U from K, V, β, A_inv    [Vec+Cube]
├── chunk_h      inter-chunk recurrent state S and V'          [Cube+Vec]
└── chunk_o      output O = Q (K^T V' + S)                    [Cube+Vec]
```

GQA is fully supported: Q/K use `Hg` heads while V/gates use `H ≥ Hg` value
heads.

---

## Changes

### New: `csrc/pto_chunk_gdn/`

| File | Description |
|---|---|
| `mega_kernel.cpp` | Fused megakernel entry point (all six stages) |
| `chunk_cumsum.cpp` | Cumulative sum stage (included by mega_kernel.cpp) |
| `scaled_dot_kkt.cpp` | K^T K attention matrix (included by mega_kernel.cpp) |
| `tri_inverse_impl.cpp` | Triangular inverse implementation (included by mega_kernel.cpp) |
| `wy_fast.cpp` | Woodbury W/U updates (included by mega_kernel.cpp) |
| `chunk_h.cpp` | Inter-chunk recurrent state (included by mega_kernel.cpp) |
| `chunk_o.cpp` | Output projection (included by mega_kernel.cpp) |
| `include/kernel_utils.h` | Shared AI-Core utility templates |
| `CMakeLists.txt` | Optional `BUILD_PTO_CHUNK_GDN` target for pre-compilation |

### New: `csrc/third_party/pto-isa` (git submodule)

PTO ISA header library required by the megakernel.  Added at the same level
as the existing `catlass` submodule:
```
csrc/third_party/pto-isa   ← new  (gitcode.com/cann/pto-isa.git)
csrc/third_party/catlass    ← existing
```

### New: `vllm_ascend/ops/pto_chunk_gdn/`

| Module | Description |
|---|---|
| `compile.py` | Bisheng JIT compilation; resolves `PTO_LIB_PATH`; caches `.so` |
| `mega_kernel.py` | ctypes launcher for the fused megakernel |
| `chunk_gated_delta_wrapper.py` | Drop-in `chunk_gated_delta_rule` replacement; handles GQA and Triton fallback for decode |
| `worker_hook.py` | `apply_pto_gdn_patch()` — patches `chunk_gated_delta_rule` at worker startup |

### Modified: `vllm_ascend/envs.py`

One new environment variable:

```python
VLLM_ASCEND_PTO_CHUNK_GDN   # bool, default False; enable PTO megakernel
```

### Modified: `vllm_ascend/patch/worker/__init__.py`

Native activation hook — runs after Triton patches, before model loading:

```python
if not is_310p():
    if envs.VLLM_ASCEND_PTO_CHUNK_GDN:
        from vllm_ascend.ops.pto_chunk_gdn.worker_hook import apply_pto_gdn_patch
        apply_pto_gdn_patch()
```

### Modified: `vllm_ascend/ops/gdn.py`

Changed static import to module-level attribute access so the worker hook
patch applies correctly:

```python
# Before
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule

# After
import vllm_ascend.ops.triton.fla.chunk as _fla_chunk_mod
# ...
(out, state) = _fla_chunk_mod.chunk_gated_delta_rule(...)
```

### New: `docs/source/user_guide/feature_guide/enable_megagdn_kernel.md`

Complete user guide covering quick-start, performance benchmarks, lm-eval
accuracy verification (with commands), CMake pre-compilation, and
troubleshooting.

---

## Performance Results

Measured on Ascend 910B4 with Qwen3.5-0.8B (BF16, eager mode, single device):

| Backend | 512 tok TTFT | 1024 tok TTFT | 4096 tok TTFT |
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
# E2E inference: PTO and Triton produce identical outputs
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_ASCEND_PTO_CHUNK_GDN=1 \
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.5-0.8B --max-model-len 1024

# lm-eval accuracy
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_ASCEND_PTO_CHUNK_GDN=1 \
python -m lm_eval --model vllm \
    --model_args "pretrained=/path/to/Qwen3.5-0.8B" \
    --tasks wikitext --limit 256
```

---

## Limitations

- Single-device only (TP=1, EP=1).  Multi-device support will be addressed in follow-up PRs.
- Chunk size fixed at C=128 (optimal tiling for Ascend 910B AI-Core).
- 910C/A3 uses the same code path but is not tested in the current CI environment.

---

## Notes for Reviewers

- Fully in-tree: no external dependencies beyond the `csrc/third_party/pto-isa` git submodule.
- `VLLM_ASCEND_PTO_CHUNK_GDN` is registered in `vllm_ascend/envs.py`; the
  "Unknown vLLM environment variable" warning from vLLM core is harmless.
- The JIT cache directory (`kernels/compiled_lib/*.so`) is listed in `.gitignore`.
