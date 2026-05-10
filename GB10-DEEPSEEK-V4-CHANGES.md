# DeepSeek V4 on GB10 - Changes Documentation

**Hardware**: NVIDIA GB10 (DGX Spark, CC 12.1, 128GB shared memory)
**CUDA**: 13.0 at `/usr/local/cuda-13.0/`, symlink at `/usr/local/cuda`
**Model**: `antirez/deepseek-v4-gguf` (DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf)
**Build**: `/home/pi/llama.cpp/build-cuda/`

---

## Commits

| Commit | Description |
|--------|-------------|
| `9aa1ebd87` | fix(cuda/fattn): add ncols2=1 template instances for DKQ=512 |
| `a4f45fc87` | fix(cuda/fattn): allow FA kernels for DKQ>256 without GQA stride alignment |
| `328cddf71` | fix: reserve compute buffers for DeepSeek V4 non-zero positions during autoregressive decode |
| `dcf74b4dd` | fix: offload input (embedding) layer to GPU to reduce splits |
| `74d73e3a2` | logging: show only not-offloaded ops (skip REPEAT/VIEW/TRANSPOSE) |
| `(next)` | scripts: add Q8_0→MXFP4 conversion tool and documentation |

---

## Modified Files

### 1. `/home/pi/llama.cpp/ggml/src/ggml-cuda/fattn.cu`

**Commit**: `a4f45fc87`

**Change 1** - Removed GQA stride alignment requirement for DKQ=512/576 dispatch:

In `ggml_cuda_get_best_fattn_kernel`, removed the early-exit guards from `case 512:` and `case 576:` that blocked kernel selection when `gqa_opt_applies` was false (i.e., when `K->ne[1] % FATTN_KQ_STRIDE != 0` during initial prompt processing).

**Before**:
```cpp
case 512: {
    if (!gqa_opt_applies) {
        return BEST_FATTN_KERNEL_NONE;  // REJECT - fatal error
    }
    // ... kernel selection
}
```

**After**:
```cpp
case 512: {
    // Removed: if (!gqa_opt_applies) { return BEST_FATTN_KERNEL_NONE; }
    // Allow fallback to ncols2=1 path when GQA opt doesn't apply
    // ... kernel selection continues
}
```

Same for `case 576:`.

**Change 2** - Replaced `GGML_ABORT` with ncols2=1 fallback:

In `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`, replaced the fatal abort for DKQ > 256 without GQA optimization with a fallback to ncols2=1:

**Before**:
```cpp
} else {
    GGML_ABORT("fatal error");
}
```

**After**:
```cpp
} else {
    // For DKQ > 256 without GQA optimization, fall back to ncols2=1
    // This processes one group at a time - slower but works for any sequence length
    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}
```

**Why**: During initial prompt processing, `K->ne[1]` (seq_len) = 1, so `1 % 256 != 0` → `gqa_opt_applies = false`. The old code rejected these cases with `BEST_FATTN_KERNEL_NONE` → `GGML_ABORT`. The fallback allows processing one GQA group at a time.

---

### 2. `/home/pi/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh`

**Commit**: `9aa1ebd87`

**Change**: Added explicit template instantiations for DKQ=512, ncols2=1:

```cpp
// ncols2=1 instantiations for DKQ=512 - fallback path when use_gqa_opt is false
// (e.g., during initial prompt processing where K->ne[1] % FATTN_KQ_STRIDE != 0)
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  1);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  1);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 32,  1);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 64,  1);
```

**Why**: The `switch_ncols1<512, 512, 1>` call from the fallback path needs explicit template instantiations. Without them, the compiler generates implicit (and potentially broken) kernels. The existing instantiations only covered ncols2=4 and ncols2=8, plus ncols2=1 for smaller DKQ values (64-256).

---

### 3. Template Instance Files

**Commit**: `9aa1ebd87`

Added `DECL_FATTN_MMA_F16_CASE(512, 512, X, 1)` to four template-instance .cu files:

| File | Added Line |
|------|------------|
| `template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu` | `DECL_FATTN_MMA_F16_CASE(512, 512, 8, 1);` |
| `template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu` | `DECL_FATTN_MMA_F16_CASE(512, 512, 16, 1);` |
| `template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu` | `DECL_FATTN_MMA_F16_CASE(512, 512, 32, 1);` |
| `template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu` | `DECL_FATTN_MMA_F16_CASE(512, 512, 64, 1);` |

**Why**: These template-instance .cu files are compiled by CMake to generate the actual template instantiations. Without adding the DKQ=512 entries here, the linker can't find the symbols even if declared in the .cuh.

---

### 4. Other Commits (Not Modified by Us, But Relevant)

**`328cddf71`**: Reserve compute buffers for DeepSeek V4 non-zero positions during autoregressive decode. Fixes issues where the model couldn't handle certain tensor positions during generation.

**`dcf74b4dd`**: Offload input (embedding) layer to GPU to reduce splits. Improves performance by keeping more computation on GPU.

**`74d73e3a2`**: Logging cleanup - show only not-offloaded ops (skip REPEAT/VIEW/TRANSPOSE). Reduces log noise.

---

## Key Architecture Details

### GB10 Compute Capability
- `GGML_CUDA_CC_DGX_SPARK` = 1210 (defined in `common.cuh`)
- `GGML_CUDA_CC_BLACKWELL` = 1200 (defined in `common.cuh`)
- GB10 satisfies `turing_mma_available(cc)` → routes to `BEST_FATTN_KERNEL_MMA_F16`
- GB10 does NOT satisfy `ggml_cuda_should_use_wmma_fattn(cc)` (WMMA only enabled for VOLTA, RDNA3, MTHREADS, CDNA, RDNA4)

### DeepSeek V4 Flash Parameters
- Head dim Q/K (DKQ) = 576
- Head dim V (DV) = 512
- GQA ratio = 64 (or 20 for GLM 4.7 Flash variant)
- `FATTN_KQ_STRIDE` = 256

### Kernel Selection Path (GB10 + DeepSeek V4)

**For DKQ=576 (Flash heads):**
```
ggml_cuda_get_best_fattn_kernel(dst)
  → turing_mma_available(1210) = true
  → Q->ne[0] = 576 → case 576: block
  → use_gqa_opt = mask && max_bias == 0.0f (always true during decode)
  → gqa_ratio = 64, 64 % 16 == 0 → switch_ncols1<576, 512, 16>
  → ncols2=16, Q->ne[1]=1 (autoregressive) → mma_f16_case<576, 512, 1, 16>
```

**For DKQ=512 (non-Flash heads, other models):**
```
ggml_cuda_get_best_fattn_kernel(dst)
  → turing_mma_available(1210) = true
  → Q->ne[0] = 512 → case 512: block
  → switch_ncols2<512, 512>
  → When use_gqa_opt = false (K->ne[1] % 256 != 0):
      → ncols2=1 → switch_ncols1<512, 512, 1>
      → mma_f16_case<512, 512, ncols1, 1> (requires explicit instantiation)
```

**Key insight**: The ncols2=1 instantiations we added are NOT used for DeepSeek V4 (which uses DKQ=576 with ncols2=16). They are only used for OTHER models with DKQ=512.

---

## Build Configuration

```bash
cmake -S . -B build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DBUILD_TESTS=OFF

cd build-cuda && make -j$(nproc)
```

**Current generation speed**: ~14.3 t/s (DeepSeek V4 Flash)
**Current prompt speed**: ~26.4 t/s
**Gemma-4-26B-A4B-it benchmark**: 49.9 t/s (CUDA backend works correctly)

**IMPORTANT**: The SQRTSOFTPLUS gating function prevents MoE fusion on GPU, causing 43 CPU splits per token. The MoE routing graph structure differs from SOFT_MAX/SIGMOID:
- llama-graph.cpp creates: UNARY(SOFTPLUS) → SQRT → ARGSORT → RESHAPE → GET_ROWS
- CUDA fusion expects: SOFT_MAX → RESHAPE → ARGSORT → VIEW → GET_ROWS
- The SQRTSOFTPLUS path has different node ordering (ARGSORT created before RESHAPE in llama-graph.cpp)
- The topological sort may reorder nodes differently for SQRTSOFTPLUS vs SOFT_MAX
- Implementing SQRTSOFTPLUS fusion requires matching the exact graph structure
- Debug logging needed to trace actual node order and dependencies

---

## MXFP4 Conversion

### Script
`scripts/convert_q8_to_mxfp4.py` — converts all Q8_0 tensors (attention projections, shared experts, output layer) to MXFP4 format in-place.

**Usage:**
```bash
python3 scripts/convert_q8_to_mxfp4.py input.gguf output_mxfp4.gguf
```

**Results on DeepSeek V4 Flash (GB10):**
| Metric | Before (Q8_0 attn) | After (MXFP4 attn) | Change |
|--------|-------------------|-------------------|--------|
| File size | 81 GB | 78 GB | -3.7% |
| VRAM | 82,697 MiB | 79,551 MiB | -3.8% |
| Generation | 15.4 t/s | 20.6 t/s | **+33.8%** |
| Prompt (short) | 30.8 t/s | 31.5 t/s | ~same |

**Kernel breakdown after conversion (nsys, 512 tokens):**
| Kernel | Time | What |
|--------|------|------|
| MXFP4 mat-vec (type=39) | 38.0% | Attention projections, shared FFN, output (was Q8_0 at 38.9%) |
| IQ2_XXS mat-vec (type=16) | 16.2% | MoE gate/up experts |
| FP16 mat-vec | 9.3% | Flash attention score×V |
| Q2_K mat-vec (type=10) | 10.4% | MoE down experts |
| dsv4_hc_split_sinkhorn | 7.0% | Hyper-connection + sinkhorn |
| dsv4_fp8_kv_quantize | 4.9% | KV cache quantization |

Absolute MXFP4 mat-vec time dropped from 25.3→18.4 ms/token. GPU utilization ~82%.

### Why It Works
- MXFP4 halves memory traffic vs Q8_0 (0.5 B/weight vs 1.0 B/weight)
- Mat-vec at batch=1 is memory-bandwidth bound, so less data → directly faster
- For prompt processing (batch>1), Blackwell native FP4 tensor cores activate via the existing mmq path (`mma.sync.aligned.kind::mxf4.block_scale.m16n8k64`)

### Conversion Process
1. Reads original GGUF via `gguf.GGUFReader`
2. For each Q8_0 tensor: dequantize to f32 → quantize to MXFP4
3. Copies all other tensors as-is (IQ2_XXS, Q2_K, F16, F32)
4. Writes new GGUF via `gguf.GGUFWriter` with `use_temp_file=True`

---

## Model Testing

### Command Line Flags
```bash
./build-cuda/bin/llama-cli -m <model.gguf> \
  -p "Hello, how are you?" \
  --no-mmap --direct-io -c 8192 -ngl 99
```

**`--no-mmap --direct-io`**: Required due to RAM constraints on the host system.

**`-c 8192`**: Context window size.

**`-ngl 99`**: Offload all layers to GPU.

### Model Paths
```
# Original (Q8_0 attention):
~/.cache/huggingface/hub/models--antirez--deepseek-v4-gguf/snapshots/.../DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf

# MXFP4 attention:
/home/pi/DeepSeek-V4-Flash-MXFP4-Attn.gguf
```

### Benchmark Commands
```bash
# Generation benchmark (1024 tokens)
echo "write a list reversing algorithm in haskell" | \
  ./build-cuda/bin/llama-cli --no-mmap --direct-io -c 8192 -n 1024 \
  --simple-io -f /dev/stdin -ngl 99 -m /path/to/model.gguf 2>&1 | tail -10

# GPU profiling (512 tokens)
echo "write a list reversing algorithm in haskell" | \
  nsys profile -t cuda -o /tmp/nsys_profile --kill=none \
  ./build-cuda/bin/llama-cli --no-mmap --direct-io -c 8192 -n 512 \
  --simple-io -f /dev/stdin -ngl 99 -m /path/to/model.gguf 2>&1 | tail -5
```

---

## Known Issues / Potential Improvements

### Current Performance Profile
- **Model size (MXFP4 attn)**: ~78 GB (was ~86.7 GB with Q8_0)
- **Generation speed**: 20.6 t/s (+33.8% from 15.4 t/s)
- **GPU utilization**: 82% (18% idle — launch gaps, sync points)
- **CPU nodes**: 2 VIEW no-ops (all other ops on GPU)

### Next Optimization Targets
1. **Speculative decoding**: Process multiple tokens in parallel → fill the 18% GPU idle gap → convert mat-vec to mat-mat for Blackwell FP4 tensor cores
2. **Self-speculative** (`--spec-type ngram-mod`): No draft model needed, ~1.3-1.5x expected speedup
3. **Draft model speculation**: Need model with DeepSeek V4-compatible tokenizer

### FA Kernel Dispatch Path
For DeepSeek V4 (DKQ=576, DV=512, GQA ratio=64):
- **During autoregressive generation**: case 576 → ncols2=16 → mma_f16_case<576, 512, 1, 16>
- **During initial prompt**: case 576 → ncols2=16 → mma_f16_case<576, 512, X, 16> (X varies by seq_len)
- **ncols2=1 instantiations are NOT used for DeepSeek V4** (only for other models with DKQ=512)

### Optimization Opportunities
1. **SQRTSOFTPLUS fusion (PRIMARY)**: The SQRTSOFTPLUS gating function creates 43 CPU splits per token. The MoE routing graph structure is:
   - llama-graph.cpp creates: UNARY(SOFTPLUS) → SQRT → ARGSORT → RESHAPE → GET_ROWS
   - CUDA fusion expects: SOFT_MAX → RESHAPE → ARGSORT → VIEW → GET_ROWS
   - The SQRTSOFTPLUS path has different node ordering (ARGSORT before RESHAPE in creation order, but topological sort may reorder)
   - Implementing SQRTSOFTPLUS fusion requires matching the exact graph structure
   - Adding fusion could significantly improve generation speed by keeping MoE routing on GPU

2. **Graph splits (87)**: Heavy GPU↔CPU data copying. Each layer is a separate split. 44 GPU splits, 43 CPU splits. The CPU splits are primarily from SQRTSOFTPLUS gating.

3. **MoE routing**: The MoE routing selects 6 out of 256 experts for each token. MUL_MAT_ID runs on GPU. ARGSORT and GET_ROWS run on GPU. But SQRTSOFTPLUS gating creates separate CPU splits.

4. **No WMMA path for GB10**: WMMA kernel explicitly excludes head dim 512/576. WMMA is also not enabled for Blackwell.

5. **Stride alignment relaxation**: Consider changing `K->ne[1] % FATTN_KQ_STRIDE == 0` to `K->ne[1] % 1 == 0` (always true) to enable GQA optimization more often.

6. **Alternative model**: `tecprovn/deepseek-v4-flash-gguf` Q3_K_M variant (94GB, Q3_M quantization) might avoid the stride issue with different memory layout.

---

## Instructions for Future Compaction

1. **Always load this file first** when continuing work on GB10 DeepSeek V4.
2. **Key files to reference**:
   - `/home/pi/llama.cpp/ggml/src/ggml-cuda/fattn.cu` - kernel selection logic
   - `/home/pi/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh` - kernel templates
   - `/home/pi/llama.cpp/ggml/src/ggml-cuda/common.cuh` - architecture constants
   - `/home/pi/llama.cpp/ggml/src/ggml-cuda/topk-moe.cu` - MoE routing (potential bottleneck)
   - `/home/pi/llama.cpp/ggml/src/ggml-backend.cpp` - scheduling logic
3. **Git history**: Run `git log --oneline` to see commit order.
4. **Template instances**: Located in `/home/pi/llama.cpp/ggml/src/ggml-cuda/template-instances/` - always add DKQ entries here when adding new instantiations to .cuh.
5. **Performance profile**: 87 graph splits (44 GPU, 43 CPU) with 91 GPU inputs. The MoE routing is likely causing CPU↔GPU data transfers.
