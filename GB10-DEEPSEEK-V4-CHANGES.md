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

### Model Path
```
/home/pi/.cache/huggingface/hub/models--antirez--deepseek-v4-gguf/snapshots/3af08b96a788790ef6f1d113e5257794622884b8/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

---

## Known Issues / Potential Improvements

### Current Performance Profile
- **Model size**: ~86.7 GB
- **Prompt speed**: 30 t/s
- **Generation speed**: 15.1 t/s
- **Graph splits**: 87 splits (44 GPU, 43 CPU) with 91 GPU inputs
- **CPU nodes**: 45 nodes (MoE routing, KV cache operations)

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
