# DeepSeek V4 on DGX Spark (GB10) — Optimization History

## Current Best: 20.6 t/s

MXFP4 attention projections + speculative decoding (WIP).

## Build Optimizations

**Baseline**: build-cuda/ (RelWithDebInfo, shared libs, compression=size, NCCL=ON, GRAPHS=ON)

| Change | tg50 t/s | vs baseline |
|--------|---------:|:-----------|
| Original (Q8_0 attention) | 13.16 | - |
| Release (`-O3` CUDA, static linking, compression=speed) | 14.33 | +8.9% |
| + CUDA graphs OFF | 14.42 | +9.6% |
| **+ SQRTSOFTPLUS fusion fix + I32 REPEAT GPU move** | 15.4 | +17% |
| **+ MXFP4 attention (Q8_0→MXFP4 conversion)** | **20.6** | **+57%** |

**Current**: tg50 = **20.6 t/s** (+57%), --simple-io benchmark

## Why Each Change Matters

### Release build (`-O3` vs `-O2 -g`)
- CUDA kernels compiled at `-O3` vs `-O2` — biggest single gain
- Host code also gets `-O3`
- The prior build was accidentally RelWithDebInfo despite `-DCMAKE_BUILD_TYPE=Release`

### Static linking (`BUILD_SHARED_LIBS=OFF`)
- Eliminates PLT/GOT indirection on ARM for cross-library calls
- All ggml/llama code linked directly into the executable

### compression=speed
- CUDA fatbinary compressed for faster decompression at load time
- Default is `size` (max compression, slowest load)

### NCCL off
- Not needed for single GPU; removes unnecessary init overhead

### CUDA graphs OFF
- MoE models change expert selection every token, causing graph re-capture
- Re-capture overhead outweighs launch overhead savings
- For dense (non-MoE) models, graphs may still help

## Run Command

```bash
/home/pi/llama.cpp/build-cuda-opt/bin/llama-cli \
  -hf antirez/deepseek-v4-gguf -ngl 99 --no-mmap --direct-io \
  -n 50 -p "Hello, how are you?" -c 8192 --simple-io
```

`--no-mmap --direct-io` is required for Grace Blackwell UMA to avoid host memory double-buffering.

## Bottleneck Analysis

The 284B MoE model at IQ2_XXS (2 bpw) is **compute-bound by dequantization**, not memory bandwidth:

- IQ2_XXS dequantization uses DP4A int8 dot-product instructions (not tensor cores)
- Blackwell GB10 has native FP4 MMA but the model is IQ2, not MXFP4
- 6/256 experts active per token means ~6.6B effective params per token
- The dequant kernels are the bottleneck, not memory reads

For comparison: Gemma-4-26B-A4B-it at Q4_K_M hits ~50 t/s on the same hardware — higher-bit quantizations have simpler dequant and better tensor core utilization.
