#!/usr/bin/env python3
"""
Convert Q8_0 tensors to MXFP4 in a GGUF model file.

Reads the original GGUF, converts all Q8_0 tensors (attention projections,
shared experts, output layer) to MXFP4 format, and writes a new GGUF file.

MXFP4 on Blackwell uses native FP4 tensor cores (m16n8k64 MMA via
mma.sync.aligned.kind::mxf4.block_scale), giving ~2x memory bandwidth
savings for generation and ~2-4x speedup for prompt processing.
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

from gguf import (
    GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType,
    dequantize, quantize,
)


def copy_metadata(reader, writer, skip_keys=None):
    """Copy all KV metadata from reader to writer."""
    if skip_keys is None:
        skip_keys = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}

    for key, field in reader.fields.items():
        if key in skip_keys:
            continue
        try:
            val = field.contents()
            types = field.types
            vtype = types[0]

            # Determine the GGUFValueType for the writer
            if vtype == GGUFValueType.STRING:
                writer.add_key_value(key, str(val), GGUFValueType.STRING)
            elif vtype == GGUFValueType.BOOL:
                writer.add_key_value(key, bool(val), GGUFValueType.BOOL)
            elif vtype == GGUFValueType.UINT8:
                writer.add_key_value(key, int(val), GGUFValueType.UINT8)
            elif vtype == GGUFValueType.INT8:
                writer.add_key_value(key, int(val), GGUFValueType.INT8)
            elif vtype == GGUFValueType.UINT16:
                writer.add_key_value(key, int(val), GGUFValueType.UINT16)
            elif vtype == GGUFValueType.INT16:
                writer.add_key_value(key, int(val), GGUFValueType.INT16)
            elif vtype == GGUFValueType.UINT32:
                writer.add_key_value(key, int(val), GGUFValueType.UINT32)
            elif vtype == GGUFValueType.INT32:
                writer.add_key_value(key, int(val), GGUFValueType.INT32)
            elif vtype == GGUFValueType.UINT64:
                writer.add_key_value(key, int(val), GGUFValueType.UINT64)
            elif vtype == GGUFValueType.INT64:
                writer.add_key_value(key, int(val), GGUFValueType.INT64)
            elif vtype == GGUFValueType.FLOAT32:
                writer.add_key_value(key, float(val), GGUFValueType.FLOAT32)
            elif vtype == GGUFValueType.FLOAT64:
                writer.add_key_value(key, float(val), GGUFValueType.FLOAT64)
            elif vtype == GGUFValueType.ARRAY:
                sub_type = types[1] if len(types) > 1 else None
                writer.add_key_value(key, val, GGUFValueType.ARRAY, sub_type)
            else:
                print(f"  WARNING: Unknown value type {vtype} for {key}, skipping")
        except Exception as e:
            print(f"  WARNING: Failed to copy metadata '{key}': {e}", file=sys.stderr)
            continue


def convert_q8_to_mxfp4(input_path, output_path, use_temp_file=True):
    """Convert all Q8_0 tensors in the model to MXFP4."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Reading model: {input_path}")
    reader = GGUFReader(str(input_path))

    arch = "deepseek4"  # known architecture
    # Get architecture from metadata if available
    if "general.architecture" in reader.fields:
        arch = reader.fields["general.architecture"].contents()
        print(f"  Architecture: {arch}")

    print(f"Creating output: {output_path}")
    writer = GGUFWriter(
        str(output_path),
        arch,
        use_temp_file=use_temp_file,
    )

    print("Copying metadata...")
    copy_metadata(reader, writer)

    # Count tensors
    total_tensors = len(reader.tensors)
    q8_count = 0
    other_count = 0
    total_bytes_in = 0
    total_bytes_out = 0

    print(f"\nProcessing {total_tensors} tensors...")
    t_start = time.time()

    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        raw_data = tensor.data
        logical_shape = tensor.shape.tolist() if hasattr(tensor.shape, 'tolist') else list(tensor.shape)
        orig_type = tensor.tensor_type

        if orig_type == GGMLQuantizationType.Q8_0:
            # Convert Q8_0 → MXFP4
            q8_count += 1
            in_bytes = raw_data.nbytes
            total_bytes_in += in_bytes

            # Dequantize Q8_0 to float32
            f32_data = dequantize(raw_data, GGMLQuantizationType.Q8_0)

            # Quantize float32 to MXFP4
            mxfp4_data = quantize(f32_data, GGMLQuantizationType.MXFP4)

            out_bytes = mxfp4_data.nbytes
            total_bytes_out += out_bytes

            writer.add_tensor(
                name, mxfp4_data,
                raw_dtype=GGMLQuantizationType.MXFP4,
            )

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{total_tensors}] Q8→MXFP4: {name} "
                      f"({in_bytes/1e6:.1f}MB → {out_bytes/1e6:.1f}MB, "
                      f"saved {(in_bytes-out_bytes)/1e6:.1f}MB) "
                      f"[{elapsed:.0f}s]", flush=True)
        else:
            # Copy as-is
            other_count += 1

            # Make a copy since raw_data may be a memmap view
            data_copy = np.array(raw_data, copy=True)
            total_bytes_in += data_copy.nbytes
            total_bytes_out += data_copy.nbytes

            # For quantized types, raw_dtype tells the writer to use
            # quant_shape_from_byte_shape to compute logical shape.
            # For F16/F32, orig_type will match numpy dtype inference.
            writer.add_tensor(
                name, data_copy,
                raw_dtype=orig_type,
            )

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{total_tensors}] copy: {name} "
                      f"({data_copy.nbytes/1e6:.1f}MB) [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nConversion complete in {elapsed:.0f}s")
    print(f"  Q8_0→MXFP4: {q8_count} tensors")
    print(f"  Copied as-is: {other_count} tensors")
    print(f"  Input size:  {total_bytes_in/1e9:.2f} GB")
    print(f"  Output size: {total_bytes_out/1e9:.2f} GB")
    print(f"  Savings:     {(total_bytes_in - total_bytes_out)/1e9:.2f} GB "
          f"({(1 - total_bytes_out/total_bytes_in)*100:.1f}%)")

    print("\nWriting header...")
    writer.write_header_to_file()
    print("Writing KV metadata...")
    writer.write_kv_data_to_file()
    print("Writing tensor data...")
    writer.write_tensors_to_file(progress=True)
    print("Closing...")
    writer.close()
    print(f"Done: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Q8_0 tensors to MXFP4 in a GGUF model file")
    parser.add_argument("input", type=str, help="Input GGUF file path")
    parser.add_argument("output", type=str, nargs="?",
                        help="Output GGUF file path (default: input_mxfp4.gguf)")
    parser.add_argument("--no-temp-file", action="store_true",
                        help="Disable temp file (keeps all tensors in RAM)")

    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_mxfp4{p.suffix}")

    print(f"=== Q8_0 → MXFP4 Converter ===")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()

    convert_q8_to_mxfp4(args.input, args.output, use_temp_file=not args.no_temp_file)
