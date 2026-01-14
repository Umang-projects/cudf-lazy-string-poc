import cudf
import cupy as cp
import torch
import numpy as np
import os
from torch.profiler import profile, record_function, ProfilerActivity

# ============================================================================
# 1. LOAD KERNEL FROM FILE
# ============================================================================
kernel_path = os.path.join("/content/cudf-lazy-split-poc/src", "lazy_extract.cu")
if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"Kernel file not found: {kernel_path}. Create 'src' folder and add 'lazy_extract.cu'.")

with open(kernel_path, "r") as f:
    kernel_source = f.read()

lazy_kernel = cp.RawKernel(kernel_source, 'lazy_extract_kernel')
print("✅ Kernel loaded from src/lazy_extract.cu")

# ============================================================================
# 2. DATA & SETUP
# ============================================================================
ROWS = 5_000_000
print(f"🚀 Generating {ROWS} rows...")
df = cudf.DataFrame({
    'text': ['user_id_' + str(x) + '_log' for x in np.random.randint(1000, 9999, ROWS)]
})

col = df['text']._column
d_offsets = cp.asarray(col.children[0])
d_chars = cp.asarray(col.base_data)

MAX_LEN = 10
d_out_chars = cp.zeros(ROWS * MAX_LEN, dtype=cp.uint8)
d_out_lens = cp.zeros(ROWS, dtype=cp.int32)

args = (
    d_chars.data.ptr, d_offsets.data.ptr,
    d_out_chars.data.ptr, d_out_lens.data.ptr,
    ROWS, ord('_'), 2, MAX_LEN
)
block_size = 256
grid_size = (ROWS + block_size - 1) // block_size

print("✅ Setup complete. Starting profiler...")

# ============================================================================
# 3. PROFILING
# ============================================================================
# Warmup
df['text'].head().str.split('_')
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    
    with record_function("cuDF_Baseline_Split"):  # Removed emoji for safety
        res = df['text'].str.split('_').list.get(2)
        torch.cuda.synchronize()

    with record_function("Custom_Lazy_Kernel"):
        lazy_kernel((grid_size,), (block_size,), args)
        torch.cuda.synchronize()

# ============================================================================
# 4. RESULTS & SPEEDUP
# ============================================================================
print("\n📊 Profiling Results (top 15 by cuda_time_total):")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

averages = prof.key_averages()

# Use cuda_time_total (in us) → ms; fallback to device_time_total for compatibility
def get_total_device_time(a):
    if hasattr(a, 'cuda_time_total'):
        return a.cuda_time_total
    elif hasattr(a, 'device_time_total'):
        return a.device_time_total
    return 0

cudf_time_us = next((get_total_device_time(a) for a in averages if a.key == "cuDF_Baseline_Split"), 0)
custom_time_us = next((get_total_device_time(a) for a in averages if a.key == "Custom_Lazy_Kernel"), 0)

cudf_time_ms = cudf_time_us / 1000.0
custom_time_ms = custom_time_us / 1000.0

if cudf_time_ms > 0 and custom_time_ms > 0:
    speedup = cudf_time_ms / custom_time_ms
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    print(f"   cuDF baseline: {cudf_time_ms:.2f} ms")
    print(f"   Custom lazy kernel: {custom_time_ms:.2f} ms")
else:
    print("\n⚠️ Speedup not calculated. Times not found.")
    print("Available keys for debugging:")
    print([a.key for a in averages if "Split" in a.key or "Kernel" in a.key])
