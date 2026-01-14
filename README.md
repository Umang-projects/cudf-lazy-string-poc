# ⚡ Accelerated String Extraction for cuDF (~160x Speedup)

**Proof of Concept:** Optimizing the `str.split().list.get()` pattern on GPUs using Fused "Lazy" Kernels.

![Status](https://img.shields.io/badge/Status-POC-yellow) ![Speedup](https://img.shields.io/badge/Speedup-160x-brightgreen) ![Hardware](https://img.shields.io/badge/Hardware-Tesla%20T4-orange)

---

## 🧐 The Problem

In ETL pipelines (e.g., Log Parsing), a common operation is extracting a specific field from a delimiter-separated string.

**Example:** Extracting the ID `1234` from `"user_id_1234_log"`.

In **cuDF (v24.xx)**, the standard approach is:

```python
df['col'].str.split('_').list.get(2)
```

### Why is this slow? (The "Eager" Bottleneck)

**Materialization:** cuDF eagerly splits the entire string into multiple tokens (user, id, 1234, log).

**Memory Pressure:** It allocates global memory for all these tokens, even though we discard 80% of them immediately.

**Kernel Overhead:** It triggers multiple kernel launches:
- count_delimiters (Scan pass)
- cub::DeviceSelect (Offset calculation)
- gather_chars (Copying data)

On a Tesla T4, this results in ~137ms latency for 5M rows.

---

## 🛠️ The Solution: "Lazy" Fused Kernel

I implemented a custom CUDA C++ kernel (injected via CuPy) that performs Lazy Extraction.

### Key Architectural Changes:

**Single Pass Scan:** The kernel scans the string in registers (one thread per row).

**No Intermediate Writes:** It counts delimiters on-the-fly. It does not write split tokens to VRAM.

**Targeted Copy:** It only writes the target substring (Index 2) to the output buffer once found.

**Zero Allocations:** Eliminates the need for temporary offset arrays or token buffers.

### 💻 Core Logic (Simplified)

```cpp
// Thread Logic (No global memory traffic for unused tokens)
for (char c : string) {
    if (c == delimiter) {
        count++;
    } 
    else if (count == target_index) {
        output_buffer[pos++] = c; // Only write what we need
    }
}
```

---

## 📊 Benchmark Results

### Configuration:

- **Hardware:** NVIDIA Tesla T4 (16GB GDDR6)
- **Data:** 5 Million Rows ("user_id_XXXX_log")
- **Operation:** Extract Index 2

| Implementation | Execution Time (Profiler) | Speedup | Memory Overhead |
|----------------|---------------------------|---------|-----------------|
| cuDF (Standard) | 137.15 ms | 1x | High (Allocates all splits) |
| Custom Kernel | 0.85 ms | ~160x 🚀 | Zero (In-place) |

(See attached PyTorch Profiler trace for detailed breakdown)

---

## 🔬 Profiling Analysis (The Proof)

The profiling trace clearly shows the difference in execution flow:

### 1. cuDF Baseline Trace

Multiple kernels running in sequence, consuming significant time:
- static_kernel (CUB) - 31.3ms
- gather_chars - 7.6ms
- batch_memcpy - 7.4ms
- count_delimiters - 2.6ms

### 2. Custom Kernel Trace

A single kernel launch lazy_extract_kernel taking only 853 microseconds (0.85ms).

---

## 🚀 How to Reproduce

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Benchmark

This script generates 5M rows, compiles the CUDA kernel at runtime, and compares performance using PyTorch Profiler.

```bash
python benchmark.py
```

---

## 🔮 Future Scope

This logic can be extended to Multi-Column Extraction (Log Parsing), allowing users to extract multiple fields (e.g., Timestamp, IP, Error) in a single pass, completely replacing complex Regex pipelines on GPU.

---

**Author:** Umang Singh

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python benchmark.py
