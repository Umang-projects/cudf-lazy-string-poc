extern "C" __global__
void lazy_extract_kernel(
    const char* __restrict__ input_chars,
    const int* __restrict__ offsets,
    char* __restrict__ output_chars,
    int* __restrict__ output_lengths,
    int num_rows,
    char delimiter,
    int target_idx,
    int max_out_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_rows) {
        int start = offsets[tid];
        int end = offsets[tid+1];

        int delimiter_count = 0;
        int current_out_len = 0;
        int write_base_ptr = tid * max_out_len; 

        // Optimized Scan Loop
        for(int i = start; i < end; i++) {
            char c = input_chars[i];
            if (c == delimiter) {
                delimiter_count++;
                if (delimiter_count > target_idx) break; 
            } else {
                if (delimiter_count == target_idx) {
                    if (current_out_len < max_out_len) {
                        output_chars[write_base_ptr + current_out_len] = c;
                        current_out_len++;
                    }
                }
            }
        }
        output_lengths[tid] = current_out_len;
    }
}
