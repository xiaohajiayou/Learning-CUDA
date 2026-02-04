#include <vector>
#include <cmath>
#include <cfloat>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../tester/utils.h"

template <typename T, size_t BS>
__global__ void trace_kernel(const T *input, T *output, size_t cols, size_t diag_n) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T sdata[BS];
  T val = 0;
  if (idx < diag_n) {
    size_t offset = idx * cols + idx;
    val = input[offset];
  }
  sdata[tid] = val;
  __syncthreads();
  for (size_t stride = BS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

template <typename T, size_t BS>
__global__ void reduce_kernel(const T *input, T *output, size_t n) {
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T sdata[BS];
  T val = 0;
  if (idx < n) {
    val = input[idx];
  }
  sdata[tid] = val;
  __syncthreads();
  for (size_t stride = BS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}



/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO(step1): handle edge cases (rows == 0 || cols == 0 || rows != cols ).
  if (rows == 0 || cols == 0) {
    return T(0);
  }
  const size_t n = min(rows, cols);
  const size_t total = rows * cols;
  // TODO(step2): allocate device buffers and copy h_input to device.
  T *d_input = nullptr;
  T *d_output = nullptr;
  T h_output = T(0);
  constexpr int threads_per_block = 256;
  int num_block = (static_cast<int>(n) + threads_per_block -1) / threads_per_block;
  RUNTIME_CHECK(cudaMalloc(&d_input, total * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_output, num_block * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), total * sizeof(T) ,cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_output, 0, num_block * sizeof(T)));
  
  // TODO(step3): set cuda resource and launch a kernel to sum diagonal elements.
  trace_kernel<T, threads_per_block><<<num_block, threads_per_block>>>(d_input, d_output, cols, n);

  size_t cur_n = num_block;
  T *cur_input = d_output;
  T *cur_output = nullptr;
  RUNTIME_CHECK(cudaMalloc(&cur_output, num_block * sizeof(T)));
  while (cur_n > 1) {
    num_block = (cur_n + threads_per_block - 1) / threads_per_block;
    dim3 block(threads_per_block);
    dim3 grid(num_block);
    reduce_kernel<T, threads_per_block><<<grid, block>>>(cur_input, cur_output, cur_n);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    cur_n = num_block;
    cur_input = cur_output;
  }
  
  // TODO(step4): copy result back and free buffers.
  RUNTIME_CHECK(cudaMemcpy(&h_output, cur_input, sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(cur_output));
  return h_output;
}




template <typename T>
__device__ inline float to_float(T v) {
  if constexpr (std::is_same<T, __half>::value) {
    return __half2float(v);
  } else {
    return static_cast<float>(v);
  }
}

template <typename T>
__device__ inline void store_val(T* dst, size_t idx, float v) {
  if constexpr (std::is_same<T, __half>::value) {
    dst[idx] = __float2half_rn(v);
  } else {
    dst[idx] = static_cast<T>(v);
  }
}

template <typename T>
inline float host_to_float(T v) {
  if constexpr (std::is_same<T, __half>::value) {
    return __half2float(v);
  } else {
    return static_cast<float>(v);
  }
}

template <typename T, int Br, int Bc>
__global__ void flash_attention_v1_kernel(const T* q, const T* k, const T* v, T* o,
                                          float* l, float* m,
                                          int bsz, int tgt_len, int src_len,
                                          int q_heads, int kv_heads, int d, bool causal) {
  int b = blockIdx.x;
  int qh = blockIdx.y;
  int tid = threadIdx.x; // 0..Br-1

  if (b >= bsz || qh >= q_heads) {
    return;
  }

  int num_group = q_heads / kv_heads;
  int kvh = (num_group > 0) ? (qh / num_group) : 0;
  const float scale = rsqrtf(static_cast<float>(d));

  // shared layout: Qi[Br*D], Kj[Bc*D], Vj[Bc*D], S[Br*Bc]
  extern __shared__ float sram[];
  float* Qi = sram;
  float* Kj = Qi + Br * d;
  float* Vj = Kj + Bc * d;
  float* S  = Vj + Bc * d;

  int Tc = (src_len + Bc - 1) / Bc;
  int Tr = (tgt_len + Br - 1) / Br;

  for (int j = 0; j < Tc; ++j) {
    int s = j * Bc + tid;
    if (tid < Bc) {
      for (int x = 0; x < d; ++x) {
        float kv = 0.f;
        if (s < src_len) {
          size_t k_base = ((static_cast<size_t>(b) * src_len + s) * kv_heads + kvh) * d;
          kv = to_float(k[k_base + x]);
        }
        Kj[tid * d + x] = kv;
      }
      for (int x = 0; x < d; ++x) {
        float vv = 0.f;
        if (s < src_len) {
          size_t v_base = ((static_cast<size_t>(b) * src_len + s) * kv_heads + kvh) * d;
          vv = to_float(v[v_base + x]);
        }
        Vj[tid * d + x] = vv;
      }
    }
    __syncthreads();

    for (int i = 0; i < Tr; ++i) {
      int t = i * Br + tid;
      if (tid < Br) {
        for (int x = 0; x < d; ++x) {
          float qv = 0.f;
          if (t < tgt_len) {
            size_t q_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d;
            qv = to_float(q[q_base + x]);
          }
          Qi[tid * d + x] = qv;
        }
      }
      __syncthreads();

      if (tid < Br) {
        using Accum = typename std::conditional<std::is_same<T, __half>::value, double, float>::type;
        float row_m_prev = -FLT_MAX;
        Accum row_l_prev = static_cast<Accum>(0.0);
        if (t < tgt_len) {
          size_t lm_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh);
          row_m_prev = m[lm_base];
          row_l_prev = static_cast<Accum>(l[lm_base]);
        }

        float row_m = -FLT_MAX;
        for (int y = 0; y < Bc; ++y) {
          int s_idx = j * Bc + y;
          float sum = 0.f;
          for (int x = 0; x < d; ++x) {
            sum += Qi[tid * d + x] * Kj[y * d + x];
          }
          sum *= scale;
          if (s_idx >= src_len) {
            sum = -FLT_MAX;
          } else if (causal && s_idx > t) {
            sum = -FLT_MAX;
          }
          S[tid * Bc + y] = sum;
          row_m = fmaxf(row_m, sum);
        }

        bool row_valid = (row_m != -FLT_MAX);

        Accum row_l = static_cast<Accum>(0.0);
        for (int y = 0; y < Bc; ++y) {
          float expv = row_valid ? expf(S[tid * Bc + y] - row_m) : 0.0f;
          S[tid * Bc + y] = expv;
          row_l += static_cast<Accum>(expv);
        }

        float row_m_new = row_valid ? fmaxf(row_m_prev, row_m) : row_m_prev;
        Accum row_l_new = row_valid
            ? static_cast<Accum>(expf(row_m_prev - row_m_new)) * row_l_prev
              + static_cast<Accum>(expf(row_m - row_m_new)) * row_l
            : row_l_prev;

        if (t < tgt_len) {
          size_t o_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d;
          for (int x = 0; x < d; ++x) {
            Accum pv = static_cast<Accum>(0.0);
            for (int y = 0; y < Bc; ++y) {
              pv += static_cast<Accum>(S[tid * Bc + y]) * static_cast<Accum>(Vj[y * d + x]);
            }
            Accum o_prev = static_cast<Accum>(to_float(o[o_base + x]));
            Accum out = row_valid
                ? (static_cast<Accum>(expf(row_m_prev - row_m_new)) * row_l_prev * o_prev
                   + static_cast<Accum>(expf(row_m - row_m_new)) * pv) / row_l_new
                : o_prev;
            store_val(o, o_base + x, static_cast<float>(out));
          }
          size_t lm_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh);
          m[lm_base] = row_m_new;
          l[lm_base] = static_cast<float>(row_l_new);
        }
      }
      __syncthreads();
    }
  }
}

template <typename T, int Br, int Bc>
__global__ void flash_attention_v2_kernel(const T* q, const T* k, const T* v, T* o,
                                          float* l, float* m,
                                          int bsz, int tgt_len, int src_len,
                                          int q_heads, int kv_heads, int d, bool causal) {
  static_assert(Br == Bc, "flash_attention_v2_kernel requires Br == Bc");
  int b = blockIdx.x;
  int qh = blockIdx.y;
  int tid = threadIdx.x; // 0..Br-1 (assume Br == Bc)

  if (b >= bsz || qh >= q_heads) {
    return;
  }

  int num_group = q_heads / kv_heads;
  int kvh = (num_group > 0) ? (qh / num_group) : 0;
  const float scale = rsqrtf(static_cast<float>(d));

  // shared layout: Qi[Br*D], Oi[Br*D], Kj[Bc*D], Vj[Bc*D], S[Br*Bc]
  extern __shared__ float sram[];
  float* Qi = sram;
  float* Oi = Qi + Br * d;
  float* Kj = Oi + Br * d;
  float* Vj = Kj + Bc * d;
  float* S  = Vj + Bc * d;

  int Tc = (src_len + Bc - 1) / Bc;
  int Tr = (tgt_len + Br - 1) / Br;

  for (int i = 0; i < Tr; ++i) {
    int t = i * Br + tid;
    if (tid < Br) {
      for (int x = 0; x < d; ++x) {
        float qv = 0.f;
        float ov = 0.f;
        if (t < tgt_len) {
          size_t q_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d;
          qv = to_float(q[q_base + x]);
          ov = to_float(o[q_base + x]);
        }
        Qi[tid * d + x] = qv;
        Oi[tid * d + x] = ov;
      }
    }
    __syncthreads();

    float row_m_prev = -FLT_MAX;
    float row_l_prev = 0.f;
    if (tid < Br && t < tgt_len) {
      size_t lm_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh);
      row_m_prev = m[lm_base];
      row_l_prev = l[lm_base];
    }

    for (int j = 0; j < Tc; ++j) {
      int s = j * Bc + tid;
      if (tid < Bc) {
        for (int x = 0; x < d; ++x) {
          float kv = 0.f;
          float vv = 0.f;
          if (s < src_len) {
            size_t k_base = ((static_cast<size_t>(b) * src_len + s) * kv_heads + kvh) * d;
            size_t v_base = ((static_cast<size_t>(b) * src_len + s) * kv_heads + kvh) * d;
            kv = to_float(k[k_base + x]);
            vv = to_float(v[v_base + x]);
          }
          Kj[tid * d + x] = kv;
          Vj[tid * d + x] = vv;
        }
      }
      __syncthreads();

      if (tid < Br && t < tgt_len) {
        float row_m = -FLT_MAX;
        for (int y = 0; y < Bc; ++y) {
          int s_idx = j * Bc + y;
          float sum = 0.f;
          for (int x = 0; x < d; ++x) {
            sum += Qi[tid * d + x] * Kj[y * d + x];
          }
          sum *= scale;
          if (s_idx >= src_len || (causal && s_idx > t)) {
            sum = -FLT_MAX;
          }
          S[tid * Bc + y] = sum;
          row_m = fmaxf(row_m, sum);
        }

        bool row_valid = (row_m != -FLT_MAX);
        float row_m_new = row_valid ? fmaxf(row_m_prev, row_m) : row_m_prev;
        float row_l = 0.f;
        if (row_valid) {
          for (int y = 0; y < Bc; ++y) {
            float expv = expf(S[tid * Bc + y] - row_m_new);
            S[tid * Bc + y] = expv;
            row_l += expv;
          }
        }
        float row_l_new = row_valid
            ? expf(row_m_prev - row_m_new) * row_l_prev + row_l
            : row_l_prev;

        if (row_valid) {
          float prev_scale = expf(row_m_prev - row_m_new);
          for (int x = 0; x < d; ++x) {
            float pv = 0.f;
            for (int y = 0; y < Bc; ++y) {
              pv += S[tid * Bc + y] * Vj[y * d + x];
            }
            Oi[tid * d + x] = prev_scale * Oi[tid * d + x] + pv;
          }
          row_m_prev = row_m_new;
          row_l_prev = row_l_new;
        }
      }
      __syncthreads();
    }

    if (tid < Br && t < tgt_len) {
      size_t o_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d;
      float inv_l = (row_l_prev > 0.f) ? (1.f / row_l_prev) : 0.f;
      for (int x = 0; x < d; ++x) {
        store_val(o, o_base + x, Oi[tid * d + x] * inv_l);
      }
      size_t lm_base = ((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh);
      m[lm_base] = row_m_prev;
      l[lm_base] = row_l_prev;
    }
    __syncthreads();
  }
}



/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO(step1): validate shapes, resize h_o if needed.
  if (batch_size == 0 || target_seq_len == 0 || src_seq_len == 0 ||
      query_heads == 0 || kv_heads == 0 || head_dim == 0) {
    return;
  }
  if (query_heads % kv_heads != 0) {
    printf("q_heads must be mutible of kv_heads");
    return;
  }
  const size_t q_size = static_cast<size_t>(batch_size * target_seq_len * query_heads * head_dim);
  const size_t k_size = static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim);
  const size_t v_size = k_size;
  const size_t o_size = q_size;
  if (h_o.size() != o_size) {
    h_o.resize(o_size);
  }

  // TODO(step2): allocate device buffers and copy h_q/h_k/h_v to device.
  T *d_q = nullptr;
  T *d_k = nullptr;
  T *d_v = nullptr;
  T *d_o = nullptr;
  
  RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
  
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice));

  // TODO(step3): launch a tiled attention kernel.
  dim3 grid(batch_size, query_heads);
  const size_t lm_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads;
  float *d_l = nullptr;
  float *d_m = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_l, lm_size * sizeof(float)));
  RUNTIME_CHECK(cudaMalloc(&d_m, lm_size * sizeof(float)));
  RUNTIME_CHECK(cudaMemset(d_l, 0, lm_size * sizeof(float)));
  {
    std::vector<float> h_m(lm_size, -FLT_MAX);
    RUNTIME_CHECK(cudaMemcpy(d_m, h_m.data(), lm_size * sizeof(float), cudaMemcpyHostToDevice));
  }
  RUNTIME_CHECK(cudaMemset(d_o, 0, o_size * sizeof(T)));

  int max_smem = 0;
  RUNTIME_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
  // Paper heuristic: Bc ~= sram_max / (4 * d), Br ~= min(Bc, d).
  const int ideal_bc = max_smem / (4 * head_dim * static_cast<int>(sizeof(float)));
  const int ideal_br = (ideal_bc < head_dim) ? ideal_bc : head_dim;

  auto smem_bytes = [head_dim](int B) {
    return static_cast<size_t>(2 * B * head_dim + 2 * B * head_dim + B * B) * sizeof(float);
  };

  int chosen = 16;
  if (ideal_br >= 128 && smem_bytes(128) <= static_cast<size_t>(max_smem)) {
    chosen = 128;
  } else if (ideal_br >= 64 && smem_bytes(64) <= static_cast<size_t>(max_smem)) {
    chosen = 64;
  } else if (ideal_br >= 32 && smem_bytes(32) <= static_cast<size_t>(max_smem)) {
    chosen = 32;
  } else if (smem_bytes(16) <= static_cast<size_t>(max_smem)) {
    chosen = 16;
  }

  if (chosen == 128) {
    constexpr int Br = 128;
    constexpr int Bc = 128;
    constexpr int threads_per_block = Br;
    flash_attention_v2_kernel<T, Br, Bc><<<grid, threads_per_block, smem_bytes(128)>>>(
        d_q, d_k, d_v, d_o, d_l, d_m,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
  } else if (chosen == 64) {
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int threads_per_block = Br;
    flash_attention_v2_kernel<T, Br, Bc><<<grid, threads_per_block, smem_bytes(64)>>>(
        d_q, d_k, d_v, d_o, d_l, d_m,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
  } else if (chosen == 32) {
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int threads_per_block = Br;
    flash_attention_v2_kernel<T, Br, Bc><<<grid, threads_per_block, smem_bytes(32)>>>(
        d_q, d_k, d_v, d_o, d_l, d_m,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
  } else {
    constexpr int Br = 16;
    constexpr int Bc = 16;
    constexpr int threads_per_block = Br;
    flash_attention_v2_kernel<T, Br, Bc><<<grid, threads_per_block, smem_bytes(16)>>>(
        d_q, d_k, d_v, d_o, d_l, d_m,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
  }

  RUNTIME_CHECK(cudaFree(d_l));
  RUNTIME_CHECK(cudaFree(d_m));
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());
  // TODO(step4): copy output back and free buffers.
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
