#include <vector>
#include <cmath>
#include <cfloat>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
__global__ void trace_kernel(const T* input, T* out, size_t cols, size_t diag_n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= diag_n) {
    return;
  }
  size_t offset = idx * cols + idx;
  atomicAdd(out, input[offset]);
}

template <typename T>
__global__ void flash_attention_kernel(const T* q, const T* k, const T* v, T* o,
                                       int bsz, int tgt_len, int src_len,
                                       int q_heads, int kv_heads, int d, bool causal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= bsz * tgt_len * q_heads) {
    return;
  }

  int b = idx / (tgt_len * q_heads);
  int rem = idx % (tgt_len * q_heads);
  int t = rem / q_heads;
  int qh = rem % q_heads;

  int group = q_heads / kv_heads;
  int kh = (group > 0) ? (qh / group) : 0;
  if (kh >= kv_heads) {
    kh = kv_heads - 1;
  }

  const float scale = rsqrtf(static_cast<float>(d));

  float max_score = -FLT_MAX;
  for (int s = 0; s < src_len; ++s) {
    if (causal && s > t) {
      continue;
    }
    float dot = 0.0f;
    const size_t q_base = (((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d);
    const size_t k_base = (((static_cast<size_t>(b) * src_len + s) * kv_heads + kh) * d);
    for (int i = 0; i < d; ++i) {
      dot += to_float(q[q_base + i]) * to_float(k[k_base + i]);
    }
    dot *= scale;
    if (dot > max_score) {
      max_score = dot;
    }
  }

  float denom = 0.0f;
  for (int s = 0; s < src_len; ++s) {
    if (causal && s > t) {
      continue;
    }
    float dot = 0.0f;
    const size_t q_base = (((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d);
    const size_t k_base = (((static_cast<size_t>(b) * src_len + s) * kv_heads + kh) * d);
    for (int i = 0; i < d; ++i) {
      dot += to_float(q[q_base + i]) * to_float(k[k_base + i]);
    }
    denom += expf(dot * scale - max_score);
  }

  const size_t o_base = (((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d);
  for (int i = 0; i < d; ++i) {
    float out = 0.0f;
    for (int s = 0; s < src_len; ++s) {
      if (causal && s > t) {
        continue;
      }
      float dot = 0.0f;
      const size_t q_base = (((static_cast<size_t>(b) * tgt_len + t) * q_heads + qh) * d);
      const size_t k_base = (((static_cast<size_t>(b) * src_len + s) * kv_heads + kh) * d);
      const size_t v_base = (((static_cast<size_t>(b) * src_len + s) * kv_heads + kh) * d);
      for (int j = 0; j < d; ++j) {
        dot += to_float(q[q_base + j]) * to_float(k[k_base + j]);
      }
      float w = expf(dot * scale - max_score) / denom;
      out += w * to_float(v[v_base + i]);
    }
    store_val(o, o_base + i, out);
  }
}
#include "../tester/utils.h"

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
  if (rows == 0 || cols == 0) {
    return T(0);
  }

  const size_t n = rows < cols ? rows : cols;
  const size_t total = rows * cols;

  T *d_input = nullptr;
  T *d_out = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, total * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_out, sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), total * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  const int threads = 256;
  const int blocks = (static_cast<int>(n) + threads - 1) / threads;

  trace_kernel<<<blocks, threads>>>(d_input, d_out, cols, n);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  T h_out = T(0);
  RUNTIME_CHECK(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_out));
  return h_out;
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
  if (batch_size == 0 || target_seq_len == 0 || src_seq_len == 0 ||
      query_heads == 0 || kv_heads == 0 || head_dim == 0) {
    return;
  }

  const size_t q_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t k_size = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  const size_t v_size = k_size;
  const size_t o_size = q_size;

  if (h_o.size() != o_size) {
    h_o.resize(o_size);
  }

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

  const int total_q = batch_size * target_seq_len * query_heads;
  const int threads = 128;
  const int blocks = (total_q + threads - 1) / threads;

  flash_attention_kernel<<<blocks, threads>>>(d_q, d_k, d_v, d_o,
                              batch_size, target_seq_len, src_seq_len,
                              query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

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
