#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

template <typename T>
__global__ void traceKernel(const T *input,T *output,size_t rows,size_t cols){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<rows&&idx<cols){
      output[idx]=input[idx*cols+idx];
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
  // TODO: Implement the trace function
   if (rows == 0 || cols == 0) {
        return T(0);  
    }
  T *h_output= (T*)malloc(std::min(rows,cols) * sizeof(T));
  T* d_input = nullptr;
  T* d_output = nullptr;
  cudaMalloc(&d_input,sizeof(T)*h_input.size());
  cudaMalloc(&d_output,sizeof(T)*(std::min(rows,cols)));
  cudaMemcpy(d_input,h_input.data(),h_input.size() * sizeof(T),cudaMemcpyHostToDevice);
  dim3 block(16,1,1);
  dim3 grad((std::min(rows, cols)+16)/16,1,1);
  traceKernel<<<grad,block>>>(d_input,d_output,rows,cols);
  cudaMemcpy(h_output,d_output,std::min(rows,cols) * sizeof(T),cudaMemcpyDeviceToHost);
  T sum=T(0);
  for(int i=0;i<std::min(rows,cols);i++){
    sum+=h_output[i];
  }
  cudaFree(d_input);
  cudaFree(d_output);
  return sum;
}
__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(half v) { return __half2float(v); }
__device__ __forceinline__ float from_f32(float v, float dummy) { return v; }
__device__ __forceinline__ half from_f32(float v, half dummy) { return __float2half(v); }

template <typename T>
__global__ void naive_attention_kernel(const T* Q, const T* K, const T* V,
     T* O,int batch_size, int tgt_len, int src_len,
    int q_heads, int kv_heads, int head_dim,
    float scale, bool is_causal
) {
    // 计算tsl中的一个
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    int b=blockIdx.z;
    int h=blockIdx.y;
    if(idx>=tgt_len) return;
    //gqa 计算出q使用的k和v
    int kv_h=h/(q_heads/kv_heads);
    int q_base=b*tgt_len*q_heads*head_dim+idx*q_heads*head_dim+h*head_dim;
    int kv_base=b*src_len*kv_heads*head_dim+kv_h*head_dim;

    // 计算q*kT的这一行的max
    // 遍历k逆的列
    float max_val=-1e20f;
    for(int i=0;i<src_len;i++){
        if(is_causal&&idx<i) break;
        float sum=0.0;
        // 遍历一列的所有内容
        for(int j=0;j<head_dim;j++){
            sum+=to_f32(Q[q_base+j])*to_f32(K[kv_base+i*kv_heads*head_dim+j]);
        }
        sum*=scale;
        if(sum>max_val) max_val=sum;
    }
    // 计算出一行的，
    for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
        // sum_exp为未softmax的值
        float sum_exp=0.0;
        for(int i=0;i<src_len;i++){
            if(is_causal&&idx<i) break;
            float sum=0.0;
            // 遍历一列的所有内容
            for(int j=0;j<head_dim;j++){
                sum+=to_f32(Q[q_base+j])*to_f32(K[kv_base+i*kv_heads*head_dim+j]);
            }
            sum*=scale;
            float exp_val=expf(sum-max_val);
            sum_exp+=exp_val*to_f32(V[kv_base+i*kv_heads*head_dim+d]);

        }
        // 计算softmax
        float sum_softmax=0.0;
        for(int i=0;i<src_len;i++){
            if(is_causal&&idx<i) break;
            float sum=0.0;
            // 遍历一列的所有内容
            for(int j=0;j<head_dim;j++){
                sum+=to_f32(Q[q_base+j])*to_f32(K[kv_base+i*kv_heads*head_dim+j]);
            }
            sum*=scale;
            sum_softmax+=expf(sum-max_val);
        }
        if(sum_softmax>0.0){
            O[q_base+d]=from_f32(sum_exp/sum_softmax,T());
        }
        
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
    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, h_q.size() * sizeof(T));
    cudaMalloc(&d_k, h_k.size() * sizeof(T));
    cudaMalloc(&d_v, h_v.size() * sizeof(T));
    cudaMalloc(&d_o, h_o.size() * sizeof(T));
    cudaMemcpy(d_q, h_q.data(), h_q.size()* sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), h_k.size()* sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), h_v.size()* sizeof(T), cudaMemcpyHostToDevice);
    float scale=1.0/sqrtf((float)head_dim);
    dim3 grid((target_seq_len + 16 - 1) / 16, query_heads, batch_size);
    // 第二维计算v的head_dim
    dim3 block(16,16,1); 
    naive_attention_kernel<T><<<grid, block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len, 
        query_heads, kv_heads, head_dim, 
        scale, is_causal
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_o.data(), d_o, h_o.size()* sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
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
