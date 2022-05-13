#include <cuda.h>
#include "utils.h"
#include <cuda_runtime.h>
#include "utils/gemm/src_iterator.h"
#include "utils/matmul/computer.h"
#include "utils/gemm/store_util.h"


#define FULL_MASK 0xffffffff

#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<int TileM, int TileN, int TileK, int wTileM, int wTileN, int M, int N, int K, int Stages>
__device__ __forceinline__ void verify_matmul_(
    const float* __restrict__ x_lb,
    const float* __restrict__ x_ub,
    const float* __restrict__ W,
    float* __restrict__ y_lb,
    float* __restrict__ y_ub,
    int batch_size_length, int dim_out, int dim_y_out,
    long blockIdx_x
){
    int k = dim_out;
    int n = dim_y_out;
    // Get static variables
    constexpr int NWarps = TileN / wTileN;
    constexpr int MWarps = TileM / wTileM;
    constexpr int NWarpTiles = wTileN / N;
    constexpr int MWarpTiles = wTileM / M;
    constexpr int NumWarp = NWarps * MWarps;
    constexpr int BatchOffset = (2 * TileM + TileN) * TileK;

    // dynamic shared memory
    extern __shared__ float smem[];

    // Warp and lane identification
    // const unsigned int warpId = threadIdx.x / 32;
    // const unsigned int laneId = threadIdx.x % 32;

    // get tile offset
    int m_offset = blockIdx_x * TileM;
    int n_offset = blockIdx.y * TileN;

    Iterator<TileM, TileN, TileK, 4, NumWarp, Stages, 0, BatchOffset, TileM> global_iter_x_l(x_lb, smem, m_offset, k);
    Iterator<TileM, TileN, TileK, 4, NumWarp, Stages, TileM * TileK, BatchOffset, TileN> global_iter_x_u(x_ub, smem, m_offset, k);
    Iterator<TileM, TileN, TileK, 4, NumWarp, Stages, 2 * TileM * TileK, BatchOffset, TileN> global_iter_w(W, smem, n_offset, k);

    ShmemIteratorA<NWarps, TileK, Stages, wTileM, wTileN, 4, M, N, K, 0, BatchOffset> shmem_iter_x_l(smem);
    ShmemIteratorA<NWarps, TileK, Stages, wTileM, wTileN, 4, M, N, K, TileM * TileK, BatchOffset> shmem_iter_x_u(smem);
    ShmemIteratorB<NWarps, TileK, Stages, wTileM, wTileN, 4, M, N, K, 2 * TileM * TileK, BatchOffset> shmem_iter_w(smem);

    wmma::fragment<wmma::accumulator, M, N, K, float> c_l[MWarpTiles][NWarpTiles];
    wmma::fragment<wmma::accumulator, M, N, K, float> c_u[MWarpTiles][NWarpTiles];

    float a_l[2][MWarpTiles][4];
    float a_u[2][MWarpTiles][4];
    float b_n[2][NWarpTiles][4];
    float b_p[2][NWarpTiles][4];
    // float b[2][NWarpTiles][4];

    // Set the fragment to 0
    #pragma unroll
    for (int i = 0; i < MWarpTiles; i++){
        #pragma unroll
        for (int j = 0; j < NWarpTiles; j++){
            wmma::fill_fragment(c_l[i][j], 0.0f);
            wmma::fill_fragment(c_u[i][j], 0.0f);
        }
    }
    // TODO: Currently, we assume that there is no residual on k dimension
    int k_batch = k / TileK; 
    int fetch_batch = 0;
    // Prologue
    #pragma unroll
    for (int stage = 0; stage < Stages-1; stage ++, fetch_batch ++){
        global_iter_x_l.Load_async(fetch_batch);
        global_iter_x_u.Load_async(fetch_batch);
        global_iter_w.Load_async(fetch_batch);
        asm volatile ("cp.async.commit_group;");
    }
    asm volatile ("cp.async.wait_group %0;" :: "n"(Stages - 2) );
    __syncthreads();
    shmem_iter_x_l.load_block_tile(a_l, 0, 0);
    shmem_iter_x_u.load_block_tile(a_u, 0, 0);
    shmem_iter_w.load_block_tile(b_n, b_p, 0, 0);

    // Main loop
    for (int compute_batch = 0; compute_batch < k_batch; compute_batch ++){
        if (fetch_batch < k_batch){
            global_iter_x_l.Load_async(fetch_batch);
            global_iter_x_u.Load_async(fetch_batch);
            global_iter_w.Load_async(fetch_batch);
            fetch_batch ++;
            asm volatile ("cp.async.commit_group;");
        }
        #pragma unroll
        for (int k_step = 0; k_step < TileK / K; k_step ++){
            if (k_step < TileK / K - 1){
                shmem_iter_x_l.load_block_tile(a_l, compute_batch, k_step + 1);
                shmem_iter_x_u.load_block_tile(a_u, compute_batch, k_step + 1);
                shmem_iter_w.load_block_tile(b_n, b_p, compute_batch, k_step + 1);
            }
            compute_block_tile<M, N, K, MWarpTiles, NWarpTiles>(c_l, a_l, b_p, k_step % 2);
            compute_block_tile<M, N, K, MWarpTiles, NWarpTiles>(c_u, a_u, b_p, k_step % 2);
            compute_block_tile<M, N, K, MWarpTiles, NWarpTiles>(c_l, a_u, b_n, k_step % 2);
            compute_block_tile<M, N, K, MWarpTiles, NWarpTiles>(c_u, a_l, b_n, k_step % 2);
        }
        asm volatile ("cp.async.wait_group %0;" :: "n"(Stages - 2) );
        __syncthreads();
        shmem_iter_x_l.load_block_tile(a_l, compute_batch + 1, 0);
        shmem_iter_x_u.load_block_tile(a_u, compute_batch + 1, 0);
        shmem_iter_w.load_block_tile(b_n, b_p, compute_batch + 1, 0);
    }

    __syncthreads();

    // Store the D fragment to shared memory
    ResStore<MWarps, NWarps, wTileM, wTileN, TileM, TileN, M, N, K, 4, NumWarp> storer1(smem, y_lb, m_offset, n_offset, n);
    ResStore<MWarps, NWarps, wTileM, wTileN, TileM, TileN, M, N, K, 4, NumWarp> storer2(smem, y_ub, m_offset, n_offset, n);
    storer1.store_block_res(c_l);

    __syncthreads();

    storer1.write_to_global(n);

    __syncthreads();

    storer2.store_block_res(c_u);

    __syncthreads();

    storer2.write_to_global(n);
}

// This version does not 1) use semantic-aware kernel fusion; 2) fuse computation of lb/ub and lw/uw
template<int TileM, int TileN, int TileK, int wTileM, int wTileN, int M, int N, int K, int Stages>
__device__ __forceinline__ void verify_matmul_unoptimized(
    const float* __restrict__ x,
    const float* __restrict__ W,
    float* __restrict__ y,
    int batch_size_length, int dim_out, int dim_y_out,
    long blockIdx_x,
    bool use_pos_weight
){
    int k = dim_out;
    int n = dim_y_out;
    // Get static variables
    constexpr int NWarps = TileN / wTileN;
    constexpr int MWarps = TileM / wTileM;
    constexpr int NWarpTiles = wTileN / N;
    constexpr int MWarpTiles = wTileM / M;
    constexpr int NumWarp = NWarps * MWarps;
    constexpr int BatchOffset = (TileM + TileN) * TileK;

    // dynamic shared memory
    extern __shared__ float smem[];

    // get tile offset
    int m_offset = blockIdx_x * TileM;
    int n_offset = blockIdx.y * TileN;

    Iterator<TileM, TileN, TileK, 4, NumWarp, Stages, 0, BatchOffset, TileM> global_iter_x(x, smem, m_offset, k);
    Iterator<TileM, TileN, TileK, 4, NumWarp, Stages, TileM * TileK, BatchOffset, TileN> global_iter_w(W, smem, n_offset, k);

    ShmemIteratorA<NWarps, TileK, Stages, wTileM, wTileN, 4, M, N, K, 0, BatchOffset> shmem_iter_x(smem);
    ShmemIteratorB<NWarps, TileK, Stages, wTileM, wTileN, 4, M, N, K, TileM * TileK, BatchOffset> shmem_iter_w(smem);

    wmma::fragment<wmma::accumulator, M, N, K, float> c[MWarpTiles][NWarpTiles];

    float a[2][MWarpTiles][4];
    float b_n[2][NWarpTiles][4];
    float b_p[2][NWarpTiles][4];

    // Set the fragment to 0
    #pragma unroll
    for (int i = 0; i < MWarpTiles; i++){
        #pragma unroll
        for (int j = 0; j < NWarpTiles; j++){
            wmma::fill_fragment(c[i][j], 0.0f);
        }
    }
    // TODO: Currently, we assume that there is no residual on k dimension
    int k_batch = k / TileK; 
    int fetch_batch = 0;
    // Prologue
    #pragma unroll
    for (int stage = 0; stage < Stages-1; stage ++, fetch_batch ++){
        global_iter_x.Load_async(fetch_batch);
        global_iter_w.Load_async(fetch_batch);
        asm volatile ("cp.async.commit_group;");
    }
    asm volatile ("cp.async.wait_group %0;" :: "n"(Stages - 2) );
    __syncthreads();
    shmem_iter_x.load_block_tile(a, 0, 0);
    shmem_iter_w.load_block_tile(b_n, b_p, 0, 0);

    // Main loop
    for (int compute_batch = 0; compute_batch < k_batch; compute_batch ++){
        if (fetch_batch < k_batch){
            global_iter_x.Load_async(fetch_batch);
            global_iter_w.Load_async(fetch_batch);
            fetch_batch ++;
            asm volatile ("cp.async.commit_group;");
        }
        #pragma unroll
        for (int k_step = 0; k_step < TileK / K; k_step ++){
            if (k_step < TileK / K - 1){
                shmem_iter_x.load_block_tile(a, compute_batch, k_step + 1);
                shmem_iter_w.load_block_tile(b_n, b_p, compute_batch, k_step + 1);
            }
            if (use_pos_weight) {
              compute_block_tile<M, N, K, MWarpTiles, NWarpTiles>(c, a, b_p, k_step % 2);
            } else {
              compute_block_tile<M, N, K, MWarpTiles, NWarpTiles>(c, a, b_n, k_step % 2);
            }
        }
        asm volatile ("cp.async.wait_group %0;" :: "n"(Stages - 2) );
        __syncthreads();
        shmem_iter_x.load_block_tile(a, compute_batch + 1, 0);
        shmem_iter_w.load_block_tile(b_n, b_p, compute_batch + 1, 0);
    }

    __syncthreads();

    // Store the D fragment to shared memory
    ResStore<MWarps, NWarps, wTileM, wTileN, TileM, TileN, M, N, K, 4, NumWarp> storer1(smem, y, m_offset, n_offset, n);
    storer1.store_block_res(c);

    __syncthreads();

    storer1.write_to_global(n);
}

__global__ void add_elementwisely(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int length
){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < length) {
    C[idx] = A[idx] + B[idx];
  }
}

template<int TileM, int TileN, int TileK, int wTileM, int wTileN, int M, int N, int K, int Stages>
__global__ void verify_matmul_w_gemm(
    const float* __restrict__ x_lw,
    const float* __restrict__ x_uw,
    const float* __restrict__ W,
    float* __restrict__ y_lw,
    float* __restrict__ y_uw,
    int batch_size_length, int dim_in, int dim_out, int dim_y_out){

    verify_matmul_<TileM, TileN, TileK, wTileM, wTileN, M, N, K, Stages>(x_lw, x_uw, W, y_lw, y_uw, 
      batch_size_length * dim_in, dim_out, dim_y_out, blockIdx.x);
}

template<int TileM, int TileN, int TileK, int wTileM, int wTileN, int M, int N, int K, int Stages>
__global__ void verify_matmul_w_gemm_unoptimized(
    const float* __restrict__ x,
    const float* __restrict__ W,
    float* __restrict__ y,
    int batch_size_length, int dim_in, int dim_out, int dim_y_out, bool use_pos_weight){

      verify_matmul_unoptimized<TileM, TileN, TileK, wTileM, wTileN, M, N, K, Stages>(x, W, y, 
      batch_size_length * dim_in, dim_out, dim_y_out, blockIdx.x, use_pos_weight);
}

template<int TileM, int TileN, int TileK, int wTileM, int wTileN, int M, int N, int K, int Stages>
__global__ void verify_matmul_b_w(
    const float* __restrict__ x_lb,
    const float* __restrict__ x_ub,
    const float* __restrict__ x_lw,
    const float* __restrict__ x_uw,
    const float* __restrict__ W,
    float* __restrict__ y_lb,
    float* __restrict__ y_ub,
    float* __restrict__ y_lw,
    float* __restrict__ y_uw,
    int batch_size_length, int dim_in, int dim_out, int dim_y_out){
    int border = batch_size_length * dim_in / TileM;
    if (blockIdx.x >= border){
        verify_matmul_<TileM, TileN, TileK, wTileM, wTileN, M, N, K, Stages>(x_lb, x_ub, W, y_lb, y_ub, 
          batch_size_length, dim_out, dim_y_out, blockIdx.x - border);
    } else{
        verify_matmul_<TileM, TileN, TileK, wTileM, wTileN, M, N, K, Stages>(x_lw, x_uw, W, y_lw, y_uw, 
          batch_size_length * dim_in, dim_out, dim_y_out, blockIdx.x);
    }
}

__global__ void verify_matmul_b(
    const float *x_lb, const float *x_ub, 
    const float *W,
    float *y_lb, float *y_ub,
    int batch_size_length, int dim_out, int dim_y_out
){
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int length_idx_i = block_pos / dim_y_out;
    const unsigned int dim_y_out_idx_j = block_pos % dim_y_out;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (length_idx_i >= batch_size_length) {
      break;
    }

    float y_lb_val = 0.0f;
    float y_ub_val = 0.0f;
    float x_lb_val, x_ub_val, w;

    for (int i=0; i<dim_out/32; i++) {
        x_lb_val = *(x_lb + length_idx_i*dim_out + i*32+threadIdx.x);
        x_ub_val = *(x_ub + length_idx_i*dim_out + i*32+threadIdx.x);
        w = *(W+dim_y_out_idx_j*dim_out + i*32 + threadIdx.x);
        if (w>0) {
            y_lb_val += w*x_lb_val;
            y_ub_val += w*x_ub_val;
        } else {
            y_lb_val += w*x_ub_val;
            y_ub_val += w*x_lb_val;
        }
    }

    int i = dim_out/32;
    if (i*32+threadIdx.x < dim_out) {
      x_lb_val = *(x_lb + length_idx_i*dim_out + i*32+threadIdx.x);
      x_ub_val = *(x_ub + length_idx_i*dim_out + i*32+threadIdx.x);
      w = *(W+dim_y_out_idx_j*dim_out + i*32 + threadIdx.x);
      if (w>0) {
          y_lb_val += w*x_lb_val;
          y_ub_val += w*x_ub_val;
      } else {
          y_lb_val += w*x_ub_val;
          y_ub_val += w*x_lb_val;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        y_lb_val += __shfl_down_sync(FULL_MASK, y_lb_val, offset);
        y_ub_val += __shfl_down_sync(FULL_MASK, y_ub_val, offset);
    }

    if (threadIdx.x == 0) {
      *(y_lb+length_idx_i*dim_y_out+dim_y_out_idx_j) = y_lb_val;
      *(y_ub+length_idx_i*dim_y_out+dim_y_out_idx_j) = y_ub_val;  
    }
    __syncthreads();
  }
}

// For medium-size matmul
__global__ void verify_matmul_w_medium(
    const float *x_lw, const float *x_uw,
    const float *W,
    float *y_lw, float *y_uw,
    int batch_size_length_dim_in, int dim_out, int dim_y_out
) {
  __shared__ float shmem[128*32];

  const int warpId = threadIdx.x/32;
  const int laneId = threadIdx.x%32;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos / (dim_y_out/64))*32;
    const unsigned int block_j = (block_pos % (dim_y_out/64))*64;

    if (block_i >= batch_size_length_dim_in) {
        break;
    }

    float y_lw_tmp[2][4];
    float y_uw_tmp[2][4];
    for (int i=0; i<2; i++) {
      for (int j=0; j<4; j++) {
        y_lw_tmp[i][j] = 0.0f;
        y_uw_tmp[i][j] = 0.0f;
      }
    }
    
    float w_val;
    float lw_val;
    float uw_val;
    int row_idx = (threadIdx.x/16)*2;
    int col_idx = (threadIdx.x%16)*4;

    for (unsigned int k=0; k<dim_out/32; k++) {
      // Stage I: Load data from GL to SHMEM
      // Read lw
      *(shmem+warpId*32+laneId) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
      *(shmem+warpId*32+laneId+8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
      *(shmem+warpId*32+laneId+16*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
      *(shmem+warpId*32+laneId+24*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);

      // Read uw
      *(shmem+warpId*32+laneId+32*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
      *(shmem+warpId*32+laneId+40*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
      *(shmem+warpId*32+laneId+48*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
      *(shmem+warpId*32+laneId+56*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);

      // // Read W
      *(shmem+warpId*32+laneId+64*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k);
      *(shmem+warpId*32+laneId+72*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
      *(shmem+warpId*32+laneId+80*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
      *(shmem+warpId*32+laneId+88*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);
      *(shmem+warpId*32+laneId+96*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 32*dim_out);
      *(shmem+warpId*32+laneId+104*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 40*dim_out);
      *(shmem+warpId*32+laneId+112*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 48*dim_out);
      *(shmem+warpId*32+laneId+120*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 56*dim_out);
      __syncthreads();

      // Stage II: Compute y_lw, y_uw
      for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
          for (int k=0; k<32; k++) {
            lw_val = *(shmem+(row_idx+i)*32+k);
            uw_val = *(shmem+(row_idx+i)*32+k+32*32);
            w_val = *(shmem + (col_idx+j)*32 + k + 64*32);
            if (w_val>0) {
              y_lw_tmp[i][j] += w_val * lw_val;
              y_uw_tmp[i][j] += w_val * uw_val;
            } else {
              y_lw_tmp[i][j] += w_val * uw_val;
              y_uw_tmp[i][j] += w_val * lw_val;    
            }
          }
        }
      }
      __syncthreads();
    }

    int k = dim_out/32;
    if (k*32+threadIdx.x < dim_out) {
      // Stage I: Load data from GL to SHMEM
      // Read lw
      *(shmem+warpId*32+laneId) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
      *(shmem+warpId*32+laneId+8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
      *(shmem+warpId*32+laneId+16*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
      *(shmem+warpId*32+laneId+24*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);

      // Read uw
      *(shmem+warpId*32+laneId+32*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
      *(shmem+warpId*32+laneId+40*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
      *(shmem+warpId*32+laneId+48*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
      *(shmem+warpId*32+laneId+56*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);

      // // Read W
      *(shmem+warpId*32+laneId+64*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k);
      *(shmem+warpId*32+laneId+72*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
      *(shmem+warpId*32+laneId+80*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
      *(shmem+warpId*32+laneId+88*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);
      *(shmem+warpId*32+laneId+96*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 32*dim_out);
      *(shmem+warpId*32+laneId+104*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 40*dim_out);
      *(shmem+warpId*32+laneId+112*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 48*dim_out);
      *(shmem+warpId*32+laneId+120*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 56*dim_out);
      __syncthreads();

      // Stage II: Compute y_lw, y_uw
      for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
          for (int k=0; k<32; k++) {
            lw_val = *(shmem+(row_idx+i)*32+k);
            uw_val = *(shmem+(row_idx+i)*32+k+32*32);
            w_val = *(shmem + (col_idx+j)*32 + k + 64*32);
            if (w_val>0) {
              y_lw_tmp[i][j] += w_val * lw_val;
              y_uw_tmp[i][j] += w_val * uw_val;
            } else {
              y_lw_tmp[i][j] += w_val * uw_val;
              y_uw_tmp[i][j] += w_val * lw_val;    
            }
          }
        }
      }
      __syncthreads();
    }

    for(int i=0; i<2; i++) {
      for (int j=0; j<4; j++) {
        *(y_lw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_lw_tmp[i][j];
        *(y_uw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_uw_tmp[i][j];
      }
    }
    __syncthreads();
  }
}

template <bool unoptimized>
void verify_matmul(int length, int dim_in, int dim_out, int dim_Y_out){
    int weight_size = dim_out * dim_Y_out;
    int xb_size = length * dim_out;
    int yb_size = length * dim_Y_out;
    int xw_size = length * dim_in * dim_out;
    int yw_size = length * dim_in * dim_Y_out;
    // Allocate the host tensors
    float* w_h = new float[weight_size];
    float* x_lb_h = new float[xb_size];
    float* x_ub_h = new float[xb_size];
    float* x_lw_h = new float[xw_size];
    float* x_uw_h = new float[xw_size];

    std::default_random_engine generator;

    MakeDenseMatrix(1, weight_size, w_h, generator);
    MakeDenseMatrix(1, xb_size, x_lb_h, generator);
    MakeDenseMatrix(1, xb_size, x_ub_h, generator);
    MakeDenseMatrix(1, xw_size, x_lw_h, generator);
    MakeDenseMatrix(1, xw_size, x_uw_h, generator);

    float* y_lb_h = new float[yb_size];
    float* y_ub_h = new float[yb_size];

    float* y_lw_h = new float[yw_size];
    float* y_uw_h = new float[yw_size];

    // Allocate the device tensors
    float *w_d, *x_lb_d, *x_ub_d, *y_lb_d, *y_ub_d, *x_lw_d, *x_uw_d, *y_lw_d, *y_uw_d, *T1_d, *T2_d, *T3_d, *T4_d;

    cudaMalloc(&w_d, weight_size * sizeof(float));
    cudaMalloc(&x_lb_d, xb_size * sizeof(float));
    cudaMalloc(&x_ub_d, xb_size * sizeof(float));
    cudaMalloc(&y_lb_d, yb_size * sizeof(float));
    cudaMalloc(&y_ub_d, yb_size * sizeof(float));
    cudaMalloc(&x_lw_d, xw_size * sizeof(float));
    cudaMalloc(&x_uw_d, xw_size * sizeof(float));
    cudaMalloc(&T1_d, yw_size * sizeof(float));
    cudaMalloc(&T2_d, yw_size * sizeof(float));
    cudaMalloc(&T3_d, yw_size * sizeof(float));
    cudaMalloc(&T4_d, yw_size * sizeof(float));
    cudaMalloc(&y_lw_d, yw_size * sizeof(float));
    cudaMalloc(&y_uw_d, yw_size * sizeof(float));

    // Copy to device
    cudaMemcpy(w_d, w_h, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_lb_d, x_lb_h, xb_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_ub_d, x_ub_h, xb_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_lw_d, x_lw_h, xw_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_uw_d, x_uw_h, xw_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Original verison
    if (unoptimized){
      // We don't fuse the two kernels here
      if (dim_Y_out >= 128){
        // int maxbytes = max(TileM * (TileN + 8) * sizeof(float), (2 * TileM + TileN) * TileK * Stages * sizeof(float));
        int maxbytes = max(128 * (128 + 8) * sizeof(float), (128 + 128) * 32 * 3 * sizeof(float));
        cudaFuncSetAttribute(verify_matmul_w_gemm_unoptimized<128, 128, 32, 64, 32, 16, 16, 8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

        dim3 grid_dim(length * dim_in / 128, dim_Y_out / 128, 1);
        dim3 block_dim(128 * 128 / 64 / 32 * 32, 1, 1);
        int add_elementwisely_grid_size = (int)ceil((float)yw_size/1024);

        cudaEventRecord(start);
        for (int i=0; i < 100; i++){
          verify_matmul_b<<<length * dim_Y_out, 32>>>(x_lb_d, x_ub_d, w_d, y_lb_d, y_ub_d, length, dim_out, dim_Y_out);
          verify_matmul_w_gemm_unoptimized<128, 128, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lw_d, w_d, T1_d, length, dim_in, dim_out, dim_Y_out, true);
          verify_matmul_w_gemm_unoptimized<128, 128, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_uw_d, w_d, T2_d, length, dim_in, dim_out, dim_Y_out, false);
          verify_matmul_w_gemm_unoptimized<128, 128, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_uw_d, w_d, T3_d, length, dim_in, dim_out, dim_Y_out, true);
          verify_matmul_w_gemm_unoptimized<128, 128, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lw_d, w_d, T4_d, length, dim_in, dim_out, dim_Y_out, false);
          add_elementwisely<<<add_elementwisely_grid_size, 1024>>>(T1_d, T2_d, y_lw_d, yw_size);
          add_elementwisely<<<add_elementwisely_grid_size, 1024>>>(T3_d, T4_d, y_uw_d, yw_size);
        }
      } else if (dim_Y_out == 64){
        int maxbytes = max(128 * (64 + 8) * sizeof(float), (2 * 128 + 64) * 32 * 3 * sizeof(float));
        cudaFuncSetAttribute(verify_matmul_w_gemm_unoptimized<128, 64, 32, 64, 32, 16, 16, 8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

        dim3 grid_dim(length * dim_in / 128, dim_Y_out / 64, 1);
        dim3 block_dim(128 * 64 / 64 / 32 * 32, 1, 1);
        int add_elementwisely_grid_size = (int)ceil((float)yw_size/1024);

        cudaEventRecord(start);
        for (int i=0; i < 100; i++){
          verify_matmul_b<<<length * dim_Y_out, 32>>>(x_lb_d, x_ub_d, w_d, y_lb_d, y_ub_d, length, dim_out, dim_Y_out);
          verify_matmul_w_gemm_unoptimized<128, 64, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lw_d, w_d, T1_d, length, dim_in, dim_out, dim_Y_out, true);
          verify_matmul_w_gemm_unoptimized<128, 64, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_uw_d, w_d, T2_d, length, dim_in, dim_out, dim_Y_out, false);
          verify_matmul_w_gemm_unoptimized<128, 64, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_uw_d, w_d, T3_d, length, dim_in, dim_out, dim_Y_out, true);
          verify_matmul_w_gemm_unoptimized<128, 64, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lw_d, w_d, T4_d, length, dim_in, dim_out, dim_Y_out, false);
          add_elementwisely<<<add_elementwisely_grid_size, 1024>>>(T1_d, T2_d, y_lw_d, yw_size);
          add_elementwisely<<<add_elementwisely_grid_size, 1024>>>(T3_d, T4_d, y_uw_d, yw_size);
        }
      }


    } else {
        if (length >= 128){
          if (dim_Y_out >= 128){
            // int maxbytes = max(TileM * (TileN + 8) * sizeof(float), (2 * TileM + TileN) * TileK * Stages * sizeof(float));
            int maxbytes = max(128 * (128 + 8) * sizeof(float), (2 * 128 + 128) * 32 * 3 * sizeof(float));
            cudaFuncSetAttribute(verify_matmul_b_w<128, 128, 32, 64, 32, 16, 16, 8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  
            dim3 grid_dim(length * (dim_in + 1) / 128, dim_Y_out / 128, 1);
            dim3 block_dim(128 * 128 / 64 / 32 * 32, 1, 1);
  
            cudaEventRecord(start);
            for (int i=0; i < 100; i++){
                verify_matmul_b_w<128, 128, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lb_d, x_ub_d, x_lw_d, x_uw_d, w_d, y_lb_d, y_ub_d, y_lw_d, y_uw_d, length, dim_in, dim_out, dim_Y_out);
            }
          } else if (dim_Y_out == 64){
            int maxbytes = max(128 * (64 + 8) * sizeof(float), (2 * 128 + 64) * 32 * 3 * sizeof(float));
            cudaFuncSetAttribute(verify_matmul_b_w<128, 64, 32, 64, 32, 16, 16, 8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  
            dim3 grid_dim(length * (dim_in + 1) / 128, dim_Y_out / 64, 1);
            dim3 block_dim(128 * 64 / 64 / 32 * 32, 1, 1);
  
            cudaEventRecord(start);
            for (int i=0; i < 100; i++){
                verify_matmul_b_w<128, 64, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lb_d, x_ub_d, x_lw_d, x_uw_d, w_d, y_lb_d, y_ub_d, y_lw_d, y_uw_d, length, dim_in, dim_out, dim_Y_out);
            }
          }
        } else {
          // We don't fuse the two kernels here
          if (dim_Y_out >= 128){
            // int maxbytes = max(TileM * (TileN + 8) * sizeof(float), (2 * TileM + TileN) * TileK * Stages * sizeof(float));
            int maxbytes = max(128 * (128 + 8) * sizeof(float), (2 * 128 + 128) * 32 * 3 * sizeof(float));
            cudaFuncSetAttribute(verify_matmul_w_gemm<128, 128, 32, 64, 32, 16, 16, 8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  
            dim3 grid_dim(length * dim_in / 128, dim_Y_out / 128, 1);
            dim3 block_dim(128 * 128 / 64 / 32 * 32, 1, 1);
  
            cudaEventRecord(start);
            for (int i=0; i < 100; i++){
              verify_matmul_b<<<length * dim_Y_out, 32>>>(x_lb_d, x_ub_d, w_d, y_lb_d, y_ub_d, length, dim_out, dim_Y_out);
              verify_matmul_w_gemm<128, 128, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lw_d, x_uw_d, w_d, y_lw_d, y_uw_d, length, dim_in, dim_out, dim_Y_out);
            }
          } else if (dim_Y_out == 64){
            int maxbytes = max(128 * (64 + 8) * sizeof(float), (2 * 128 + 64) * 32 * 3 * sizeof(float));
            cudaFuncSetAttribute(verify_matmul_w_gemm<128, 64, 32, 64, 32, 16, 16, 8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  
            dim3 grid_dim(length * dim_in / 128, dim_Y_out / 64, 1);
            dim3 block_dim(128 * 64 / 64 / 32 * 32, 1, 1);
  
            cudaEventRecord(start);
            for (int i=0; i < 100; i++){
              verify_matmul_b<<<length * dim_Y_out, 32>>>(x_lb_d, x_ub_d, w_d, y_lb_d, y_ub_d, length, dim_out, dim_Y_out);
              verify_matmul_w_gemm<128, 64, 32, 64, 32, 16, 16, 8, 3><<<grid_dim, block_dim, maxbytes>>>(x_lw_d, x_uw_d, w_d, y_lw_d, y_uw_d, length, dim_in, dim_out, dim_Y_out);
            }
          }
        }
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ElapsedTime: %.4f ms\n", milliseconds/100);

    // Load result back
    cudaMemcpy(y_lb_h, y_lb_d, yb_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_ub_h, y_ub_d, yb_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(y_lw_h, y_lw_d, yw_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_uw_h, y_uw_d, yw_size * sizeof(float), cudaMemcpyDeviceToHost);

    float* y_lb_h_ref = new float[yb_size];
    float* y_ub_h_ref = new float[yb_size];
    float* y_lw_h_ref = new float[yw_size];
    float* y_uw_h_ref = new float[yw_size];

    // Verify the result on host
    for (int i=0; i < length; i++){
        for (int j=0; j < dim_Y_out; j++){
            float tmp_l = 0;
            float tmp_u = 0;
            for (int k=0; k < dim_out; k++){
                if (w_h[j * dim_out + k] >= 0){
                    tmp_l += w_h[j * dim_out + k] * x_lb_h[i * dim_out + k];
                    tmp_u += w_h[j * dim_out + k] * x_ub_h[i * dim_out + k];
                } else {
                    tmp_l += w_h[j * dim_out + k] * x_ub_h[i * dim_out + k];
                    tmp_u += w_h[j * dim_out + k] * x_lb_h[i * dim_out + k];
                }
            }
            y_lb_h_ref[i * dim_Y_out + j] = tmp_l;
            y_ub_h_ref[i * dim_Y_out + j] = tmp_u;
        }
    }

    for (int i=0; i < length * dim_in; i++){
        for (int j=0; j < dim_Y_out; j++){
            float tmp_l = 0;
            float tmp_u = 0;
            for (int k=0; k < dim_out; k++){
                if (w_h[j * dim_out + k] >= 0){
                    tmp_l += w_h[j * dim_out + k] * x_lw_h[i * dim_out + k];
                    tmp_u += w_h[j * dim_out + k] * x_uw_h[i * dim_out + k];
                } else {
                    tmp_l += w_h[j * dim_out + k] * x_uw_h[i * dim_out + k];
                    tmp_u += w_h[j * dim_out + k] * x_lw_h[i * dim_out + k];
                }
            }
            y_lw_h_ref[i * dim_Y_out + j] = tmp_l;
            y_uw_h_ref[i * dim_Y_out + j] = tmp_u;
        }
    }

    int errors_lb = 0;
    int errors_ub = 0;
    for (int i=0; i < length * dim_Y_out; i++){
        if(abs(y_lb_h[i] - y_lb_h_ref[i]) > 0.02){
            errors_lb ++;
        }
        if(abs(y_ub_h[i] - y_ub_h_ref[i]) > 0.02){
            errors_ub ++;
        }
    }

    int errors_lw = 0;
    int errors_uw = 0;
    for (int i=0; i < length * dim_Y_out; i++){
        if(abs(y_lw_h[i] - y_lw_h_ref[i]) > 0.02){
            printf("%.4f, %.4f\n", y_lw_h[i], y_lw_h_ref[i]);
            errors_lw ++;
        }
        if(abs(y_uw_h[i] - y_uw_h_ref[i]) > 0.02){
            errors_uw ++;
        }
    }

    if (errors_lb > 0) {
        printf(
            "matmul lb does not agree with SEQUENTIAL! %d errors!\n",
            errors_lb);
    }else {
        printf("Results verified: matmul lb agree.\n");
    }

    if (errors_ub > 0) {
        printf(
            "matmul ub does not agree with SEQUENTIAL! %d errors!\n",
            errors_ub);
    }else {
        printf("Results verified: matmul ub agree.\n");
    }

    if (errors_lw > 0) {
        printf(
            "matmul lw does not agree with SEQUENTIAL! %d errors!\n",
            errors_lw);
    }else {
        printf("Results verified: matmul lw agree.\n");
    }

    if (errors_uw > 0) {
        printf(
            "matmul uw does not agree with SEQUENTIAL! %d errors!\n",
            errors_uw);
    }else {
        printf("Results verified: matmul uw agree.\n");
    }
}

int main(int argc, char **argv){
    // verify_matmul<true>(128, 128, 128, 128);
    // verify_matmul<false>(128, 128, 128, 128);

    // verify_matmul<true>(128, 64, 64, 64);
    // verify_matmul<false>(128, 64, 64, 64);

    // verify_matmul<true>(64, 64, 64, 64);
    // verify_matmul<false>(64, 64, 64, 64);

    // verify_matmul<true>(64, 128, 128, 128);
    // verify_matmul<false>(64, 128, 128, 128);

    // verify_matmul<true>(64, 256, 256, 256);
    // verify_matmul<false>(64, 256, 256, 256);

    // verify_matmul<true>(64, 512, 512, 512);
    // verify_matmul<false>(64, 512, 512, 512);

    // verify_matmul<true>(64, 1024, 1024, 1024);
    // verify_matmul<false>(64, 1024, 1024, 1024);

    // verify_matmul<true>(32, 64, 64, 64);
    // verify_matmul<false>(32, 64, 64, 64);

    // verify_matmul<true>(32, 128, 128, 128);
    // verify_matmul<false>(32, 128, 128, 128);

    // verify_matmul<true>(32, 256, 256, 256);
    // verify_matmul<false>(32, 256, 256, 256);

    // verify_matmul<true>(32, 512, 512, 512);
    // verify_matmul<false>(32, 512, 512, 512);

    // verify_matmul<true>(32, 1024, 1024, 1024);
    // verify_matmul<false>(32, 1024, 1024, 1024);

    // verify_matmul<true>(16, 64, 64, 64);
    // verify_matmul<false>(16, 64, 64, 64);

    // verify_matmul<true>(16, 128, 128, 128);
    // verify_matmul<false>(16, 128, 128, 128);

    // verify_matmul<true>(16, 256, 256, 256);
    // verify_matmul<false>(16, 256, 256, 256);

    // verify_matmul<true>(16, 512, 512, 512);
    // verify_matmul<false>(16, 512, 512, 512);

    // verify_matmul<true>(16, 1024, 1024, 1024);
    // verify_matmul<false>(16, 1024, 1024, 1024);

    // verify_matmul<true>(8, 64, 64, 64);
    // verify_matmul<false>(8, 64, 64, 64);

    // verify_matmul<true>(8, 128, 128, 128);
    // verify_matmul<false>(8, 128, 128, 128);

    // verify_matmul<true>(8, 256, 256, 256);
    // verify_matmul<false>(8, 256, 256, 256);

    // verify_matmul<true>(8, 512, 512, 512);
    // verify_matmul<false>(8, 512, 512, 512);

    // verify_matmul<true>(8, 1024, 1024, 1024);
    // verify_matmul<false>(8, 1024, 1024, 1024);

    // verify_matmul<true>(4, 64, 64, 64);
    // verify_matmul<false>(4, 64, 64, 64);

    // verify_matmul<true>(4, 128, 128, 128);
    // verify_matmul<false>(4, 128, 128, 128);

    // verify_matmul<true>(4, 256, 256, 256);
    // verify_matmul<false>(4, 256, 256, 256);

    // verify_matmul<true>(4, 512, 512, 512);
    // verify_matmul<false>(4, 512, 512, 512);

    // verify_matmul<true>(4, 1024, 1024, 1024);
    // verify_matmul<false>(4, 1024, 1024, 1024);

    // for (int length=2; length<=128; length*=2) {
    //   printf("length: %d, dim_in: %d, dim_out: %d, dim_Y_out: %d\n", length, 128, 128, 128);
    //   printf("Unoptimized: ");
    //   verify_matmul<true>(length, 128, 128, 128);
    //   printf("Optimized: ");
    //   verify_matmul<false>(length, 128, 128, 128);  
    // }
    
    // for (int idx=1; idx<=10; idx+=1) {
      // int dim = idx*64;
    // for (int dim = 64; dim < 1024; dim*= 2) {
    //   printf("length: %d, dim_in: %d, dim_out: %d, dim_Y_out: %d\n", 16, dim, dim, dim);
    //   printf("Unoptimized: ");
    //   verify_matmul<true>(16, dim, dim, dim);
    //   printf("Optimized: ");
    //   verify_matmul<false>(16, dim, dim, dim);  
    // }
    
    int dim=640;
    printf("Unoptimized: ");
    verify_matmul<true>(16, dim, dim, dim);
    printf("Optimized: ");
    verify_matmul<false>(16, dim, dim, dim);  

    return 0;
}