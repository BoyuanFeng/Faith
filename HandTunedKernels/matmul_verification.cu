/*
Command to compile on Winnie, A6000:
  nvcc -arch=sm_75 -o matmul_verification matmul_verification.cu
Command to run on Winnie, A6000:
  ./matmul_verification
Note: Current implementation assumes:
  1) dim_y_out is a multiplier of 32
  2) length*dim_in is a multiplier of 16
  3) dim_out is a multiplier of 32
*/

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include "../tvm_kernels/cuda_kernel/mma/mma_kernel.h"
#include "matmul_config.h"
// GPU configuration.

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define SKEW 0

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

// using namespace nvcuda;

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
        // if (block_pos==0) {
        //   printf("threadIdx.x: %d, w: %f, x_lb_val: %f, x_ub_val: %f, y_lb_val: %f\n", threadIdx.x, w, x_lb_val, x_ub_val, y_lb_val);
        // }
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
    // if (block_pos==0 && threadIdx.x == 0) {
    //   printf("y_lb_val: %f\n", y_lb_val);
    // }

    if (threadIdx.x == 0) {
      *(y_lb+length_idx_i*dim_y_out+dim_y_out_idx_j) = y_lb_val;
      *(y_ub+length_idx_i*dim_y_out+dim_y_out_idx_j) = y_ub_val;  
    }
    __syncthreads();
  }
}

__global__ void verify_matmul_w_small(
    const float *x_lw, const float *x_uw,
    const float *W,
    float *y_lw, float *y_uw,
    int batch_size_length_dim_in, int dim_out, int dim_y_out
) {
    __shared__ float shmem[64*32];

    const int warpId = threadIdx.x/32;
    const int laneId = threadIdx.x%32;

    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
      const unsigned int block_i = (block_pos / (dim_y_out/32))*16;
      const unsigned int block_j = (block_pos % (dim_y_out/32))*32;

      if (block_i >= batch_size_length_dim_in) {
          break;
      }

      float y_lw_val0 = 0.0f;
      float y_lw_val1 = 0.0f;
      float y_uw_val0 = 0.0f;
      float y_uw_val1 = 0.0f;
      int row_idx = threadIdx.x/16;
      int col_idx = (threadIdx.x%16)*2;

      for (unsigned int k=0; k<dim_out/32; k++) {
        // Stage I: Load data from GL to SHMEM
        // Read lw
        *(shmem+warpId*32+laneId) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
        *(shmem+warpId*32+laneId+8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);

        // Read uw
        *(shmem+warpId*32+laneId+16*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
        *(shmem+warpId*32+laneId+24*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);

        // // Read W
        *(shmem+warpId*32+laneId+32*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k);
        *(shmem+warpId*32+laneId+40*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
        *(shmem+warpId*32+laneId+48*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
        *(shmem+warpId*32+laneId+56*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);
        __syncthreads();

        // Stage II: Compute y_lw, y_uw
        float w_val, lw_val, uw_val;

        for (int i=0; i<32; i++) {
          lw_val = *(shmem + row_idx*32 + i);
          uw_val = *(shmem + row_idx*32 + i + 16*32);
          w_val = *(shmem + col_idx*32 + i + 32*32);
          if (w_val>0) {
            y_lw_val0 += w_val * lw_val;
            y_uw_val0 += w_val * uw_val;
          } else {
            y_lw_val0 += w_val * uw_val;
            y_uw_val0 += w_val * lw_val;
          }

          w_val = *(shmem + (col_idx+1)*32 + i + 32*32);
          if (w_val>0) {
            y_lw_val1 += w_val * lw_val;
            y_uw_val1 += w_val * uw_val;
          } else {
            y_lw_val1 += w_val * uw_val;
            y_uw_val1 += w_val * lw_val;
          }
        }
      }

      int k=dim_out/32;
      if (k*32+threadIdx.x < dim_out) {
        // Stage I: Load data from GL to SHMEM
        // Read lw
        *(shmem+warpId*32+laneId) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
        *(shmem+warpId*32+laneId+8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);

        // Read uw
        *(shmem+warpId*32+laneId+16*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k);
        *(shmem+warpId*32+laneId+24*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);

        // // Read W
        *(shmem+warpId*32+laneId+32*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k);
        *(shmem+warpId*32+laneId+40*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 8*dim_out);
        *(shmem+warpId*32+laneId+48*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 16*dim_out);
        *(shmem+warpId*32+laneId+56*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + 24*dim_out);
        __syncthreads();

        // Stage II: Compute y_lw, y_uw
        float w_val, lw_val, uw_val;

        for (int i=0; i<32; i++) {
          lw_val = *(shmem + row_idx*32 + i);
          uw_val = *(shmem + row_idx*32 + i + 16*32);
          w_val = *(shmem + col_idx*32 + i + 32*32);
          if (w_val>0) {
            y_lw_val0 += w_val * lw_val;
            y_uw_val0 += w_val * uw_val;
          } else {
            y_lw_val0 += w_val * uw_val;
            y_uw_val0 += w_val * lw_val;
          }

          w_val = *(shmem + (col_idx+1)*32 + i + 32*32);
          if (w_val>0) {
            y_lw_val1 += w_val * lw_val;
            y_uw_val1 += w_val * uw_val;
          } else {
            y_lw_val1 += w_val * uw_val;
            y_uw_val1 += w_val * lw_val;
          }
        }
      }

      *(y_lw + block_i*dim_y_out + block_j + row_idx*dim_y_out + col_idx) = y_lw_val0;
      *(y_lw + block_i*dim_y_out + block_j + row_idx*dim_y_out + col_idx + 1) = y_lw_val1;
      *(y_uw + block_i*dim_y_out + block_j + row_idx*dim_y_out + col_idx) = y_uw_val0;
      *(y_uw + block_i*dim_y_out + block_j + row_idx*dim_y_out + col_idx + 1) = y_uw_val1;
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


// For large-size matmul
__global__ void verify_matmul_w_large(
  const float *x_lw, const float *x_uw,
  const float *W,
  float *y_lw, float *y_uw,
  int batch_size_length_dim_in, int dim_out, int dim_y_out
) {
  __shared__ float shmem[256*32];

  const int warpId = threadIdx.x/32;
  const int laneId = threadIdx.x%32;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos / (dim_y_out/128))*64;
    const unsigned int block_j = (block_pos % (dim_y_out/128))*128;

    if (block_i >= batch_size_length_dim_in) {
        break;
    }

    float y_lw_tmp[4][8];
    float y_uw_tmp[4][8];
    for (int i=0; i<4; i++) {
      for (int j=0; j<8; j++) {
        y_lw_tmp[i][j] = 0.0f;
        y_uw_tmp[i][j] = 0.0f;
      }
    }
    
    float w_val[8];
    float lw_val[4];
    float uw_val[4];
    int row_idx = (threadIdx.x/16)*4;
    int col_idx = (threadIdx.x%16)*8;

    for (unsigned int k=0; k<dim_out/32; k++) {
      // Stage I: Load data from GL to SHMEM
      // Read lw
#pragma unroll
      for (int i=0; i<8; i++) {
        *(shmem+warpId*32+laneId+i*8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // Read uw
#pragma unroll
      for (int i=0; i<8; i++) {
        *(shmem+warpId*32+laneId+64*32+i*8*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // // Read W
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+128*32+i*8*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }
      __syncthreads();

      // Stage II: Compute y_lw, y_uw
      for (int k=0; k<32; k++) {
        for (int i=0; i<4; i++) {
          lw_val[i] = *(shmem+(row_idx+i)*32+k); 
          uw_val[i] = *(shmem+(row_idx+i)*32+k+64*32);
        }
        for (int j=0; j<8; j++) {
          w_val[j] = *(shmem + (col_idx+j)*32 + k + 128*32);
        }

        for (int i=0; i<4; i++) {
          for (int j=0; j<8; j++) {
            if (w_val>0) {
              y_lw_tmp[i][j] += w_val[j] * lw_val[i];
              y_uw_tmp[i][j] += w_val[j] * uw_val[i];
            } else {
              y_lw_tmp[i][j] += w_val[j] * uw_val[i];
              y_uw_tmp[i][j] += w_val[j] * lw_val[i];
            }
          }
        }
      }
    }

    int k = dim_out/32;
    if (k*32+threadIdx.x < dim_out) {
      // Stage I: Load data from GL to SHMEM
      // Read lw
      #pragma unroll
      for (int i=0; i<8; i++) {
        *(shmem+warpId*32+laneId+i*8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // Read uw
#pragma unroll
      for (int i=0; i<8; i++) {
        *(shmem+warpId*32+laneId+64*32+i*8*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // // Read W
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+128*32+i*8*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }
      __syncthreads();

      // Stage II: Compute y_lw, y_uw
      for (int k=0; k<32; k++) {
        for (int i=0; i<4; i++) {
          lw_val[i] = *(shmem+(row_idx+i)*32+k); 
          uw_val[i] = *(shmem+(row_idx+i)*32+k+64*32);
        }
        for (int j=0; j<8; j++) {
          w_val[j] = *(shmem + (col_idx+j)*32 + k + 128*32);
        }

        for (int i=0; i<4; i++) {
          for (int j=0; j<8; j++) {
            if (w_val>0) {
              y_lw_tmp[i][j] += w_val[j] * lw_val[i];
              y_uw_tmp[i][j] += w_val[j] * uw_val[i];
            } else {
              y_lw_tmp[i][j] += w_val[j] * uw_val[i];
              y_uw_tmp[i][j] += w_val[j] * lw_val[i];
            }
          }
        }
      }
    }

    for(int i=0; i<4; i++) {
      for (int j=0; j<8; j++) {
        *(y_lw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_lw_tmp[i][j];
        *(y_uw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_uw_tmp[i][j];
      }
    }
    __syncthreads();
  }
}


// Tiling size: 128x64.
__global__ void verify_matmul_w_large_128_64(
  const float *x_lw, const float *x_uw,
  const float *W,
  float *y_lw, float *y_uw,
  int batch_size_length_dim_in, int dim_out, int dim_y_out
) {
  __shared__ float shmem[192*32];

  const int warpId = threadIdx.x/32;
  const int laneId = threadIdx.x%32;

  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   printf("x_lw: ");
  //   for (int i=0; i < dim_out; i++) {
  //     printf("%f, ", *(x_lw+dim_out+i));
  //   }

  //   printf("\nW: ");
  //   for (int i=0; i < dim_out; i++) {
  //     printf("%f, ", *(W+i));
  //   }
  //   printf("\n");
  // }

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos / (dim_y_out/64))*64;
    const unsigned int block_j = (block_pos % (dim_y_out/64))*64;

    if (block_i >= batch_size_length_dim_in) {
        break;
    }

    float y_lw_tmp[4][4];
    float y_uw_tmp[4][4];
    for (int i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
        y_lw_tmp[i][j] = 0.0f;
        y_uw_tmp[i][j] = 0.0f;
      }
    }
    
    float w_val[4];
    float lw_val[4];
    float uw_val[4];
    int row_idx = (threadIdx.x/16)*4;
    int col_idx = (threadIdx.x%16)*4;

    for (unsigned int k=0; k<dim_out/32; k++) {
      // Stage I: Load data from GL to SHMEM
      // Read lw
#pragma unroll
      for (int i=0; i<8; i++) {
        *(shmem+warpId*32+laneId+i*8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // Read uw
#pragma unroll
      for (int i=0; i<8; i++) {
        *(shmem+warpId*32+laneId+64*32+i*8*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // // Read W
#pragma unroll
      for (int i=0; i<8; i++) {
        *(shmem+warpId*32+laneId+128*32+i*8*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }
      __syncthreads();

        // if (block_pos == 0 && threadIdx.x == 0) {
        //   printf("in shmem. x_lw: ");
        //   for (int i=0; i < 32; i++) {
        //     printf("%f, ", *(shmem+i));
        //   }

        //   printf("\n in shmem. W: ");
        //   for (int i=0; i < 32; i++) {
        //     float val = *(shmem+i+128*32);
        //     if (val <0) {
        //       val = 0.0f;
        //     }
        //     printf("%f, ", val);
        //   }
        //   printf("\n");
        // }

      // Stage II: Compute y_lw, y_uw
      for (int k=0; k<32; k++) {
        for (int i=0; i<4; i++) {
          lw_val[i] = *(shmem+(row_idx+i)*32+k); 
          uw_val[i] = *(shmem+(row_idx+i)*32+k+64*32);
        }
        for (int j=0; j<4; j++) {
          w_val[j] = *(shmem + (col_idx+j)*32 + k + 128*32);
        }

        for (int i=0; i<4; i++) {
          for (int j=0; j<4; j++) {
            if (w_val[j]>0) {
              y_lw_tmp[i][j] += w_val[j] * lw_val[i];
              y_uw_tmp[i][j] += w_val[j] * uw_val[i];
            } else {
              y_lw_tmp[i][j] += w_val[j] * uw_val[i];
              y_uw_tmp[i][j] += w_val[j] * lw_val[i];
            }
            // if (block_pos == 0 && threadIdx.x == 0) {
            //   printf("k: %d, y_lw_tmp[%d][%d]: %f, w_val[%d]: %f, lw_val[%d]: %f\n", k, i,j,y_lw_tmp[i][j], i, w_val[i], j, lw_val[j]);
            // }      
          }
        }
      }
      __syncthreads();
    }

    // for(int i=0; i<4; i++) {
    //   for (int j=0; j<4; j++) {
    //     printf("y_lw_tmp[%d][%d]: %f\n", i, j, y_lw_tmp[i][j]);
    //   }
    // }

    for(int i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
        *(y_lw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_lw_tmp[i][j];
        *(y_uw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_uw_tmp[i][j];
      }
    }
    __syncthreads();
  }
}


// For huge-size matmul. Tiling size: 256x128.
__global__ void verify_matmul_w_huge(
  const float *x_lw, const float *x_uw,
  const float *W,
  float *y_lw, float *y_uw,
  int batch_size_length_dim_in, int dim_out, int dim_y_out
) {
  __shared__ float shmem[384*32];

  const int warpId = threadIdx.x/32;
  const int laneId = threadIdx.x%32;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos / (dim_y_out/128))*128;
    const unsigned int block_j = (block_pos % (dim_y_out/128))*128;

    if (block_i >= batch_size_length_dim_in) {
        break;
    }

    float y_lw_tmp[8][8];
    float y_uw_tmp[8][8];
    for (int i=0; i<8; i++) {
      for (int j=0; j<8; j++) {
        y_lw_tmp[i][j] = 0.0f;
        y_uw_tmp[i][j] = 0.0f;
      }
    }
    
    float w_val[8];
    float lw_val[8];
    float uw_val[8];
    int row_idx = (threadIdx.x/16)*8;
    int col_idx = (threadIdx.x%16)*8;

    for (unsigned int k=0; k<dim_out/32; k++) {
      // Stage I: Load data from GL to SHMEM
      // Read lw
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+i*8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // Read uw
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+128*32+i*8*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // // Read W
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+256*32+i*8*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }
      __syncthreads();

      // Stage II: Compute y_lw, y_uw
      for (int k=0; k<32; k++) {
        for (int i=0; i<8; i++) {
          lw_val[i] = *(shmem+(row_idx+i)*32+k); 
          uw_val[i] = *(shmem+(row_idx+i)*32+k+128*32);
        }
        for (int j=0; j<8; j++) {
          w_val[j] = *(shmem + (col_idx+j)*32 + k + 256*32);
        }

        for (int i=0; i<8; i++) {
          for (int j=0; j<8; j++) {
            if (w_val>0) {
              y_lw_tmp[i][j] += w_val[j] * lw_val[i];
              y_uw_tmp[i][j] += w_val[j] * uw_val[i];
            } else {
              y_lw_tmp[i][j] += w_val[j] * uw_val[i];
              y_uw_tmp[i][j] += w_val[j] * lw_val[i];
            }
          }
        }
      }
    }

    int k = dim_out/32;
    if (k*32+threadIdx.x < dim_out) {
      // Stage I: Load data from GL to SHMEM
      // Read lw
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+i*8*32) = *(x_lw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // Read uw
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+128*32+i*8*32) = *(x_uw + block_i*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }

      // // Read W
#pragma unroll
      for (int i=0; i<16; i++) {
        *(shmem+warpId*32+laneId+256*32+i*8*32) = *(W + block_j*dim_out + warpId*dim_out + laneId + 32*k + i*8*dim_out);
      }
      __syncthreads();

      // Stage II: Compute y_lw, y_uw
      for (int k=0; k<32; k++) {
        for (int i=0; i<8; i++) {
          lw_val[i] = *(shmem+(row_idx+i)*32+k); 
          uw_val[i] = *(shmem+(row_idx+i)*32+k+128*32);
        }
        for (int j=0; j<8; j++) {
          w_val[j] = *(shmem + (col_idx+j)*32 + k + 256*32);
        }

        for (int i=0; i<8; i++) {
          for (int j=0; j<8; j++) {
            if (w_val>0) {
              y_lw_tmp[i][j] += w_val[j] * lw_val[i];
              y_uw_tmp[i][j] += w_val[j] * uw_val[i];
            } else {
              y_lw_tmp[i][j] += w_val[j] * uw_val[i];
              y_uw_tmp[i][j] += w_val[j] * lw_val[i];
            }
          }
        }
      }
    }

    for(int i=0; i<8; i++) {
      for (int j=0; j<8; j++) {
        *(y_lw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_lw_tmp[i][j];
        *(y_uw +(block_i+row_idx+i)*dim_y_out + block_j + col_idx + j) = y_uw_tmp[i][j];
      }
    }
    __syncthreads();
  }
}

// void call_matmul_verification(  
//   const float *x_lb, const float *x_ub, 
//   const float *x_lw, const float *x_uw,
//   const float *W,
//   float *y_lb, float *y_ub, float *y_lw, float *y_uw,
//   int batch_size, int length, int dim_in, int dim_out, int dim_y_out){

//     cudaDeviceProp deviceProp;
//     checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));
//     verify_matmul_b<<<32*deviceProp.multiProcessorCount, 32>>>(x_lb, x_ub,
//       W,
//       y_lb, y_ub,
//       batch_size*length, dim_out, dim_y_out
//     );

//     verify_matmul_w_large_128_64<<<32*deviceProp.multiProcessorCount, 32*8>>>(x_lw, x_uw,
//       W,
//       y_lw, y_uw,
//       batch_size*length*dim_in, dim_out, dim_y_out
//     );
// }

void call_matmul_verification(
  float *x_lb, float *x_ub, 
  float *x_lw, float *x_uw,
  float *W,
  float *y_lb, float *y_ub, float *y_lw, float *y_uw,
  int batch_size, int length, int dim_in, int dim_out, int dim_y_out){
  
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  // unfused version
  // verify_matmul_b<<<32*deviceProp.multiProcessorCount, 32>>>(x_lb, x_ub,
  //   W,
  //   y_lb, y_ub,
  //   batch_size*length, dim_out, dim_y_out
  // );

  // verify_matmul_fn<ThreadblockShape, WarpShape, NumStages>(
  //   batch_size*length*dim_in, dim_y_out, dim_out,
  //   x_lw, x_uw, W, y_lw, y_uw
  // );

  // fused version
  verify_matmul_fuse_fn<ThreadblockShape, WarpShape, NumStages>(
    batch_size*length, dim_in, dim_y_out, dim_out,
    x_lb, x_ub, W, y_lb, y_ub,
    x_lw, x_uw, y_lw, y_uw
  );
}


void test_matmul_small() {
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  for (int batch_size=1; batch_size<=1; batch_size*=2) {
    for (int length = 2; length <= 128; length*=2) {
      for (int dim_out=64; dim_out <= 1024; dim_out*=2) {
         int dim_y_out = dim_out;
         int dim_in = dim_out;
   
         float *x_lb, *x_ub, *x_lw, *x_uw, *W, *y_lb, *y_ub, *y_lw, *y_uw;
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * batch_size * length * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * batch_size * length * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * batch_size * length * dim_y_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * batch_size * length * dim_y_out));
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * batch_size * length * dim_in * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * batch_size * length * dim_in * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * batch_size * length * dim_in * dim_y_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * batch_size * length * dim_in * dim_y_out));
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&W), sizeof(float) * dim_y_out * dim_out));
   
         // Run ours NUM_PROFILES times and record time.
         float bmma_ms_avg = 0.0f;
         int NUM_PROFILES = 50;
         for(int iter=0; iter<NUM_PROFILES; ++iter){
                 float bmma_ms = 0.0f;
                 cudaEvent_t bmma_start;
                 cudaEvent_t bmma_end;
                 cudaEventCreate(&bmma_start);
                 cudaEventCreate(&bmma_end);
                 cudaEventRecord(bmma_start);
                 checkKernelErrors(
                   (verify_matmul_b<<<32*deviceProp.multiProcessorCount, 32>>>(x_lb, x_ub,
                     W,
                     y_lb, y_ub,
                     batch_size*length, dim_out, dim_y_out
                   )));
                 checkKernelErrors(
                   (verify_matmul_w_small<<<32*deviceProp.multiProcessorCount, 32*8>>>(x_lw, x_uw,
                       W,
                       y_lw, y_uw,
                       batch_size*length*dim_in, dim_out, dim_y_out
                   )));
                 cudaEventRecord(bmma_end);
                 cudaEventSynchronize(bmma_end);
                 cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
                 cudaEventDestroy(bmma_start);
                 cudaEventDestroy(bmma_end);
                 bmma_ms_avg += bmma_ms;
         }
       
         bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
     
         printf("test_matmul_small. batch_size: %d, length: %d, dim_out: %d, dim_y_out: %d\n", batch_size, length, dim_out, dim_y_out);
         printf("Time: %f ms\n", bmma_ms_avg);  
       
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_lb)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_ub)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_lb)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_ub)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(W)));
       }
     }   
  }
}

// (1*16*256)x256, 256x256
// 4096x256, 256x256
// If 16x32 tiling, we have 2048 tiles
// If 32x64 tiling, we have 512 tiles


void test_matmul_large() {
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  for (int batch_size=1; batch_size<=1; batch_size*=2) {
    for (int length = 2; length <= 128; length*=2) {
      for (int dim_out=64; dim_out <= 1024; dim_out*=2) {
  // for (int batch_size=1; batch_size<=1; batch_size*=2) {
  //   for (int length = 2; length <= 2; length*=2) {
  //     for (int dim_out=64; dim_out <= 64; dim_out*=2) {
          int dim_y_out = dim_out;
         int dim_in = dim_out;
   
         float *x_lb, *x_ub, *x_lw, *x_uw, *W, *y_lb, *y_ub, *y_lw, *y_uw;
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * batch_size * length * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * batch_size * length * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * batch_size * length * dim_y_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * batch_size * length * dim_y_out));
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * batch_size * length * dim_in * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * batch_size * length * dim_in * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * batch_size * length * dim_in * dim_y_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * batch_size * length * dim_in * dim_y_out));
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&W), sizeof(float) * dim_y_out * dim_out));
   
         // Run ours NUM_PROFILES times and record time.
         float bmma_ms_avg = 0.0f;
         int NUM_PROFILES = 100;
         for(int iter=0; iter<NUM_PROFILES; ++iter){
                 float bmma_ms = 0.0f;
                 cudaEvent_t bmma_start;
                 cudaEvent_t bmma_end;
                 cudaEventCreate(&bmma_start);
                 cudaEventCreate(&bmma_end);
                 cudaEventRecord(bmma_start);
                 checkKernelErrors(
                   (verify_matmul_b<<<32*deviceProp.multiProcessorCount, 32>>>(x_lb, x_ub,
                     W,
                     y_lb, y_ub,
                     batch_size*length, dim_out, dim_y_out
                   )));
                 checkKernelErrors(
                   (verify_matmul_w_large_128_64<<<32*deviceProp.multiProcessorCount, 32*8>>>(x_lw, x_uw,
                       W,
                       y_lw, y_uw,
                       batch_size*length*dim_in, dim_out, dim_y_out
                   )));
                 cudaEventRecord(bmma_end);
                 cudaEventSynchronize(bmma_end);
                 cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
                 cudaEventDestroy(bmma_start);
                 cudaEventDestroy(bmma_end);
                 bmma_ms_avg += bmma_ms;
         }
       
         bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
     
         printf("test_matmul_large. batch_size: %d, length: %d, dim_out: %d, dim_y_out: %d\n", batch_size, length, dim_out, dim_y_out);
         printf("Time: %f ms\n", bmma_ms_avg);  
       
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_lb)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_ub)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_lb)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_ub)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(W)));
       }
     }   
  }
}


void test_matmul_template() {
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  for (int batch_size=1; batch_size<=1; batch_size*=2) {
    for (int length = 2; length <= 128; length*=2) {
      for (int dim_out=128; dim_out <= 128; dim_out*=2) {
  // for (int batch_size=1; batch_size<=1; batch_size*=2) {
  //   for (int length = 2; length <= 2; length*=2) {
  //     for (int dim_out=64; dim_out <= 64; dim_out*=2) {
          int dim_y_out = dim_out;
         int dim_in = dim_out;
   
         float *x_lb, *x_ub, *x_lw, *x_uw, *W, *y_lb, *y_ub, *y_lw, *y_uw;
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * batch_size * length * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * batch_size * length * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * batch_size * length * dim_y_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * batch_size * length * dim_y_out));
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * batch_size * length * dim_in * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * batch_size * length * dim_in * dim_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * batch_size * length * dim_in * dim_y_out));
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * batch_size * length * dim_in * dim_y_out));
   
         checkKernelErrors(
             cudaMalloc(reinterpret_cast<void **>(&W), sizeof(float) * dim_y_out * dim_out));
   
         // Run ours NUM_PROFILES times and record time.
         float bmma_ms_avg = 0.0f;
         int NUM_PROFILES = 100;
         for(int iter=0; iter<NUM_PROFILES; ++iter){
                 float bmma_ms = 0.0f;
                 cudaEvent_t bmma_start;
                 cudaEvent_t bmma_end;
                 cudaEventCreate(&bmma_start);
                 cudaEventCreate(&bmma_end);
                 cudaEventRecord(bmma_start);
                  verify_matmul_fuse_fn<ThreadblockShape, WarpShape, NumStages>(
                    batch_size*length, dim_in, dim_y_out, dim_out,
                    x_lb, x_ub, W, y_lb, y_ub,
                    x_lw, x_uw, y_lw, y_uw
                  );
                 cudaEventRecord(bmma_end);
                 cudaEventSynchronize(bmma_end);
                 cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
                 cudaEventDestroy(bmma_start);
                 cudaEventDestroy(bmma_end);
                 bmma_ms_avg += bmma_ms;
         }
       
         bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
     
         printf("test_matmul_template. batch_size: %d, length: %d, dim_out: %d, dim_y_out: %d\n", batch_size, length, dim_out, dim_y_out);
         printf("Time: %f ms\n", bmma_ms_avg);  
       
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_lb)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_ub)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_lb)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_ub)));
         checkKernelErrors(cudaFree(reinterpret_cast<void *>(W)));
       }
     }   
  }
}

int main(int argc, char **argv) {
  // test_matmul_small();
  // test_matmul_large();
  test_matmul_template();
  return EXIT_SUCCESS;
}
