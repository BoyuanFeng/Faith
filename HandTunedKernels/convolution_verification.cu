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

__device__ bool is_valid(int row, int col, int Width, int Height){
  if ((row<0) || (col <0) || (row>=Height) || (col >= Width)) {
    return 0;
  } else {
    return 1;
  }
}

// using namespace nvcuda;
__global__ void verify_conv(
    const float *x_lb, const float *x_ub, 
    const float *W,
    float *y_lb, float *y_ub,
    int Batch_size, int Height, int Width, int CIN, int COUT, int K, int Stride, int padding_size
){
  // First 16*32 for lb; second 16*32 for ub; third 16*32 for W
  __shared__ float shmem[48*32];

  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int batch_idx = (block_pos/(COUT/16)) / (Width/4*Height/4);
    const unsigned int block_i = ((block_pos/(COUT/16))%(Width/4*Height/4)) / (Width/4) * 4;
    const unsigned int block_j = ((block_pos/(COUT/16))%(Width/4*Height/4)) % (Width/4) * 4;
    const unsigned int block_z = (block_pos % (COUT/16)) * 16;
    if (batch_idx >= Batch_size) {
      break;
    }
    float y_lb_val = 0;
    float y_ub_val = 0;
    int c_out_idx = threadIdx.x % 16;
    int out_row_idx = (threadIdx.x / 16)/4;
    int out_col_idx = (threadIdx.x / 16)%4;

    for (unsigned int tile_k=0; tile_k+32<K*K*CIN; tile_k+=32) {
      int k = tile_k + laneId;
      int cin_idx = k/(K*K);
      int k1_idx = (k%(K*K))/K;
      int k2_idx = k%K;

      // Stage I: Load data from GL to SHMEM
      // Read lb
      int row_idx = warpId/4*Stride-padding_size;
      int col_idx = warpId%4*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+warpId*32+laneId) = *(x_lb + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+warpId*32+laneId) = 0.0f;
      }

      row_idx = (warpId+8)/4;
      col_idx = (warpId+8)%4;
      row_idx = row_idx*Stride-padding_size;
      col_idx = col_idx*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+(8+warpId)*32+laneId) = *(x_lb + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+(8+warpId)*32+laneId) = 0.0f;
      }

      // Read ub
      row_idx = warpId/4;
      col_idx = warpId%4;
      row_idx = row_idx*Stride-padding_size;
      col_idx = col_idx*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+(16+warpId)*32+laneId) = *(x_ub + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+(16+warpId)*32+laneId) = 0.0f;
      }

      row_idx = (warpId+8)/4;
      col_idx = (warpId+8)%4;
      row_idx = row_idx*Stride-padding_size;
      col_idx = col_idx*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+(24+warpId)*32+laneId) = *(x_ub + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+(24+warpId)*32+laneId) = 0.0f;
      }

      // Read W
      *(shmem+(32+warpId)*32+laneId) = *(W + (block_j+warpId)*CIN*K*K + cin_idx*K*K + k1_idx*K + k2_idx);
      *(shmem+(40+warpId)*32+laneId) = *(W + (block_j+warpId+8)*CIN*K*K + cin_idx*K*K + k1_idx*K + k2_idx);
      __syncthreads();

      // Stage II: Compute y_lb, y_ub

      float w_val, lb_val, ub_val;
      for (int i=0; i<32; i++) {
        lb_val = *(shmem + (out_row_idx*4+out_col_idx)*32+i);
        ub_val = *(shmem + (out_row_idx*4+out_col_idx)*32+i + 16*32);
        w_val = *(shmem + c_out_idx*32 + i + 32*32);
        if (w_val>0) {
          y_lb_val += w_val * lb_val;
          y_ub_val += w_val * ub_val;
        } else {
          y_lb_val += w_val * ub_val;
          y_ub_val += w_val * lb_val;
        }
      }
    }

    // Handle irregular input shapes
    int tile_k = ((K*K*CIN)/32)*32;
    int k = tile_k + laneId;
    if (k < K*K*CIN) {
      int cin_idx = k/(K*K);
      int k1_idx = (k%(K*K))/K;
      int k2_idx = k%K;

      // Stage I: Load data from GL to SHMEM
      // Read lb
      int row_idx = warpId/4*Stride-padding_size;
      int col_idx = warpId%4*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+warpId*32+laneId) = *(x_lb + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+warpId*32+laneId) = 0.0f;
      }

      row_idx = (warpId+8)/4;
      col_idx = (warpId+8)%4;
      row_idx = row_idx*Stride-padding_size;
      col_idx = col_idx*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+(8+warpId)*32+laneId) = *(x_lb + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+(8+warpId)*32+laneId) = 0.0f;
      }

      // Read ub
      row_idx = warpId/4;
      col_idx = warpId%4;
      row_idx = row_idx*Stride-padding_size;
      col_idx = col_idx*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+(16+warpId)*32+laneId) = *(x_ub + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+(16+warpId)*32+laneId) = 0.0f;
      }

      row_idx = (warpId+8)/4;
      col_idx = (warpId+8)%4;
      row_idx = row_idx*Stride-padding_size;
      col_idx = col_idx*Stride-padding_size;
      if (is_valid(row_idx, col_idx, Width, Height)) {
        *(shmem+(24+warpId)*32+laneId) = *(x_ub + batch_idx*CIN*Height*Width + cin_idx*Height*Width + (block_i+row_idx+k1_idx)*Width + block_j+col_idx+k2_idx);
      } else {
        *(shmem+(24+warpId)*32+laneId) = 0.0f;
      }

      // Read W
      *(shmem+(32+warpId)*32+laneId) = *(W + (block_j+warpId)*CIN*K*K + cin_idx*K*K + k1_idx*K + k2_idx);
      *(shmem+(40+warpId)*32+laneId) = *(W + (block_j+warpId+8)*CIN*K*K + cin_idx*K*K + k1_idx*K + k2_idx);
      __syncthreads();

      // Stage II: Compute y_lb, y_ub

      float w_val, lb_val, ub_val;
      for (int i=0; i<32; i++) {
        lb_val = *(shmem + (out_row_idx*4+out_col_idx)*32+i);
        ub_val = *(shmem + (out_row_idx*4+out_col_idx)*32+i + 16*32);
        w_val = *(shmem + c_out_idx*32 + i + 32*32);
        if (w_val>0) {
          y_lb_val += w_val * lb_val;
          y_ub_val += w_val * ub_val;
        } else {
          y_lb_val += w_val * ub_val;
          y_ub_val += w_val * lb_val;
        }
      }
    }


    *(y_lb + batch_idx*COUT*Height*Width + (block_z+c_out_idx)*Height*Width + (block_i+out_row_idx)*Width + (block_j+out_col_idx)) = y_lb_val;
    *(y_ub + batch_idx*COUT*Height*Width + (block_z+c_out_idx)*Height*Width + (block_i+out_row_idx)*Width + (block_j+out_col_idx)) = y_ub_val;
    __syncthreads();
  }
}

void call_convolution_verification(  
  const float *x_lb, const float *x_ub, 
  const float *x_lw, const float *x_uw,
  const float *W,
  float *y_lb, float *y_ub, float *y_lw, float *y_uw,
  int batch_size, int Height, int Width, int CIN, int COUT, int K, int Stride, int padding_size, int dim_in){

    cudaDeviceProp deviceProp;
    checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));
    verify_conv<<<32*deviceProp.multiProcessorCount, 32*8>>>(x_lb, x_ub,
      W,
      y_lb, y_ub,
      1, Height, Width, CIN, COUT, K, Stride, padding_size
    );

    verify_conv<<<32*deviceProp.multiProcessorCount, 32*8>>>(x_lw, x_uw,
      W,
      y_lw, y_uw,
      dim_in, Height, Width, CIN, COUT, K, Stride, padding_size
    );
}

void test_conv() {
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  for (int batch_size=1; batch_size<=1; batch_size*=2) {
    for (int CIN = 16; CIN <= 128; CIN*=2) {
      int COUT = CIN;
      for (int Width = 4; Width <= 32; Width*=2) {
        int Height = Width;
        int Stride = 2;
        int padding_size = 1;
        int K = 3;
        int dim_in = 64;

        float *x_lb, *x_ub, *x_lw, *x_uw, *W, *y_lb, *y_ub, *y_lw, *y_uw;

        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * batch_size * CIN * Height * Width));
        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * batch_size * CIN * Height * Width));
        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * batch_size * CIN * Height * Width));
        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * batch_size * CIN * Height * Width));

        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * batch_size * dim_in * CIN * Height * Width));
        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * batch_size * dim_in * CIN * Height * Width));
        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * batch_size * dim_in * CIN * Height * Width));
        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * batch_size * dim_in * CIN * Height * Width));

        checkKernelErrors(
            cudaMalloc(reinterpret_cast<void **>(&W), sizeof(float) * COUT * CIN * K * K));

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
                  (verify_conv<<<32*deviceProp.multiProcessorCount, 32*8>>>(x_lb, x_ub,
                    W,
                    y_lb, y_ub,
                    1, Height, Width, CIN, COUT, K, Stride, padding_size
                  )));
                checkKernelErrors(
                  (verify_conv<<<32*deviceProp.multiProcessorCount, 32*8>>>(x_lw, x_uw,
                    W,
                    y_lw, y_uw,
                    dim_in, Height, Width, CIN, COUT, K, Stride, padding_size
                  )));
                cudaEventRecord(bmma_end);
                cudaEventSynchronize(bmma_end);
                cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
                cudaEventDestroy(bmma_start);
                cudaEventDestroy(bmma_end);
                bmma_ms_avg += bmma_ms;
        }
      
        bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
    
        printf("test_matmul_small. batch_size: %d, CIN: %d, Width: %d\n", batch_size, CIN, Width);
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

// int main(int argc, char **argv) {
//   test_conv();
//   return EXIT_SUCCESS;
// }
