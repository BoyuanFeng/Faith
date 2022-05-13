/*
Command to compile on Winnie, A6000:
  nvcc -arch=sm_75 -o relu_verification relu_verification.cu
Command to run on Winnie, A6000:
  ./relu_verification
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

using namespace nvcuda;

__global__ void relu_verification(
    const float *src_lb, const float *src_ub, const float *src_lw, const float *src_uw,
    float *out_lb, float *out_ub, float *out_lw, float *out_uw,
    int length, int dim_in, int dim_out, float epsilon
){
  extern __shared__ float shmem[];

  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int length_idx = block_pos / dim_out;
    const unsigned int dim_out_idx = block_pos % dim_out;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (length_idx >= length) {
      break;
    }

    int idx = length_idx*dim_out + dim_out_idx;
    float src_lb_val = *(src_lb + idx);
    float src_ub_val = *(src_ub + idx);

    // Read src_lw to shmem
    int base_idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in;
    for (int i=0; i<dim_in/32; i++) {
      *(shmem+i*32+threadIdx.x) = *(src_lw + base_idx + i*32 + threadIdx.x);
      *(shmem+dim_in+i*32+threadIdx.x) = *(src_uw + base_idx + i*32 + threadIdx.x);
    }

    // Compute norm
    float mean_lw = 0.0f;
    float mean_uw = 0.0f;
    float square_lw = 0.0f;
    float square_uw = 0.0f;
    float val_lw, val_uw, square_val_lw, square_val_uw;
    for (int i=0; i<dim_in/32; i++) {
      val_lw = *(shmem+i*32+threadIdx.x);
      square_val_lw = val_lw*val_lw;
      val_uw = *(shmem+dim_in+i*32+threadIdx.x);
      square_val_uw = val_uw;
      for (int offset = 16; offset > 0; offset /= 2) {
        val_lw += __shfl_down_sync(FULL_MASK, val_lw, offset);
        square_val_lw += __shfl_down_sync(FULL_MASK, square_val_lw, offset);
        val_uw += __shfl_down_sync(FULL_MASK, val_uw, offset);
        square_val_uw += __shfl_down_sync(FULL_MASK, square_val_uw, offset);
      }
      if (threadIdx.x == 0) {
        mean_lw += val_lw;
        mean_uw += val_uw;
        square_lw += square_val_lw;
        square_uw += square_val_uw;
      }
    }
    float l_val = -epsilon * sqrt(square_lw - dim_in * mean_lw) + src_lb_val;
    float u_val = epsilon * sqrt(square_uw - dim_in * mean_uw) + src_ub_val;

    float lk = 0;
    float uk = 0;
    float l_x0 = 0.0;
    float l_y0 = 0.0;
    float u_x0 = 0.0;
    float u_y0 = 0.0;

    if (l_val >= 0) {
      // mask_pos
      lk = 1.0f; uk = 1.0f;
    } else if (l_val<0 && u_val>0) {
      // mask_both
      // l_val < 0 < u_val
      float epsilon = 1.0f/1000000000000.0f;
      uk = u_val/(u_val-l_val+epsilon);
      l_x0 = l_val;

      if (u_val > (-1*l_val)) {
        lk=1.0f;
      } // Else: lk = 0.0f
    }
    // else {
    // u_val < 0
    // lk = uk = 0.0f;
    // }

    if (threadIdx.x == 0) {
      *(out_lb + idx) = (src_lb_val - l_x0) * lk + l_y0;
      *(out_ub + idx) = (src_ub_val - u_x0) * uk + u_y0;
    }

    for (int i=0; i<dim_in/32; i++) {
      idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in
            + i*32+threadIdx.x;
      *(out_lw+idx) = *(shmem+i*32+threadIdx.x) * lk;
      *(out_uw+idx) = *(shmem+dim_in+i*32+threadIdx.x) * uk;
    }
  }
}

int main(int argc, char **argv) {

  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  float epsilon = 0.1;
  for (int length = 2; length <= 128; length*=2) {
   for (int dim_in=64; dim_in <= 1024; dim_in*=2) {
      int dim_out = dim_in;

      float *src_lb, *src_ub, *src_lw, *src_uw, *out_lb, *out_ub, *out_lw, *out_uw;

      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&src_lb), sizeof(float) * length * dim_out));
      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&src_ub), sizeof(float) * length * dim_out));
      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&out_lb), sizeof(float) * length * dim_out));
      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&out_ub), sizeof(float) * length * dim_out));

      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&src_lw), sizeof(float) * length * dim_in * dim_out));
      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&src_uw), sizeof(float) * length * dim_in * dim_out));
      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&out_lw), sizeof(float) * length * dim_in * dim_out));
      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&out_uw), sizeof(float) * length * dim_in * dim_out));

      // Run ours NUM_PROFILES times and record time.
      float bmma_ms_avg = 0.0f;
      int NUM_PROFILES = 1000;
      for(int iter=0; iter<NUM_PROFILES; ++iter){
              float bmma_ms = 0.0f;
              cudaEvent_t bmma_start;
              cudaEvent_t bmma_end;
              cudaEventCreate(&bmma_start);
              cudaEventCreate(&bmma_end);
              cudaEventRecord(bmma_start);
              checkKernelErrors(
                (relu_verification<<<deviceProp.multiProcessorCount, 32, dim_in*2*sizeof(float)>>>(src_lb, src_ub, src_lw, src_uw,
                  out_lb, out_ub, out_lw, out_uw,
                  length, dim_in, dim_out, epsilon
                )));
              cudaEventRecord(bmma_end);
              cudaEventSynchronize(bmma_end);
              cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
              cudaEventDestroy(bmma_start);
              cudaEventDestroy(bmma_end);
              bmma_ms_avg += bmma_ms;
      }
    
      bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
  
      printf("length: %d, dim_in: %d, dim_out: %d\n", length, dim_in, dim_out);
      printf("Time: %f ms\n", bmma_ms_avg);  
    
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_lb)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_ub)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_lw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_uw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_lb)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_ub)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_lw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_uw)));
    }
  }
  return EXIT_SUCCESS;
}