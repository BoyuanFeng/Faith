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
    const float *l, const float *u,
    const float *src_lb, const float *src_ub, const float *src_lw, const float *src_uw,
    float *out_lb, float *out_ub, float *out_lw, float *out_uw,
    int length, int dim_in, int dim_out
){
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int length_idx = block_pos / dim_out;
    const unsigned int dim_out_idx = block_pos % dim_out;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (length_idx >= length) {
      break;
    }

    int idx = length_idx*dim_out + dim_out_idx;
    float l_val = *(src_lb + idx);
    float u_val = *(src_ub + idx);
    float src_lb_val = *(src_lb + idx);
    float src_ub_val = *(src_ub + idx);

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
      *(out_lw+idx) = *(src_lw+idx) * lk;
      *(out_uw+idx) = *(src_uw+idx) * uk;
    }
  }
}

int main(int argc, char **argv) {

  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  // for (int length = 2; length <= 128; length*=2) {
  //  for (int dim_in=64; dim_in <= 1024; dim_in*=2) {
      int length = 2;
      int dim_in = 64;
      int dim_out = dim_in;

      float *l, *u, *src_lb, *src_ub, *src_lw, *src_uw, *out_lb, *out_ub, *out_lw, *out_uw;

      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&l), sizeof(float) * length * dim_out));
      checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&u), sizeof(float) * length * dim_out));
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
                (relu_verification<<<deviceProp.multiProcessorCount, 32>>>(l, u,
                  src_lb, src_ub, src_lw, src_uw,
                  out_lb, out_ub, out_lw, out_uw,
                  length, dim_in, dim_out
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
      // printf("TOPS: %.2f\n", (((double)(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)/(bmma_ms_avg/1000.)) / 1e12);
    
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(l)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(u)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_lb)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_ub)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_lw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_uw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_lb)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_ub)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_lw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_uw)));
    // }
  // }
  return EXIT_SUCCESS;
}