/*
Command to compile on Winnie, A6000:
  nvcc -arch=sm_75 -o relu_verification relu_verification.cu
Command to run on Winnie, A6000:
  ./relu_verification
Note: Current implementation assumes:
  1) dim_in can be divided by 32.
  2) dim_in is smaller tha 16K. Otherwise a float vector of size dim_in cannot be cached in shared memory.
        This is a reasonable assumption since dim_in is usually less than 2K.
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

// using namespace nvcuda;

__device__ float dtanh(float x) {
  float v = (1+expf(-2*x))/(2*expf(-1*x));
  return 1/(v*v);
}

__device__ float diff_lower(float d, float u) {
  return (tanh(u)-tanh(d))/(u-d+0.000000000001f) - dtanh(d);
}

__device__ float diff_upper(float d, float l) {
  return (tanh(d)-tanh(l))/(d-l+0.000000000001f) - dtanh(d);
}

__global__ void tanh_verification(
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
    float l_val, u_val;

    // Read src_lw to shmem
    int base_idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in;
    for (int i=0; i<dim_in/32; i++) {
      *(shmem+i*32+threadIdx.x) = *(src_lw + base_idx + i*32 + threadIdx.x);
      *(shmem+dim_in+i*32+threadIdx.x) = *(src_uw + base_idx + i*32 + threadIdx.x);
    }
    __syncthreads();

    // Compute norm
    float square_lw = 0.0f;
    float square_uw = 0.0f;
    float val_lw, val_uw, square_val_lw, square_val_uw;
    for (int i=0; i<dim_in/32; i++) {
      val_lw = *(shmem+i*32+threadIdx.x);
      square_val_lw = val_lw*val_lw;
      val_uw = *(shmem+dim_in+i*32+threadIdx.x);
      square_val_uw = val_uw*val_uw;
      for (int offset = 16; offset > 0; offset /= 2) {
        square_val_lw += __shfl_down_sync(FULL_MASK, square_val_lw, offset);
        square_val_uw += __shfl_down_sync(FULL_MASK, square_val_uw, offset);
      }
      if (threadIdx.x == 0) {
        square_lw += square_val_lw;
        square_uw += square_val_uw;
      }
    }

    if(threadIdx.x == 0) {  
      l_val = -epsilon * sqrt(square_lw) + src_lb_val;
      u_val = epsilon * sqrt(square_uw) + src_ub_val;  
      // Since warp reduce result is only stored at laneId 0, this thread need to write the value
      //   to shmem such that all other threads in the warp can also access the l_val and u_val
      //   when computing the lw and uw.
      *(shmem+2*dim_in) = l_val;
      *(shmem+2*dim_in+1) = u_val;
    }
    __syncthreads();

    // All threads in the warp access l_val and u_val from shared memory.
    l_val = *(shmem+2*dim_in);
    u_val = *(shmem+2*dim_in+1);

    float k_l, k_u, b_l, b_u, m, d, _l, _u, v;

    // debug
    k_l = 0;
    k_u = 0;
    b_l = 0;
    b_u = 0;


    if (u_val < 0) {
      m = (l_val+u_val)/2;
      k_l = dtanh(m);
      b_l = tanh(m) - k_l*m;
      k_u = (tanh(u_val)-tanh(l_val))/(u_val-l_val+0.000000000001f);
      b_u = tanh(l_val) - k_u*l_val;
    } else if (l_val >= 0) {
      k_l = (tanh(u_val)-tanh(l_val))/(u_val-l_val+0.000000000001f);
      b_l = tanh(l_val) - k_l * l_val;
      m = (l_val+u_val)/2;
      k_u = dtanh(m);
      b_u = tanh(m) - k_u*m;
    } else {
      // Lower bound
      d = l_val/2;
      _l = l_val;
      _u = 0;
      for (int t=0; t<10; t++) { // Binary search for 10 rounds
        v = diff_lower(d, u_val);
        if (v>0) {
          _l = d;
          // _u = _u;
          d = (d+_u)/2;
        } else {
          // _l = _l;
          _u = d;
          d = (d+_l)/2;
        }
      }
      k_l = (tanh(d)-tanh(u_val))/(d-u_val+0.000000000001f);
      b_l = tanh(d) - k_l*d;

      // Upper bound
      d = u_val/2;
      _l = 0;
      _u = u_val;
      for (int t=0; t<10; t++) {
        v = diff_upper(d, l_val);
        if(v>0) {
          // _l = _l;
          _u = d;
          d = (d+_l)/2;
        } else {
          _l = d;
          // _u = _u;
          d = (d+_u)/2;
        }
      }
      k_u = (tanh(d)-tanh(l_val))/(d-l_val+0.000000000001f);
      b_u = tanh(d) - k_u*d;

      // if (block_pos == 0 && threadIdx.x == 0) {
      //   printf("cuda d[0][0][0]: %f, k_u[0][0][0]: %f, tanh(d)[0][0][0]: %f, tanh(l_val)[0][0][0]: %f, l_val[0][0][0]: %f\n", d, k_u, tanh(d), tanh(l_val), l_val);
      //   printf("cuda. tanh(d)-tanh(l_val): %f, d-l_val+0.000000000001f: %f, k_u: %f\n", tanh(d)-tanh(l_val), d-l_val+0.000000000001f, k_u);
      // }
    }

    if (threadIdx.x == 0) {
      *(out_lb + idx) = k_l*src_lb_val+b_l;
      *(out_ub + idx) = k_u*src_ub_val+b_u;
    }

    for (int i=0; i<dim_in/32; i++) {
      idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in
            + i*32+threadIdx.x;
      *(out_lw+idx) = (*(shmem+i*32+threadIdx.x)) * k_l;
      *(out_uw+idx) = (*(shmem+dim_in+i*32+threadIdx.x)) * k_u;
    }
    __syncthreads();
  }
}


void call_tanh_verification(
  const float *src_lb, const float *src_ub, const float *src_lw, const float *src_uw,
    float *out_lb, float *out_ub, float *out_lw, float *out_uw,
    int length, int dim_in, int dim_out, float epsilon
)
{
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));
  tanh_verification<<<32*deviceProp.multiProcessorCount, 32, dim_in*2*sizeof(float)+2>>>
  ( src_lb, src_ub, src_lw, src_uw,
    out_lb, out_ub, out_lw, out_uw,
    length, dim_in, dim_out, epsilon
  );
}

int main(int argc, char **argv) {

  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  float epsilon = 0.5;
  for (int length = 2; length <= 128; length*=2) {
  //  for (int dim_in=64; dim_in <= 1024; dim_in*=2) {
      int dim_in=128;
      int dim_out = dim_in;
      // int length = 4;
      // int dim_in = 32;
      // int dim_out = 16;

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

      int NUM_PROFILES = 1000;

      // Run ours NUM_PROFILES times and record time.
      float bmma_ms_avg = 0.0f;
      for(int iter=0; iter<NUM_PROFILES; ++iter){
              float bmma_ms = 0.0f;
              cudaEvent_t bmma_start;
              cudaEvent_t bmma_end;
              cudaEventCreate(&bmma_start);
              cudaEventCreate(&bmma_end);
              cudaEventRecord(bmma_start);
              checkKernelErrors(
                (tanh_verification<<<16*deviceProp.multiProcessorCount, 32, dim_in*2*sizeof(float)+2>>>(src_lb, src_ub, src_lw, src_uw,
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
    // }
  }
  return EXIT_SUCCESS;
}
