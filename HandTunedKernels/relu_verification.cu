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
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      square_val_lw += __shfl_down_sync(FULL_MASK, square_val_lw, offset);
      square_val_uw += __shfl_down_sync(FULL_MASK, square_val_uw, offset);
    }
    if (threadIdx.x == 0) {
      square_lw += square_val_lw;
      square_uw += square_val_uw;
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

    float lk = 0.0f;
    float uk = 0.0f;
    float l_x0 = 0.0f;

    if (l_val >= 0) {
      // mask_pos
      lk = 1.0f; uk = 1.0f;
    } else if (l_val<0 && u_val>0) {
      // mask_both
      // l_val < 0 < u_val
      uk = u_val/(u_val-l_val+0.000000000001f);
      l_x0 = l_val;
      if (u_val > (-1*l_val)) {
        lk=1.0f;
      } // Else: lk = 0.0f
    }

    if (threadIdx.x == 0) {
      *(out_lb + idx) = src_lb_val * lk;
      *(out_ub + idx) = (src_ub_val - l_x0) * uk;
    }

    for (int i=0; i<dim_in/32; i++) {
      idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in
            + i*32+threadIdx.x;
      *(out_lw+idx) = (*(shmem+i*32+threadIdx.x)) * lk;
      *(out_uw+idx) = (*(shmem+dim_in+i*32+threadIdx.x)) * uk;
    }
    __syncthreads();
  }
}


void call_relu_verification(
  const float *src_lb, const float *src_ub, const float *src_lw, const float *src_uw,
    float *out_lb, float *out_ub, float *out_lw, float *out_uw,
    int length, int dim_in, int dim_out, float epsilon
)
{
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));
  relu_verification<<<deviceProp.multiProcessorCount, 32, dim_in*2*sizeof(float)+2>>>
  ( src_lb, src_ub, src_lw, src_uw,
    out_lb, out_ub, out_lw, out_uw,
    length, dim_in, dim_out, epsilon
  );
}

void init(float *lb, float *ub, float *lw, float *uw, int length, int dim_in, int dim_out) {
  printf("Initializing...\n");
  for (int i=0; i<length; i++) {
    for (int j=0; j<dim_out; j++) {
      *(lb+i*dim_out+j) = 1.0f;
      *(ub+i*dim_out+j) = 1.0f;
      for (int k=0; k<dim_in; k++) {
        *(lw+i*dim_out*dim_in+j*dim_in+k) = 1.0f;
        *(uw+i*dim_out*dim_in+j*dim_in+k) = 1.0f;
      }
    }
  }
  printf("Finished initialization\n");
}

void init_manual_test_case(float *lb, float *ub, float *lw, float *uw, int length, int dim_in, int dim_out) {
  // Left empty for debugging.
}

void print_b(float *lb, float *ub, int length, int dim_out) {
  printf("Printing lb & ub values: \n");
  for (int i=0; i<length; i++) {
    for (int j=0; j<dim_out; j++) {
      float lb_val = *(lb+i*dim_out+j);
      float ub_val = *(ub+i*dim_out+j);
      printf("i: %d, j: %d, lb: %f, ub: %f\n", i, j, lb_val, ub_val);
    }
  }
  printf("\n\n");
}

void print_w(float *lw, float *uw, int length, int dim_out, int dim_in) {
  printf("Printing lw & uw values: \n");
  for (int i=0; i<length; i++) {
    for (int j=0; j<dim_out; j++) {
      for (int k=0; k<dim_in; k++) {
        float lw_val = *(lw+i*dim_out*dim_in+j*dim_in+k);
        float uw_val = *(uw+i*dim_out*dim_in+j*dim_in+k);
        printf("i: %d, j: %d, k: %d, lw: %f, uw: %f\n", i, j, k, lw_val, uw_val);
      }
    }
  }
  printf("\n\n");
}

// To use verify_output, please first initialize values with init_manual_test_case(). 
//    Make sure that sizes are specified correctly, e.g., length, dim_in, and dim_out.
//    Note that there are certain requirements on sizes. See requirement on the top.
// #define verify_output

// int main(int argc, char **argv) {

//   cudaDeviceProp deviceProp;
//   checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

//   float epsilon = 0.5;
//   for (int length = 2; length <= 128; length*=2) {
//    for (int dim_in=64; dim_in <= 1024; dim_in*=2) {
//       int dim_out = dim_in;
//       // int length = 4;
//       // int dim_in = 32;
//       // int dim_out = 16;

//       float *src_lb, *src_ub, *src_lw, *src_uw, *out_lb, *out_ub, *out_lw, *out_uw;

//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&src_lb), sizeof(float) * length * dim_out));
//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&src_ub), sizeof(float) * length * dim_out));
//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&out_lb), sizeof(float) * length * dim_out));
//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&out_ub), sizeof(float) * length * dim_out));

//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&src_lw), sizeof(float) * length * dim_in * dim_out));
//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&src_uw), sizeof(float) * length * dim_in * dim_out));
//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&out_lw), sizeof(float) * length * dim_in * dim_out));
//       checkKernelErrors(
//           cudaMalloc(reinterpret_cast<void **>(&out_uw), sizeof(float) * length * dim_in * dim_out));

//       int NUM_PROFILES = 1000;

// #ifdef verify_output
//       float *src_lb_h = NULL;
//       float *src_ub_h = NULL;
//       float *src_lw_h = NULL;
//       float *src_uw_h = NULL;

//       src_lb_h = (float *)malloc(sizeof(float) * length * dim_out);
//       src_ub_h = (float *)malloc(sizeof(float) * length * dim_out);
//       src_lw_h = (float *)malloc(sizeof(float) * length * dim_out * dim_in);
//       src_uw_h = (float *)malloc(sizeof(float) * length * dim_out * dim_in);

//       // init(src_lb_h, src_ub_h, src_lw_h, src_uw_h, length, dim_in, dim_out);
//       init_manual_test_case(src_lb_h, src_ub_h, src_lw_h, src_uw_h, length, dim_in, dim_out);
//       // print_b(src_lb_h, src_ub_h, length, dim_out);
//       // print_w(src_lw_h, src_uw_h, length, dim_out, dim_in);

//       checkKernelErrors(cudaMemcpy(src_lb, src_lb_h, sizeof(float) * length * dim_out, cudaMemcpyHostToDevice));
//       checkKernelErrors(cudaMemcpy(src_ub, src_ub_h, sizeof(float) * length * dim_out, cudaMemcpyHostToDevice));
//       checkKernelErrors(cudaMemcpy(src_lw, src_lw_h, sizeof(float) * length * dim_out * dim_in, cudaMemcpyHostToDevice));
//       checkKernelErrors(cudaMemcpy(src_uw, src_uw_h, sizeof(float) * length * dim_out * dim_in, cudaMemcpyHostToDevice));

//       NUM_PROFILES = 1;
// #endif

//       // Run ours NUM_PROFILES times and record time.
//       float bmma_ms_avg = 0.0f;
//       for(int iter=0; iter<NUM_PROFILES; ++iter){
//               float bmma_ms = 0.0f;
//               cudaEvent_t bmma_start;
//               cudaEvent_t bmma_end;
//               cudaEventCreate(&bmma_start);
//               cudaEventCreate(&bmma_end);
//               cudaEventRecord(bmma_start);
//               checkKernelErrors(
//                 (relu_verification<<<16*deviceProp.multiProcessorCount, 32, dim_in*2*sizeof(float)+2>>>(src_lb, src_ub, src_lw, src_uw,
//                   out_lb, out_ub, out_lw, out_uw,
//                   length, dim_in, dim_out, epsilon
//                 )));
//               cudaEventRecord(bmma_end);
//               cudaEventSynchronize(bmma_end);
//               cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
//               cudaEventDestroy(bmma_start);
//               cudaEventDestroy(bmma_end);
//               bmma_ms_avg += bmma_ms;
//       }
    
//       bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
  
//       printf("length: %d, dim_in: %d, dim_out: %d\n", length, dim_in, dim_out);
//       printf("Time: %f ms\n", bmma_ms_avg);  

// #ifdef verify_output
//       float *out_lb_h = NULL;
//       float *out_ub_h = NULL;
//       float *out_lw_h = NULL;
//       float *out_uw_h = NULL;

//       out_lb_h = (float *)malloc(sizeof(float) * length * dim_out);
//       out_ub_h = (float *)malloc(sizeof(float) * length * dim_out);
//       out_lw_h = (float *)malloc(sizeof(float) * length * dim_out * dim_in);
//       out_uw_h = (float *)malloc(sizeof(float) * length * dim_out * dim_in);

//       checkKernelErrors(cudaMemcpy(out_lb_h, out_lb, sizeof(float) * length * dim_out, cudaMemcpyDeviceToHost));
//       checkKernelErrors(cudaMemcpy(out_ub_h, out_ub, sizeof(float) * length * dim_out, cudaMemcpyDeviceToHost));
//       checkKernelErrors(cudaMemcpy(out_lw_h, out_lw, sizeof(float) * length * dim_out * dim_in, cudaMemcpyDeviceToHost));
//       checkKernelErrors(cudaMemcpy(out_uw_h, out_uw, sizeof(float) * length * dim_out * dim_in, cudaMemcpyDeviceToHost));

//       print_b(out_lb_h, out_ub_h, length, dim_out);
//       // print_w(out_lw_h, out_uw_h, length, dim_out, dim_in);
// #endif

//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_lb)));
//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_ub)));
//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_lw)));
//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(src_uw)));
//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_lb)));
//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_ub)));
//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_lw)));
//       checkKernelErrors(cudaFree(reinterpret_cast<void *>(out_uw)));
//     }
//   }
//   return EXIT_SUCCESS;
// }
