#include <cuda.h>
#include "utils.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff


__device__ __forceinline__ float square_reduce(float4 val){
    return val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
}

__device__ __forceinline__ float square_reduce(float2 val){
    return val.x * val.x + val.y * val.y;
}

__device__ __forceinline__ float square_reduce(float val){
    return val * val;
}

// Optimal hand tuned kernel
// template <typename LoadType>
// __global__ void __launch_bounds__(32) concretize(
//     const float* __restrict__ lw, const float* __restrict__ uw, 
//     const float* __restrict__ lb, const float* __restrict__ ub, 
//     float *l, float *u,
//     int length, int dim_out, int dim_in, float p, float epsilon
// ){
//   constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);


//   // Currently support only $p=2$.
//   // Warp and lane identification.
//   const unsigned int length_idx = blockIdx.x / dim_out;
//   const unsigned int dim_out_idx = blockIdx.x % dim_out;

//   int idx = length_idx*dim_out + dim_out_idx;
//   float src_lb_val = __ldg(lb + idx);
//   float src_ub_val = __ldg(ub + idx);
//   int base_idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in;

//   // Compute norm
//   float square_lw = 0.0f;
//   float square_uw = 0.0f;
//   LoadType val_lw, val_uw;

//   const LoadType* val_lw_vec = reinterpret_cast<const LoadType *>(lw + base_idx) + threadIdx.x;
//   const LoadType* val_uw_vec = reinterpret_cast<const LoadType *>(uw + base_idx) + threadIdx.x;

//   for (int i=0; i<dim_in/32/kValuesPerLoad; i++) {
//     val_lw = __ldg(val_lw_vec);
//     val_uw = __ldg(val_uw_vec);
//     square_lw += square_reduce(val_lw);
//     square_uw += square_reduce(val_uw);
//     val_lw_vec += 32;
//     val_uw_vec += 32;
//   }

//   #pragma unroll
//   for (int offset = 16; offset > 0; offset /= 2) {
//     square_lw += __shfl_down_sync(FULL_MASK, square_lw, offset);
//     square_uw += __shfl_down_sync(FULL_MASK, square_uw, offset);
//   }

//   if (threadIdx.x == 0) {
//     *(l+idx) = -epsilon * sqrt(square_lw) + src_lb_val;
//     *(u+idx) = epsilon * sqrt(square_uw) + src_ub_val;
//   }
// }

template <typename LoadType>
__device__ __forceinline__ void concretize_single(
        const float* __restrict__ w,
        const float* __restrict__ b,
        float *o,
        int length, int dim_out, int dim_in, float p, float epsilon
){
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);


    // Currently support only $p=2$.
    // Warp and lane identification.
    const unsigned int length_idx = blockIdx.x / dim_out;
    const unsigned int dim_out_idx = blockIdx.x % dim_out;

    int idx = length_idx*dim_out + dim_out_idx;
    float src_b_val = __ldg(b + idx);
    int base_idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in;

    // Compute norm
    float square_w = 0.0f;
    LoadType val_w;

    const LoadType* val_w_vec = reinterpret_cast<const LoadType *>(w + base_idx) + threadIdx.x;

    for (int i=0; i<dim_in/32/kValuesPerLoad; i++) {
    val_w = __ldg(val_w_vec);
    square_w += square_reduce(val_w);
    val_w_vec += 32;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
    square_w += __shfl_down_sync(FULL_MASK, square_w, offset);
    }

    if (threadIdx.x == 0) {
    *(o+idx) = epsilon * sqrt(square_w) + src_b_val;
    }
}

template <typename LoadType>
__global__ void __launch_bounds__(32) concretize(
    const float* __restrict__ lw, const float* __restrict__ uw, 
    const float* __restrict__ lb, const float* __restrict__ ub, 
    float *l, float *u,
    int length, int dim_out, int dim_in, float p, float epsilon
){
    if (blockIdx.y == 0){
        concretize_single<LoadType>(lw, lb, l, length, dim_out, dim_in, p, -epsilon);
    } else{
        concretize_single<LoadType>(uw, ub, u, length, dim_out, dim_in, p, epsilon);
    }
}


void verify_concretize(int length, int dim_in, int dim_out, float epsilon = 1e-6){

    int w_size = length * dim_in * dim_out;
    int b_size = length * dim_out;

    // Allocate the host tensors
    float* x_lw_h = new float[w_size];
    float* x_uw_h = new float[w_size];
    float* x_lb_h = new float[b_size];
    float* x_ub_h = new float[b_size];

    std::default_random_engine generator;

    MakeDenseMatrix(1, w_size, x_lw_h, generator);
    MakeDenseMatrix(1, w_size, x_uw_h, generator);
    MakeDenseMatrix(1, b_size, x_lb_h, generator);
    MakeDenseMatrix(1, b_size, x_ub_h, generator);

    float* x_l_h = new float[b_size];
    float* x_u_h = new float[b_size];

    // Allocate the device tensors
    float *x_lw_d, *x_uw_d, *x_lb_d, *x_ub_d, *x_l_d, *x_u_d;

    cudaMalloc(&x_lw_d, w_size * sizeof(float));
    cudaMalloc(&x_uw_d, w_size * sizeof(float));
    cudaMalloc(&x_lb_d, b_size * sizeof(float));
    cudaMalloc(&x_ub_d, b_size * sizeof(float));
    cudaMalloc(&x_l_d, b_size * sizeof(float));
    cudaMalloc(&x_u_d, b_size * sizeof(float));

    // Copy to device
    cudaMemcpy(x_lw_d, x_lw_h, w_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_uw_d, x_uw_h, w_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_lb_d, x_lb_h, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_ub_d, x_ub_h, b_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 grid_dim(length * dim_out, 2, 1);
    dim3 block_dim(32, 1, 1);
    
    // Create cuda events for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i=0; i < 100; i++){
        if (dim_in % 128 == 0){
            concretize<float4><<<grid_dim, block_dim>>>(x_lw_d, x_uw_d, x_lb_d, x_ub_d, x_l_d, x_u_d, 
                length, dim_out, dim_in, 2, epsilon);
        } else if (dim_in % 64 == 0){
            concretize<float2><<<grid_dim, block_dim>>>(x_lw_d, x_uw_d, x_lb_d, x_ub_d, x_l_d, x_u_d, 
                length, dim_out, dim_in, 2, epsilon);
        } else {
            concretize<float><<<grid_dim, block_dim>>>(x_lw_d, x_uw_d, x_lb_d, x_ub_d, x_l_d, x_u_d, 
                length, dim_out, dim_in, 2, epsilon);
        }
        
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ElapsedTime: %.4f ms\n", milliseconds/100);
    
    // Load result back
    cudaMemcpy(x_l_h, x_l_d, b_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x_u_h, x_u_d, b_size * sizeof(float), cudaMemcpyDeviceToHost);

    float* x_l_h_ref = new float[b_size];
    float* x_u_h_ref = new float[b_size];

    // Verify the result on host
    for (int i=0; i < length; i++){
        for (int j=0; j < dim_out; j++){
            float tmp_l = 0;
            float tmp_u = 0;
            for (int k=0; k < dim_in; k++){
                tmp_l += pow(x_lw_h[i * dim_out * dim_in + j * dim_in + k], 2);
                tmp_u += pow(x_uw_h[i * dim_out * dim_in + j * dim_in + k], 2);
            }
            x_l_h_ref[i * dim_out + j] = x_lb_h[i * dim_out + j] - epsilon * sqrt(tmp_l);
            x_u_h_ref[i * dim_out + j] = x_ub_h[i * dim_out + j] + epsilon * sqrt(tmp_u);
        }
    }

    int errors_l = 0;
    int errors_u = 0;
    for (int i=0; i < length * dim_out; i++){
        if(abs(x_l_h[i] - x_l_h_ref[i]) > 0.01){
            break;
            errors_l ++;
        }
        if(abs(x_u_h[i] - x_u_h_ref[i]) > 0.01){
            errors_u ++;
        }
    }

    if (errors_l > 0) {
        printf(
            "Concretize l does not agree with SEQUENTIAL! %d errors!\n",
            errors_l);
    }else {
        printf("Results verified: Concretize l agree.\n");
    }

    if (errors_u > 0) {
        printf(
            "Concretize u does not agree with SEQUENTIAL! %d errors!\n",
            errors_u);
    }else {
        printf("Results verified: Concretize u agree.\n");
    }
}

int main(int argc, char **argv){
    verify_concretize(128, 1024, 1024, 0.3);
    verify_concretize(128, 1024, 1088, 0.3);
    verify_concretize(128, 1024, 1056, 0.3);
    return 0;
}