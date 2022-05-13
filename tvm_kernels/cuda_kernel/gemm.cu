#include <mma.h>
#include <cuda_runtime.h>
#include "utils.h"

#include "utils/gemm/src_iterator.h"
#include "utils/gemm/computer.h"
#include "utils/gemm/store_util.h"


using namespace nvcuda;

// Define Problem size
// Instruction size
#define M 16
#define N 16
#define K 8

// Block Tile Size
#define TileN 64
#define TileM 128
#define TileK 32

// Pipeline
#define Stages 4

// Warp Tile Size
#define wTileN 64
#define wTileM 64

//
//  Gemm Baseline Implementation
//



__global__ void GemmKernel(
    int m, int n, int k,
    const float* __restrict__ lhs_matrix,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix)
{
    // Get static variables
    constexpr int NWarps = TileN / wTileN;  // Number of warps along the N dimension
    constexpr int MWarps = TileM / wTileM;  // Number of warps along the M dimension
    constexpr int NWarpTiles = wTileN / N;
    constexpr int MWarpTiles = wTileM / M;
    constexpr int NumWarp = NWarps * MWarps;
    constexpr int BatchOffset = (TileM + TileN) * TileK;

    // dynamic shared memory
    extern __shared__ float smem[];

    // Warp and lane identification
    // const unsigned int warpId = threadIdx.x / 32;
    // const unsigned int laneId = threadIdx.x % 32;

    // get tile offset
    int m_offset = blockIdx.x * TileM;
    int n_offset = blockIdx.y * TileN;

    Iterator<TileM, TileN, TileK, 4, NumWarp, Stages, 0, BatchOffset, TileM> src_loader_a(lhs_matrix, smem, m_offset, k);
    Iterator<TileM, TileN, TileK, 4, NumWarp, Stages, TileM * TileK, BatchOffset, TileN> src_loader_b(rhs_matrix, smem, n_offset, k);

    ShmemIteratorA<NWarps, TileK, Stages, wTileM, wTileN, 4, M, N, K, 0, BatchOffset> shmem_iter_a(smem);
    ShmemIteratorB<NWarps, TileK, Stages, wTileM, wTileN, 4, M, N, K, TileM * TileK, BatchOffset> shmem_iter_b(smem);
    

    wmma::fragment<wmma::accumulator, M, N, K, float> c[MWarpTiles][NWarpTiles];

    float a[2][MWarpTiles][4];
    float b[2][NWarpTiles][4];    

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
        src_loader_a.Load_async(fetch_batch);
        src_loader_b.Load_async(fetch_batch);
        asm volatile ("cp.async.commit_group;");
    }
    asm volatile ("cp.async.wait_group %0;" :: "n"(Stages - 2) );
    __syncthreads();
    shmem_iter_a.load_block_tile(a, 0, 0);
    shmem_iter_b.load_block_tile(b, 0, 0);

    // Main loop
    for (int compute_batch = 0; compute_batch < k_batch; compute_batch ++){
        if (fetch_batch < k_batch){
            src_loader_a.Load_async(fetch_batch);
            src_loader_b.Load_async(fetch_batch);
            fetch_batch ++;
            asm volatile ("cp.async.commit_group;");
        }
        #pragma unroll
        for (int k_step = 0; k_step < TileK / K; k_step ++){
            if (k_step < TileK / K - 1){
                shmem_iter_a.load_block_tile(a, compute_batch, k_step + 1);
                shmem_iter_b.load_block_tile(b, compute_batch, k_step + 1);
            }
            compute_block_tile<M, N, K, MWarpTiles, NWarpTiles>(c, a, b, k_step % 2);
        }
        asm volatile ("cp.async.wait_group %0;" :: "n"(Stages - 2) );
        __syncthreads();
        shmem_iter_a.load_block_tile(a, compute_batch + 1, 0);
        shmem_iter_b.load_block_tile(b, compute_batch + 1, 0);
    }

    __syncthreads();

    // Store the D fragment to shared memory
    ResStore<MWarps, NWarps, wTileM, wTileN, TileM, TileN, M, N, K, 4, NumWarp> storer(smem, output_matrix, m_offset, n_offset, n);
    storer.store_block_res(c);

    __syncthreads();

    storer.write_to_global(n);

    // Now that shared memory contains all the D tiles, stream them to global memory
    // float4* global_res = reinterpret_cast<float4 *>(output_matrix + (m_offset + warpId) * n + n_offset) + laneId;
    // float4* shared_res = reinterpret_cast<float4 *>(smem + warpId * (TileN + 8)) + laneId;
    // #pragma unroll
    // for (int i = 0; i < TileM / NumWarp; i++){
    //     float4* global_res_t = global_res;
    //     float4* shared_res_t = shared_res;
    //     #pragma unroll
    //     for (int j = 0; j < TileN / 128; j++){
    //         *(global_res_t) = *(shared_res_t);
    //         global_res_t += 32;
    //         shared_res_t += 32;
    //     }
    //     global_res += NumWarp * n / 4;
    //     shared_res += NumWarp * (TileN + 8) / 4;
    // }
    
}


void verify_matmul(int m, int n, int k){

    int lhs_size = m * k;
    int rhs_size = n * k;
    int out_size = m * n;
    // Allocate the host tensors
    float* lhs_h = new float[lhs_size];
    float* rhs_h = new float[rhs_size];

    std::default_random_engine generator;

    MakeDenseMatrix(1, lhs_size, lhs_h, generator);
    MakeDenseMatrix(1, rhs_size, rhs_h, generator);

    float* out_h = new float[out_size];

    // Allocate the device tensors
    float *lhs_d, *rhs_d, *out_d;

    cudaMalloc(&lhs_d, lhs_size * sizeof(float));
    cudaMalloc(&rhs_d, rhs_size * sizeof(float));
    cudaMalloc(&out_d, out_size * sizeof(float));

    // Copy to device
    cudaMemcpy(lhs_d, lhs_h, lhs_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_d, rhs_h, rhs_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int maxbytes = max(TileM * (TileN + 8) * sizeof(float), (TileM + TileN) * TileK * Stages * sizeof(float));
    cudaFuncSetAttribute(GemmKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    dim3 grid_dim(m / TileM, n / TileN, 1);
    dim3 block_dim(TileM * TileN / wTileM / wTileN * 32, 1, 1);

    // Create cuda events for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // GemmKernel<<<grid_dim, block_dim, maxbytes>>>(m, n, k, lhs_d, rhs_d, out_d);
    for (int i=0; i < 10; i++){
        GemmKernel<<<grid_dim, block_dim, maxbytes>>>(m, n, k, lhs_d, rhs_d, out_d);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ElapsedTime: %.4f ms\n", milliseconds/10);

    // Load result back
    cudaMemcpy(out_h, out_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    float* out_h_ref = new float[out_size];

    // Verify the result on host
    for (int i=0; i < m; i++){
        for (int j=0; j < n; j++){
            float tmp = 0;
            for (int t=0; t < k; t++){
                tmp += lhs_h[i * k + t] * rhs_h[j * k + t];
            }
            out_h_ref[i * n + j] = tmp;
        }
    }

    int errors = 0;
    for (int i=0; i < m * n; i++){
        if(abs(out_h[i] - out_h_ref[i]) > 0.02){
            errors ++;
        }
    }

    if (errors > 0) {
        printf(
            "GemmKernel does not agree with SEQUENTIAL! %d errors!\n",
            errors);
    }else {
        printf("Results verified: They agree.\n");
    }
}

int main(int argc, char **argv){
    verify_matmul(1024, 1024, 1024);
    return 0;
}