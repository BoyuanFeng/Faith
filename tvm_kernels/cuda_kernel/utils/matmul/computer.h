#ifndef GEMM_COMPUTER_H
#define GEMM_COMPUTER_H

#include <mma.h>
using namespace nvcuda;


template<int NWarps, int TileK, int STAGES, int wTileM, int wTileN, int AlignK, int M, int N, int K, int ShmemOffset, int BatchOffset>
struct ShmemIteratorA{

    static constexpr int MWarpTiles = wTileM / M;
    static constexpr int NWarpTiles = wTileN / N;

    //
    //  Member variables
    //

    const float* lhs_fragment_ptr_base;

    int skew;

    __device__ __forceinline__ ShmemIteratorA(
        const float* smem)
    {
        int warpId = threadIdx.x / 32;
        int MwarpId = warpId / NWarps;
        int laneId = threadIdx.x % 32;

        int m = MwarpId * wTileM + laneId % 16;

        int row_group_id = laneId % 8;
        int col_offset = laneId / 16;
        skew = row_group_id ^ col_offset;


        lhs_fragment_ptr_base = smem + ShmemOffset + m * TileK;
    }

    __device__ __forceinline__ void load_block_tile(
        float a[][MWarpTiles][4],
        int batch_idx, int k_step){
        int reg_buffer = k_step & 1;
        int shared_idx = batch_idx % STAGES;
        const float* lhs_fragment_ptr = lhs_fragment_ptr_base + shared_idx * BatchOffset;

        int k_step2 = k_step * 2;
        int skew_t = skew ^ k_step2;
        const float* lhs_fragment_t = lhs_fragment_ptr + skew_t * AlignK;

        #pragma unroll
        for (int i = 0; i < MWarpTiles; i++){
            unsigned shared_lhs_offset_t = __nv_cvta_generic_to_shared_impl((void*)lhs_fragment_t);
            int* a_int = reinterpret_cast<int*>(&a[reg_buffer][i][0]);
            asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(a_int[0]), "=r"(a_int[1]), "=r"(a_int[2]), "=r"(a_int[3]): "r"(shared_lhs_offset_t));
            #pragma unroll
            for (int t = 0; t < 4; t++){
                a[reg_buffer][i][t] = wmma::__float_to_tf32(a[reg_buffer][i][t]);
            }
            lhs_fragment_t += M * TileK;
        }
    }
};


template<int NWarps, int TileK, int STAGES, int wTileM, int wTileN, int AlignK, int M, int N, int K, int ShmemOffset, int BatchOffset>
struct ShmemIteratorB{

    static constexpr int MWarpTiles = wTileM / M;
    static constexpr int NWarpTiles = wTileN / N;

    //
    //  Member variables
    //

    const float* rhs_fragment_ptr_base;

    int skew;

    __device__ __forceinline__ ShmemIteratorB(
        const float* smem)
    {
        int warpId = threadIdx.x / 32;
        int NwarpId = warpId % NWarps;
        int laneId = threadIdx.x % 32;

        int n = NwarpId * wTileN + laneId % 16;

        int row_group_id = laneId % 8;
        int col_offset = laneId / 16;
        skew = row_group_id ^ col_offset;


        rhs_fragment_ptr_base = smem + ShmemOffset + n * TileK;
    }

    __device__ __forceinline__ void load_block_tile(
        float b_n[][NWarpTiles][4],
        float b_p[][NWarpTiles][4],
        int batch_idx, int k_step){
        int reg_buffer = k_step & 1;
        int shared_idx = batch_idx % STAGES;
        const float* rhs_fragment_ptr = rhs_fragment_ptr_base + shared_idx * BatchOffset;

        int k_step2 = k_step * 2;
        int skew_t = skew ^ k_step2;
        const float* rhs_fragment_t = rhs_fragment_ptr + skew_t * AlignK;

#pragma unroll
        for (int j = 0; j < NWarpTiles; j++){
            int* b_int = reinterpret_cast<int*>(&b_n[reg_buffer][j][0]);
            unsigned shared_rhs_offset_t = __nv_cvta_generic_to_shared_impl((void*)rhs_fragment_t);
            asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(b_int[0]), "=r"(b_int[2]), "=r"(b_int[1]), "=r"(b_int[3]): "r"(shared_rhs_offset_t));
#pragma unroll
            for (int t = 0; t < 4; t++){
                if (b_n[reg_buffer][j][t] > 0){
                    b_p[reg_buffer][j][t] = wmma::__float_to_tf32(b_n[reg_buffer][j][t]);
                    b_n[reg_buffer][j][t] = wmma::__float_to_tf32(0.0f);
                } else {
                    b_p[reg_buffer][j][t] = wmma::__float_to_tf32(0.0f);
                    b_n[reg_buffer][j][t] = wmma::__float_to_tf32(b_n[reg_buffer][j][t]);
                }
            }
            rhs_fragment_t += N * TileK;
        }

    }
};


template<int M, int N, int K, int MWarpTiles, int NWarpTiles>
__device__ __forceinline__ void compute_block_tile(
    wmma::fragment<wmma::accumulator, M, N, K, float> c[][NWarpTiles],
    float a[][MWarpTiles][4], float b[][NWarpTiles][4], int reg_buffer){

    #pragma unroll
    for (int i = 0; i < MWarpTiles; i++){
        int* a_int = reinterpret_cast<int*>(&a[reg_buffer][i][0]);

        #pragma unroll
        for (int j = 0; j < NWarpTiles; j++){
            int* b_int = reinterpret_cast<int*>(&b[reg_buffer][j][0]);

            float *c_float = c[i][j].x;

            asm volatile ("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                            : "+f"(c_float[0]), "+f"(c_float[1]), "+f"(c_float[2]), "+f"(c_float[3])
                            : "r"(a_int[0]), "r"(a_int[1]), "r"(a_int[2]), "r"(a_int[3]), "r"(b_int[0]), "r"(b_int[1]));
                
            asm volatile ("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                            : "+f"(c_float[4]), "+f"(c_float[5]), "+f"(c_float[6]), "+f"(c_float[7])
                            : "r"(a_int[0]), "r"(a_int[1]), "r"(a_int[2]), "r"(a_int[3]), "r"(b_int[2]), "r"(b_int[3]));
        }
    }
}

#endif