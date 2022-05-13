#ifndef GEMM_STORE_UTIL_H
#define GEMM_STORE_UTIL_H

#include <mma.h>
using namespace nvcuda;

template<int MWarps, int NWarps, int wTileM, int wTileN, int TileM, int TileN, int M, int N, int K, int AlignN, int NumWarp>
struct ResStore{

    static constexpr int SHM_STRIDE = TileN + 8;
    static constexpr int MWarpTiles = wTileM / M;
    static constexpr int NWarpTiles = wTileN / N;

    // Each subwarp will store a row with length min(128, TileN)
    static constexpr int SubWarpSize = TileN / AlignN;
    static_assert(SubWarpSize <= 32, "TileN / AlignK should be smaller than 32");
    static_assert(TileN % AlignN == 0, "TileN should be multiple of AlignN");
    static_assert(AlignN == 4, "Currently only support AlignN=4");
    
    static constexpr int NumSubWarp = NumWarp * 32 / SubWarpSize;
    static_assert(NumSubWarp <= TileM, "Number of subwarp should be <= TileM");
    //
    //  Member variables
    //

    float* shared_res_ptr;
    float4* global_res_ptr;
    float4* shared_res_ptr_out;

    __device__ __forceinline__ ResStore(
        float* smem, float* gmem, int m_offset, int n_offset, int n)
    {
        int warpId = threadIdx.x / 32;
        int MwarpId = warpId / NWarps;
        int NwarpId = warpId % NWarps;

        shared_res_ptr = smem + MwarpId * wTileM * SHM_STRIDE + NwarpId * wTileN;

        int subwarpId = threadIdx.x / SubWarpSize;
        int sublaneId = threadIdx.x % SubWarpSize;

        global_res_ptr = reinterpret_cast<float4 *>(gmem + (m_offset + subwarpId) * n + n_offset) + sublaneId;
        shared_res_ptr_out = reinterpret_cast<float4 *>(smem + subwarpId * (TileN + 8)) + sublaneId;
    }

    __device__ __forceinline__ void store_block_res(wmma::fragment<wmma::accumulator, M, N, K, float> c[][NWarpTiles]){
        #pragma unroll
        for (int i = 0; i < MWarpTiles; i++){
            float* shared_res_ptr_t = shared_res_ptr;
            #pragma unroll
            for (int j = 0; j < NWarpTiles; j++){
                wmma::store_matrix_sync(shared_res_ptr_t, c[i][j], SHM_STRIDE, wmma::mem_row_major);
                shared_res_ptr_t += N;
            }
            shared_res_ptr += M * SHM_STRIDE;
        }
    }

    __device__ __forceinline__ void write_to_global(int n){
        #pragma unroll
        for (int i=0; i < TileM / NumSubWarp; i++){
            *(global_res_ptr) = *(shared_res_ptr_out);
            global_res_ptr += NumSubWarp * n / AlignN;
            shared_res_ptr_out += NumSubWarp * (TileN + 8) / AlignN;
        }
    }
};

#endif