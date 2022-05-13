#ifndef GEMM_SRC_ITERATOR_H
#define GEMM_SRC_ITERATOR_H



// The tile size of a warp is TileM * TileN * TileK
// Element: the scalar type of the element {float, half, ...}
// AlignK: The length of the vector type. E.g. when loading float with float4, AlignK = 4
// NumWarp: number of warps per thread block

template <int TileM, int TileN, int TileK, int AlignK, int NumWarp, int STAGES, int ShmemOffset, int BatchOffset, int SpatialDim>
struct Iterator
{
    //
    //  Static Members
    //

    // Each subwarp will load a row with length TileK from the global memory
    static constexpr int SubWarpSize = TileK / AlignK;
    static_assert(SubWarpSize <= 32, "TileK / AlignK should be smaller than 32");
    static_assert(TileK % AlignK == 0, "TileK should be multiple of AlignK");
    static_assert(TileK == 32, "Currently only support TileK = 32");
    static_assert(AlignK == 4, "Currently only support float4");

    static constexpr int NumSubWarp = NumWarp * 32 / SubWarpSize;
    static_assert(NumSubWarp <= TileM, "Number of subwarp should be <= TileM");
    static_assert(NumSubWarp <= TileN, "Number of subwarp should be <= TileN");

    // static constexpr int BatchOffset = (TileM + TileN) * TileK;
    //
    //  Member variables
    //

    const float* matrix_ptr;
    float* fragment_ptr;
    int k;

    __device__ __forceinline__ Iterator(
        const float* matrix, float* smem, int offset, int k_)
    {
        k = k_;
        int subwarpId = threadIdx.x / SubWarpSize;
        int sublaneId = threadIdx.x % SubWarpSize;

        int row_group_id = subwarpId % 8;
        int skew = sublaneId ^ row_group_id;

        matrix_ptr = matrix + (offset + subwarpId) * k + sublaneId * AlignK;
        fragment_ptr = smem + ShmemOffset + subwarpId * TileK + skew * AlignK;
    }

    __device__ __forceinline__ void Load_async(int batch_idx){
        int shared_idx = batch_idx % STAGES;
        // Get the pointers
        const float* matrix_t = matrix_ptr + batch_idx * TileK;

        float* fragment_t = fragment_ptr + shared_idx * BatchOffset;

        #pragma unroll
        for (int step_m = 0; step_m < SpatialDim / NumSubWarp; step_m ++){
            unsigned fragment_offset_t = __nv_cvta_generic_to_shared_impl((void*)fragment_t);
            asm("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(fragment_offset_t), "l"(matrix_t), "n"(16));
            matrix_t += NumSubWarp * k;
            fragment_t += NumSubWarp * TileK;
        }
    }
};


#endif