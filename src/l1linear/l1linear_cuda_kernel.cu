#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>


/* Defines */

/**
 * On pre-Turing hardware, 2048 threads (in 2 blocks of 1024) could always be
 * launched given low-enough register and SMEM usage.
 * 
 * On Turing hardware, only a maximum of 1024 threads can be launched.
 */

#if __CUDA_ARCH__ >= 750
# define LAUNCH_BOUNDS __launch_bounds__(1024, 1)
#else
# define LAUNCH_BOUNDS __launch_bounds__(1024, 2)
#endif

/**
 * Ceiling Division (Round Up To Next Multiple)
 */

#define CEIL_DIV(X, D)  (((X)-1+(D)) / (D))

/**
 * Maximum dimensions of CUDA Textures.
 * 
 * The actual values are 65000x65000 [1], but we reduce to the next lower
 * multiple of 1K (1K=1024) because 63K is a nice multiple of 128, our tile
 * size.
 */

#define SIXTY_THREE_K       (63*1024L)
#define CUDA_TEX_MAX_HEIGHT SIXTY_THREE_K
#define CUDA_TEX_MAX_WIDTH  SIXTY_THREE_K



namespace{

/**
 * @brief Fused Absolute Difference and Accumulate
 * 
 * Computes |a-b|+c.
 * 
 * In reality, no optimization of this truly exists.
 * 
 * @param [in]  a
 * @param [in]  b
 * @param [in]  c
 * @return |a-b|+c
 */

template <typename scalar_t>
__device__ __forceinline__ scalar_t fabsdiffa(scalar_t a, scalar_t b, scalar_t c){
    return abs(a-b)+c;
}

/**
 * @brief L1 Linear Layer Forward Pass
 * 
 * Matrix "Multiply" with absdiff rather than multiplication:
 * 
 *     C_{mp} = alpha * sum_{n=0}^{n-1}  |A_{mn}-B_{np}| + beta*C_{mp}
 * 
 * @param [in]     alpha   The scalar by which to multiply the "product" AxB.
 *                         As a special case, alpha=0 eliminates AxB.
 * @param [in]     Atex    A MxN left-hand-side  matrix A, as a texture.
 * @param [in]     Btex    A NxP right-hand-side matrix B, as a texture.
 * @param [in]     N       The width of A and height of B. Required as an
 *                         explicit argument because it cannot be inferred from
 *                         texture objects.
 * @param [in]     beta    The scalar by which to multiply the accumulator C.
 *                         As a special case, beta=0 unconditionally overwrites.
 * @param [in,out] Cten    A MxP accumulator matrix C=alpha*AxB+beta*C.
 * @note We handle all four transposition cases efficiently: NN,NT,TN,TT.
 */

template <typename scalar_t, typename texture_t, bool AT, bool BT> LAUNCH_BOUNDS
__global__ void l1linear_cuda_forward_kernel(const scalar_t            alpha,
                                             const cudaTextureObject_t Atex,
                                             const cudaTextureObject_t Btex,
                                             const int                 N,
                                             const scalar_t            beta,
                                                   torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> Cten){
    /* SHARED MEMORY: 4096 scalar_t elements. 16KB for float, 32KB for double. */
    __shared__ scalar_t Ashared[2][8][128];
    __shared__ scalar_t Bshared[2][8][128];
    
    
    /* REGISTERS */
    scalar_t Aglobal, Bglobal;       /*   2 */
    scalar_t A[8],    B[8];          /* 8+8 */
    scalar_t C[16]={0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0};/*  16 */
    float    i,j,k;                  /*   3 */
    int      TOGGLE = 0, x, y, J;    /*   1 */
    
    
    /* MATRIX PRODUCT */
    if(alpha){
        
    }
    
    
    /* WRITEOUT */
    for(J=0;J<16;J++){
        y = blockIdx.y*128 + threadIdx.y + J*8;
        x = blockIdx.x*128 + threadIdx.x;
        
        if(y < Cten.size(0) && x < Cten.size(1)){
            scalar_t& Cglobal = Cten[y][x];
            scalar_t  AB      = C[J];
            
            if     (!alpha && !beta) Cglobal = 0;
            else if(!alpha &&  beta) Cglobal =            beta*Cglobal;
            else if( alpha && !beta) Cglobal = alpha*AB;
            else                     Cglobal = alpha*AB + beta*Cglobal;
        }
    }
}

/**
 * @brief L1 Linear Layer Backward Pass
 */

template <typename scalar_t> LAUNCH_BOUNDS
__global__ void l1linear_cuda_backward_kernel(const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
                                              const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
                                              const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_output,
                                                    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_input,
                                                    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_weight){
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
}
} // namespace


/**
 * @brief L1 Linear Layer Forward Pass
 * 
 * Matrix "Multiply" with absdiff rather than multiplication:
 * 
 *     output_{ik} = sum_{j=0}^{j-1}  |a_{ij}-b_{jk}|
 * 
 * @param [in]  input   A IxJ matrix A.
 * @param [in]  weight  A JxK matrix B.
 * @return A IxK matrix C=AxB.
 */

torch::Tensor l1linear_forward_cuda(torch::Tensor A,
                                    torch::Tensor B){
    /**
     * TENSOR PREPARATION
     * 
     * We allocate the destination tensor C, then compute the critical
     * transposition configuration of the operation.
     */
    
    auto C = torch::zeros({A.size(0), B.size(1)},
                          torch::TensorOptions().dtype (A.dtype())
                                                .layout(A.layout())
                                                .device(A.device()));
    const auto AT = A.stride(1) != 1;
    const auto BT = B.stride(1) != 1;
    
    
    /**
     * TYPE DISPATCH
     * 
     * We dispatch to the kernels by scalar type. Currently, only two types are
     * supported:
     * 
     *   - torch.float32
     *   - torch.float64
     */
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "l1linear_forward_cuda", ([&] {
        /**
         * KERNEL SELECT
         * 
         * CUDA does not support double textures, and explicitly recommends an
         * "int2"-based workaround (because 2xsizeof(int) == sizeof(double)).
         * We materialize a texture_t type and for double only, set it to
         * float2 before using it in templates.
         */
        
        cudaStream_t stream  = c10::cuda::getCurrentCUDAStream().stream();
        using texture_t = std::conditional<std::is_same<scalar_t, double>::value,
                                           float2, scalar_t>;
        auto KERNEL = [&]{
            if     (!AT && !BT) return l1linear_cuda_forward_kernel<scalar_t, texture_t, false, false>;
            else if(!AT &&  BT) return l1linear_cuda_forward_kernel<scalar_t, texture_t, false, true >;
            else if( AT && !BT) return l1linear_cuda_forward_kernel<scalar_t, texture_t, true,  false>;
            else                return l1linear_cuda_forward_kernel<scalar_t, texture_t, true,  true >;
        }();
        cudaFuncSetSharedMemConfig(KERNEL, A.element_size()==8 ? cudaSharedMemBankSizeEightByte :
                                           A.element_size()==4 ? cudaSharedMemBankSizeFourByte  :
                                                                 cudaSharedMemBankSizeDefault);
        
        
        /**
         * TEXTURE CONFIG
         * 
         * Our core kernel is based on a rather obscure trick: Matrices A and B
         * will be accessed via textures. The advantage of this strategy is
         * that all of the following are handled for us by the L1 Texture Cache:
         * 
         *   - Matrix A transposition   (L1Tex handles 2D spatial locality)
         *   - Matrix B transposition   (L1Tex handles 2D spatial locality)
         *   - Border check             (L1Tex replaces out-of-bounds accesses
         *                               with a border "colour", in this case 0,
         *                               which is the additive identity)
         * 
         * Moreover, L1Tex is separate from L1, so we can choose to prefer an
         * L1/shared memory split that heavily favours shared memory, because
         * only reads and writes to Matrix C will pass via L1.
         * 
         * The disadvantages are:
         * 
         *   - L1Tex is smaller than L1.
         *   - Textures are limited to approximately 63K by 63K pixels.
         *     Therefore, the kernel may have to be executed multiple times,
         *     performing a different part of the operation each time.
         *   - Indexing is done by floats.
         *   - No support for double, so an ugly float2 hack is required.
         *   - No support for stride(1) != 1, so extra effort required to
         *     prepare the texture.
         * 
         * On the whole, the convenience and reduced complexity of outsourcing
         * transpositions to the L1Tex make the tradeoff still worth it, so we
         * configure textures for our needs. This is a rather involved process.
         */
        
        struct cudaTextureDesc  Atd, Btd;
        memset(&Atd, 0, sizeof(Atd));
        Atd.addressMode[0] = cudaAddressModeBorder;
        Atd.addressMode[1] = cudaAddressModeBorder;
        Atd.filterMode     = cudaFilterModePoint;
        Atd.readMode       = cudaReadModeElementType;
        Atd.borderColor[0] = 0;
        Atd.borderColor[1] = 0;
        Atd.borderColor[2] = 0;
        Atd.borderColor[3] = 0;
        Btd = Atd;
        
        struct cudaResourceDesc Ard, Brd;
        memset(&Ard, 0, sizeof(Ard));
        Ard.resType                  = cudaResourceTypePitch2D;
        Ard.res.pitch2D.desc.f       = cudaChannelFormatKindFloat;
        if(std::is_same<scalar_t, double>::value){
            Ard.res.pitch2D.desc.x = Ard.res.pitch2D.desc.y = 32;/* float2 hack */
        }else{
            Ard.res.pitch2D.desc.x = 8*A.element_size();
        }
        Brd = Ard;
        Ard.res.pitch2D.pitchInBytes = A.element_size()*A.stride(AT?1:0);
        Brd.res.pitch2D.pitchInBytes = B.element_size()*B.stride(BT?1:0);
        
        
        /**
         * SUPERBLOCK LOOP
         * 
         * Because we choose the kernel using textures, we are limited to
         * approximately 63Kx63K blocks (the maximum texture size). Luckily,
         * GEMM-like operations can be (super)blocked. We break down the
         * computation of C into (63K,63K)x(63K,63K) superblocks, each of which
         * uses two textures: One over superblock A and one over superblock B.
         * As many CUDA grids of depth up to 63K are then launched as required
         * to complete the superblock C.
         */
        
        scalar_t ALPHA=1.0, BETA=0.0;
        for(int64_t MM=0; MM<C.size(0); MM+=SIXTY_THREE_K){
            for(int64_t PP=0; PP<C.size(1); PP+=SIXTY_THREE_K){
                auto height = std::min(C.size(0)-MM, SIXTY_THREE_K);
                auto width  = std::min(C.size(1)-PP, SIXTY_THREE_K);
                auto Csuper = C.narrow(0, MM, height).narrow(1, PP, width);
                
                const dim3 threads(128, 8);
                const dim3 blocks(CEIL_DIV(width, 128), CEIL_DIV(height, 128));
                
                for(int64_t NN=0; NN<A.size(1); NN+=SIXTY_THREE_K){
                    
                    /**
                     * GRID LAUNCH
                     * 
                     * One C superblock can be updated by many grid launches. Each
                     * grid partitions the work of updating the C superblock into
                     * an even number of 128x128 tile, which are performed by a
                     * 128x8=1024-CUDA-thread block. There will be up to 504x504
                     * such tiles (and therefore CUDA thread blocks) per grid.
                     * 
                     * The grid receives a textures mapped over superblocks A and B
                     * and updates the superblock C. We now prepare these
                     * textures and launch the kernel.
                     */
                    
                    const auto depth = std::min(A.size(1)-NN, SIXTY_THREE_K);
                    auto Asuper = A.narrow(0, MM, height)
                                   .narrow(1, NN, depth);
                    auto Bsuper = B.narrow(0, NN, depth)
                                   .narrow(1, PP, width);
                    
                    Ard.res.pitch2D.devPtr = A.data_ptr();
                    Ard.res.pitch2D.height = height;
                    Ard.res.pitch2D.width  = depth;
                    Brd.res.pitch2D.devPtr = B.data_ptr();
                    Brd.res.pitch2D.height = depth;
                    Brd.res.pitch2D.width  = width;
                    
                    cudaTextureObject_t Atx=0, Btx=0;
                    cudaCreateTextureObject(&Atx, &Ard, &Atd, NULL);
                    cudaCreateTextureObject(&Btx, &Brd, &Btd, NULL);
                    
                    /* LAUNCH KERNEL */
                    KERNEL<<<blocks, threads, 0, stream>>>(
                        (scalar_t)ALPHA, Atx, Btx, depth, (scalar_t)(NN ? 1.0 : BETA),
                        Csuper.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
                    );
                    
                    cudaDestroyTextureObject(Atx);
                    cudaDestroyTextureObject(Btx);
                }
            }
        }
    }));
    
    return C;
}

/**
 * @brief L1 Linear Layer Backward Pass
 */

std::vector<torch::Tensor> l1linear_backward_cuda(torch::Tensor input,
                                                  torch::Tensor weight,
                                                  torch::Tensor d_output){
    const auto batch_size       = input.size(0);
    const auto in_features      = weight.size(0);
    const auto out_features     = weight.size(1);
    const auto batch_size_u32   = (batch_size   + 32 - 1) / 32;
    const auto out_features_u32 = (out_features + 32 - 1) / 32;
    
    auto d_input  = torch::zeros_like(input);
    auto d_weight = torch::zeros_like(weight);
    
    const dim3 threads(32, 32);
    const dim3 blocks(batch_size_u32, std::min(out_features_u32, 32768L));
    
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "l1linear_backward_cuda", ([&] {
      l1linear_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
          input   .packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          weight  .packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_input .packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          d_weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));
    
    return {d_input, d_weight};
}
