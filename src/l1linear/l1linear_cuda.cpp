#include <torch/extension.h>
#include <vector>


/* CUDA forward declarations */
torch::Tensor l1linear_forward_cuda(torch::Tensor A,
                                    torch::Tensor B);

std::vector<torch::Tensor> l1linear_backward_cuda(torch::Tensor A,
                                                  torch::Tensor B,
                                                  torch::Tensor dC);


/* Implementation of Python interface */

/**
 * @brief Perform the L1 Linear "product" of two tensors.
 * 
 * @param [in] A  The LHS Tensor to "multiply".
 * @param [in] B  The RHS Tensor to "multiply".
 * @return C=AxB
 */

torch::Tensor l1linear_forward(torch::Tensor A,
                               torch::Tensor B){
    /**
     * Sanity Checks.
     */
    
    AT_ASSERTM(A.dim() == 2 && B.dim() == 2,
               "Tensors A & B must be matrices (2 dimensions)!");
    AT_ASSERTM(A.device() == B.device(),
               "Matrices A & B must reside on the same device!");
    AT_ASSERTM(A.scalar_type() == B.scalar_type(),
               "Matrices A & B must have the same dtype!");
    AT_ASSERTM(A.layout() == c10::Layout::Strided &&
               B.layout() == c10::Layout::Strided,
               "Matrices A & B must both have a strided layout!");
    AT_ASSERTM(A.size(1) == B.size(0),
               "Matrices A (%zd x %zd) shape-incompatible with B (%zd x %zd)!",
               A.size(0), A.size(1), B.size(0), B.size(1));
    
    
    /**
     * Special-case handling and optimization using already-made PyTorch ops.
     */
    
    if(A.size(0) == 0 || A.size(1) == 0 || B.size(1) == 0)
        /**
         * The inputs are empty tensors. The output is therefore either empty
         * itself, or zero.
         */
        return torch::zeros({A.size(0), B.size(1)},
                            torch::TensorOptions().dtype (A.dtype())
                                                  .layout(A.layout())
                                                  .device(A.device()));
    else if(A.size(1) == 1)
        /**
         * This is an outer "product" (M,1)x(1,P). Let PyTorch's broadcasting
         * rules handle computing the (M,P)-sized result.
         */
        return A.sub(B).abs_();
    else if(A.size(0) == 1)
        /**
         * This is a vector-matrix "product" (1,N)x(N,P). Use PyTorch's .norm()
         * to handle computing the (1,P)-sized result.
         * 
         * As a special case, this handles the inner "product" (1,N)x(N,1).
         */
        return A.transpose(0, 1).sub(B).norm(/* p= */ 1, {0}, /* keepdims= */ true);
    else if(B.size(1) == 1)
        /**
         * This is a matrix-vector "product" (M,N)x(N,1). Use PyTorch's .norm()
         * to handle computing the (M,1)-sized result.
         */
        return A.sub(B.transpose(0, 1)).norm(/* p= */ 1, {1}, /* keepdims= */ true);
    
    
    /**
     * We failed to bypass our custom op. We therefore make additional sanity
     * checks...
     */
    
    AT_ASSERTM(A.stride(0) == 1 || A.stride(1) == 1,
               "Matrix A not contiguous along either rows or columns!");
    AT_ASSERTM(B.stride(0) == 1 || B.stride(1) == 1,
               "Matrix B not contiguous along either rows or columns!");
    
    
    /**
     * ... and dispatch to the right implementation.
     */
    
    if(A.device().is_cuda()){
        return l1linear_forward_cuda(A, B);
    }else{
        AT_ASSERTM(A.device().is_cuda(), "CPU inputs not yet supported!");
    }
}

std::vector<torch::Tensor> l1linear_backward(torch::Tensor A,
                                             torch::Tensor B,
                                             torch::Tensor dC){
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
    
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(dC);
    return l1linear_backward_cuda(A, B, dC);
    
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &l1linear_forward,  "L1Linear forward (CUDA)");
  m.def("backward", &l1linear_backward, "L1Linear backward (CUDA)");
}
