#include <torch/extension.h>
#include <vector>


/* CUDA forward declarations */
torch::Tensor l1linear_cuda_forward(torch::Tensor input,
                                    torch::Tensor weight);

std::vector<torch::Tensor> l1linear_cuda_backward(torch::Tensor input,
                                                  torch::Tensor weight,
                                                  torch::Tensor d_output);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor l1linear_forward(torch::Tensor input,
                               torch::Tensor weight){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return l1linear_cuda_forward(input, weight);
}

std::vector<torch::Tensor> l1linear_backward(torch::Tensor input,
                                             torch::Tensor weight,
                                             torch::Tensor d_output){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(d_output);
    return l1linear_cuda_backward(input,
                                  weight,
                                  d_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &l1linear_forward,  "L1Linear forward (CUDA)");
  m.def("backward", &l1linear_backward, "L1Linear backward (CUDA)");
}
