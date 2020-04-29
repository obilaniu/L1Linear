#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>


namespace{
#if 0
/* Leftovers */
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}
#endif

template <typename scalar_t>
__global__ void l1linear_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output){
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
#if 0
    /* Leftovers */
    if (c < gates.size(2)){
      input_gate[n][c] = sigmoid(gates[n][0][c]);
      output_gate[n][c] = sigmoid(gates[n][1][c]);
      candidate_cell[n][c] = elu(gates[n][2][c]);
      new_cell[n][c] =
          old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
      new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
    }
#endif
}

template <typename scalar_t>
__global__ void l1linear_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_output,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_weight){
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
#if 0
    /* Leftovers */
    if (c < d_gates.size(2)){
      const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
      const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
      const auto d_new_cell =
          d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];
    
    
      d_old_cell[n][c] = d_new_cell;
      const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
      const auto d_input_gate = candidate_cell[n][c] * d_new_cell;
    
      d_gates[n][0][c] =
          d_input_gate * d_sigmoid(gate_weights[n][0][c]);
      d_gates[n][1][c] =
          d_output_gate * d_sigmoid(gate_weights[n][1][c]);
      d_gates[n][2][c] =
          d_candidate_cell * d_elu(gate_weights[n][2][c]);
    }
#endif
}
} // namespace

torch::Tensor l1linear_cuda_forward(torch::Tensor input,
                                    torch::Tensor weight){
    const auto batch_size   = input.size(0);
    const auto in_features  = weight.size(0);
    const auto out_features = weight.size(1);
    
    auto options = torch::TensorOptions().dtype (input.dtype())
                                         .layout(input.layout())
                                         .device(input.device());
    auto output = torch::zeros({batch_size, out_features}, options);
    
    const dim3 threads(32, 32);
    const dim3 blocks((batch_size + 1024 - 1) / 1024, batch_size);/* FIXME: Wrong. */
    
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "l1linear_forward_cuda", ([&] {
      l1linear_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
          input .packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));
    
    return output;
}

std::vector<torch::Tensor> l1linear_cuda_backward(torch::Tensor input,
                                                  torch::Tensor weight,
                                                  torch::Tensor d_output){
    const auto batch_size   = input.size(0);
    const auto in_features  = weight.size(0);
    const auto out_features = weight.size(1);
    
    auto d_input  = torch::zeros_like(input);
    auto d_weight = torch::zeros_like(weight);
    
    const dim3 threads(32, 32);
    const dim3 blocks((batch_size + 1024 - 1) / 1024, batch_size);/* FIXME: Wrong. */
    
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
