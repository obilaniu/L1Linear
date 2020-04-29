import math
import torch
from torch import nn
from .     import l1linear_cuda


class L1LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        output = l1linear_cuda.forward(input, weight)
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_variables
        d_input, d_weight = l1linear_cuda.backward(input, weight,
                                                   d_output.contiguous())
        return d_input, d_weight

l1linear = L1LinearFunction.apply


class L1Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        if bias:
            raise NotImplementedError
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        return l1linear(input, self.weight)
