import os.path
import torch.utils.cpp_extension

l1linear_cuda = torch.utils.cpp_extension.load('l1linear_cuda',
    [
        os.path.join(os.path.dirname(__file__), 'l1linear_cuda.cpp'),
        os.path.join(os.path.dirname(__file__), 'l1linear_cuda_kernel.cu'),
    ],
    verbose=True,
)
