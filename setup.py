from os.path import join
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name                 = 'l1linear',
    version              = '0.0.0.dev0',
    description          = 'PyTorch Linear Module where the element multiplication a_i \\times b_i is replaced with absolute difference | a_i - b_i |.',
    ext_modules          = [
        CUDAExtension('l1linear.l1linear_cuda', [
            join('src', 'l1linear', 'l1linear_cuda.cpp'),
            join('src', 'l1linear', 'l1linear_cuda_kernel.cu'),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe             = False,
    python_requires      = '>=3.6.2',
    install_requires     = [
        "torch>=1.1.0",
    ],
    packages             = find_packages("src"),
    package_dir          = {'': 'src'},
)
