import torch
try:
    from .    import l1linear_cuda
except ImportError:
    from .jit import l1linear_cuda
from . import l1linear
