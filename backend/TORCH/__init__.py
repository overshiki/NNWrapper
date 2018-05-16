import numpy as np
import torch

from ...utils.serialization import *
from ..base import base



from .basic import Variable, Parameter, tensor, VarBase, TensorBase, device_guard

from ..operation import _OperationBase

from .operation import TensorOperation, VariableOperation

# from .pool import task_parallel

from .base import Graph, GraphList
# from .wrapper import wrapper

from .nodes import *

from .math.tensor_math import install_variable_arithmetics as iva_tensor
from .math.variable_math import install_variable_arithmetics as iva_variable

iva_tensor()
iva_variable()