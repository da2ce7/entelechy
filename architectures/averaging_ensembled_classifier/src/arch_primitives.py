# arch_primitives.py

import abc
import numpy as np
from typing import Type


class PrecisionContext(abc.ABC):
    """An abstract contract for any object that defines a scalar precision."""

    @property
    @abc.abstractmethod
    def SCALAR_NP_TYPE(self) -> Type[np.floating]:
        """The canonical NumPy floating-point type object (e.g., np.float32)."""
        pass

    @property
    @abc.abstractmethod
    def SCALAR_C_TYPE_NAME(self) -> str:
        """The string name of the type for the OpenCL compiler (e.g., 'float', 'half')."""
        pass


class Float32Context(PrecisionContext):
    """The concrete implementation of the FP32 precision contract."""

    @property
    def SCALAR_NP_TYPE(self) -> Type[np.floating]:
        return np.float32

    @property
    def SCALAR_C_TYPE_NAME(self) -> str:
        return "float"


class Float16Context(PrecisionContext):
    """The concrete implementation of the FP16 precision contract."""

    @property
    def SCALAR_NP_TYPE(self) -> Type[np.floating]:
        return np.float16

    @property
    def SCALAR_C_TYPE_NAME(self) -> str:
        return "half"