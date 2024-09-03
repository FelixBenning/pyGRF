import numpy as np

from pygrf.adapted_span import LazyAdaptedSpan


class LinIsotropicGRF():
    """ Object which represents a lazily evaluated linear Isotropic Gaussian Random Function """

    __slots__ = 'mean', 'kernel', 'dim', 'adapted_span'

    def __init__(self, * , dim = np.inf) -> None:
        self.dim = dim
        self.adapted_span = LazyAdaptedSpan()

    def __call__(self, x) -> np.array:
        pass

    def gradient(self, x):
        pass
