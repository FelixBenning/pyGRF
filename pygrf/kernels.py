from abc import abstractmethod


class Kernel:

    @abstractmethod
    def covariance(self, c1, c2, /, *, derivatives=0):
        """ Compute the covariance between c1 and c2, if derivatives>0 also compute the derivatives """
        pass

class LinearIsotropicKernel(Kernel):
    pass

class StationaryKernel(Kernel):
    pass

class IsotropicKernel(StationaryKernel, LinearIsotropicKernel):
    pass

