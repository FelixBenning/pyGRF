import numpy as np

def is_scalar(x):
    """ boolean check if x is a scalar """
    return not hasattr(x, '__len__') # is this a sufficient check?

class LazyAdaptedSpan():
    """ A lazy version of the Adapted Span V_t """
    __slots__ = '_dim', '_basis'

    def __init__(self, *, dim=np.inf, init_basis=None) -> None:
        self._basis = [] if init_basis is None else init_basis
        self._dim = dim

    def as_standard_basis(self, coeffs):
        """ Convert the coefficients to the standard basis """
        self.eager(len(coeffs))
        return sum(coeff * basis for coeff, basis in zip(coeffs, self._basis)) 


    def __getitem__(self, idx):
        if idx < len(self._basis):
            return self._basis[idx]
        if idx < self._dim:
            self.eager(idx + 1)
            return self._basis[idx]

    def to_span(self, vec) -> 'LazyVec':
        """ Convert a vector to LazyVec of the span """

        # no-op if already LazyVec with this span
        if isinstance(vec, LazyVec) and vec.basis == self:
            return vec

        # unfortunately have to do work now
        coeffs = []
        w = vec
        for v in enumerate(self._basis):
            coeff = np.dot(v,w)
            coeffs.append(coeff)
            w -= coeff * v
            if w == 0:
                break

        if w != 0:
            # vec was linear independent of existing span
            coeff = np.linalg.norm(w)
            coeffs.append(coeff)
            self._basis.append(w/coeff) # normalize & append

        return LazyVec(basis_ref=self, coeffs=coeffs)


    def eager(self, n):

        pass


class LazyVec():
    """ Object which represents a lazily evaluated vector """
    __slots__ = 'basis', 'coeffs'

    def __init__(self, basis_ref: LazyAdaptedSpan, coeffs) -> None:
        self.basis = basis_ref
        self.coeffs = coeffs

    def eager(self) -> np.array:
        """ Cacluate the vector eagerly """
        return self.basis.as_standard_basis(self.coeffs)

    def __add__(self, other):
        if isinstance(other, LazyVec) and self.basis == other.basis:
            return LazyVec(self.basis, self.coeffs + other.coeffs)

        return self.eager() + other

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if is_scalar(other):
            return LazyVec(self.basis, other * self.coeffs)

        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if isinstance(other, LazyVec) and self.basis == other.basis:
            return LazyVec(self.basis, self.coeffs - other.coeffs)

        return self.eager() - other

    def __rsub__(self, other):
        if isinstance(other, LazyVec) and self.basis == other.basis:
            return LazyVec(self.basis, self.coeffs - other.coeffs)

        return other - self.eager()