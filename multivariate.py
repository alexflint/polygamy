import scipy.optimize

import polynomial


def polish_root(polynomials, root, **kwargs):
    import scipy.optimize
    fun = polynomial.polynomial_vector(polynomials)
    jac = polynomial.polynomial_jacobian(polynomials)
    return scipy.optimize.root(fun=lambda x: fun(*x),
                               jac=lambda x: jac(*x),
                               x0=root,
                               **kwargs)
