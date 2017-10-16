import numpy as np
import math
from operator import mul

def factorial(n):
    '''Compute n!'''
    return reduce(mul, xrange(1,n+1), 1)

def fdcoeffV(k,xbar,x):
    '''
    Compute coefficients for finite difference approximation for the derivative
    of order k at xbar based on grid values at points in x.  x may be a list or
    a NumPy array; if an array, it is assumed to have shape (n,) or (n,1).

    WARNING: This approach is numerically unstable for large values of n since
    the Vandermonde matrix is poorly conditioned.  Use fdcoeffF.py instead,
    which is based on Fornberg's method.

    This function returns a row vector c of dimension 1 by n, containing
    coefficients to approximate u^{(k)}(xbar), the k'th derivative of u
    evaluated at xbar,  based on n values of u at x(0), x(1), ... x(n-1).

    If U is a column vector containing u(x) at these n points, then
    c*U will give the approximation to u^{(k)}(xbar).

    Note for k=0 this can be used to evaluate the interpolating polynomial
    itself.

    Requires len(x) > k.
    Usually the elements x[i] are monotonically increasing
    and x[0] <= xbar <= x[n-1], but neither condition is required.
    The x values need not be equally spaced but must be distinct.

    From  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
    '''

    n = len(x)
    if k >= n:
        raise Exception('len(x) must be larger than k')

    x = np.array(x, dtype=float) # make sure x is a float array

    A = np.ones((n,n))

    xrow = x - xbar              # displacements x-xbar
    xrow.shape = (1,n)           # make sure xrow is a row vector

    for i in xrange(1,n):
        A[i,:] = (xrow ** i) / factorial(i)

    b = np.zeros((n,1))          # b is right hand side

    b[k] = 1                     # so k'th derivative term remains

    c = np.linalg.solve(A,b)     # solve Ac = b (n by n system for coefficients)
    return c.T                   # return as a row vector
