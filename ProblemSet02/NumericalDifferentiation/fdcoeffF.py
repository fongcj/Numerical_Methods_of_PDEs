import numpy as np

def fdcoeffF(k, xbar, x):
    '''
    Compute coefficients for finite difference approximation for the
    derivative of order k at xbar based on grid values at points in x.

    This function returns a row vector c of dimension 1 by n, where n=len(x),
    containing coefficients to approximate u^{(k)}(xbar),
    the k'th derivative of u evaluated at xbar,  based on n values
    of u at x[0], x[1], ... x[n-1].

    If U is a column vector containing u(x) at these n points, then
    dot(c,U) will give the approximation to u^{(k)}(xbar).

    Note for k=0 this can be used to evaluate the interpolating polynomial
    itself.

    Requires len(x) > k.

    Usually the elements x[i] are monotonically increasing
    and x[0] <= xbar <= x[n-1], but neither condition is required.
    The x values need not be equally spaced but must be distinct.

    This program should give the same results as fdcoeffV.py, but for large
    values of n is much more stable numerically.

    Based on the program "weights" in
      B. Fornberg, "Calculation of weights in finite difference formulas",
      SIAM Review 40 (1998), pp. 685-691.

    Note: Forberg's algorithm can be used to simultaneously compute the
    coefficients for derivatives of order 0, 1, ..., m where m <= n-1.
    This gives an n by m coefficient matrix C whose k'th column gives
    the coefficients for the k'th derivative.

    In this version we set m=k and only compute the coefficients for
    derivatives of order up to order k, and then return only the k'th column
    of the resulting C matrix (converted to a row vector).
    This routine is then compatible with fdcoeffV.
    It can be easily modified to return the whole array if desired.

    From  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
    '''

    n = len(x)
    if k >= n:
        raise Exception('*** length(x) must be larger than k')

    m = k   # change to m=n-1 if you want to compute coefficients for all
            # possible derivatives.  Then modify to output all of C.
    c1 = 1
    c4 = x[0] - xbar
    C = np.zeros( (n, m+1), dtype=float)
    C[0,0] = 1
    for i in xrange(1, n):
        mn = min(i, m)
        c2 = 1
        c5 = c4
        c4 = x[i] - xbar
        for j in xrange(i):
            c3 = x[i] - x[j]
            c2 = c2*c3
            if j == i-1:
                for s in xrange(mn, 0, -1):
                    C[i,s] = c1*(s*C[i-1,s-1] - c5*C[i-1,s])/c2

                C[i,0] = -c1*c5*C[i-1,0]/c2

            for s in xrange(mn, 0, -1):
                C[j,s] = (c4*C[j,s] - s*C[j,s-1])/c3

            C[j,0] = c4*C[j,0]/c3

        c1 = c2

    return C[:,-1].T            # last column of c gives desired row vector
