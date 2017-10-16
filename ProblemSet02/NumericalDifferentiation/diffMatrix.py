# -*- coding: utf-8 -*-
"""
diffMatrix:  example program to set up sparse differentiation matrix
Created on Tue Sep 18 01:27:13 2012

@author: mspieg
"""

import scipy.sparse as sp
import numpy as np
import pylab

from fdcoeffF import fdcoeffF

def setD(k,x):
    """ 
    example function for setting k'th order sparse differentiation matrix over 
    arbitrary mesh x with a 3 point stencil
    
    input:
        k = degree of derivative <= 2
        x = numpy array of coordinates >=3 in length
    returns:
        D sparse differention matric
    """
    
    assert(k < 3) # check to make sure k < 3
    assert(len(x) > 2)
    
    N = len(x)
    # initialize a sparse NxN matrix in "lil" (linked list) format
    D = sp.lil_matrix((N,N))
    # assign the one-sided k'th derivative at x[0]
    D[0,0:3] = fdcoeffF(k,x[0],x[0:3])
    # assign centered k'th ordered derivatives in the interior
    for i in xrange(1,N-1):
        D[i,i-1:i+2] = fdcoeffF(k,x[i],x[i-1:i+2])
    # assign one sided k'th derivative at end point x[-1]
    D[N-1,-3:] = fdcoeffF(k,x[N-1],x[-3:])
    
    # convert to csr (compressed row storage) and return
    return D.tocsr()

# quicky test program
def plotDf(x,f,title=None):
    """ Quick test routine to display derivative matrices and plot
        derivatives of an arbitrary function f(x)
        input: x: numpy array of mesh-points
               f: function pointer to function to differentiate
    """  
    # calculate first and second derivative matrices
    D1 = setD(1,x) 
    D2 = setD(2,x)
    
    print D1
    print D2
    
    # show sparsity pattern of D1
    pylab.figure()
    pylab.spy(D1)
    pylab.title("Sparsity pattern of D1")
        
    # plot a function and it's derivatives
    y = f(x)    
    pylab.figure()
    pylab.plot(x,y,x,D1*y,x,D2*y)
    pylab.legend(['f','D1*f','D2*f'],loc="best")
    if title:
        pylab.title(title)
    pylab.show()


def main():
    # set numpy grid array to be evenly spaced    
    x = np.linspace(0,2,30)
    # choose a simple quadratic function     
    def f(x):
        return x**2 + x
        
    plotDf(x,f,"f=x^2 + x")

    

if __name__ == "__main__":
    main()
