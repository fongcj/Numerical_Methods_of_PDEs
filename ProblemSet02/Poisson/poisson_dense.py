"""
poisson_dense.py  -- solve the Poisson problem u_{xx} + u_{yy} = f(x,y)
                on [a,b] x [a,b].
                
                using python/numpy dense linear algebra

     The 5-point Laplacian is used at interior grid points.
     This system of equations is then solved using numpy.linalg.solve
     which defaults to umfpack

     code modified from poisson.py from 
     http://www.amath.washington.edu/~rjl/fdmbook/chapter3  (2007)
"""

import numpy as np
#from numpy.linalg  import solve
from scipy.linalg import solve
from scipy.linalg import toeplitz
import pylab
from mpl_toolkits.mplot3d import Axes3D

a = 0.0
b = 1.0
m = 75                     # number of interior points
h = (b-a)/(m+1)
x = np.linspace(a,b,m+2)   # grid points x including boundaries
y = np.linspace(a,b,m+2)   # grid points y including boundaries


X,Y = np.meshgrid(x,y)     # 2d arrays of x,y values
X = X.T                    # transpose so that X(i,j),Y(i,j) are
Y = Y.T                    # coordinates of (i,j) point

Xint = X[1:-1,1:-1]        # interior points
Yint = Y[1:-1,1:-1]

def f(x,y):
    return -np.ones(x.shape)

rhs = f(Xint,Yint)         # evaluate f at interior points for right hand side
                           # rhs is modified below for boundary conditions.

# set boundary conditions around edges of usoln array:

usoln = np.zeros(X.shape)     # here we just zero everything  
                           # This sets full array, but only boundary values
                           # are used below.  For a problem where utrue
                           # is not known, would have to set each edge of
                           # usoln to the desired Dirichlet boundary values.


# adjust the rhs to include boundary terms: 
rhs[:,0] -= usoln[1:-1,0] / h**2
rhs[:,-1] -= usoln[1:-1,-1] / h**2
rhs[0,:] -= usoln[0,1:-1] / h**2
rhs[-1,:] -= usoln[-1,1:-1] / h**2


# convert the 2d grid function rhs into a column vector for rhs of system:
F = rhs.reshape((m*m,1))

# form matrix A:
I = np.eye(m,m)
e = np.ones(m)
Tcol = np.zeros( (m,) )
Tcol[:2] = [-4, 1]
T = toeplitz(Tcol)
Scol = np.zeros( (m,) )
Scol[:2] = [0, 1]

S = toeplitz(Scol)
A = (np.kron(I,T) + np.kron(S,I)) / h**2

show_matrix = False
if (show_matrix):
    pylab.spy(A,marker='.')
    pylab.show()

# Solve the linear system:

uvec = solve(A, F)

# reshape vector solution uvec as a grid function and
# insert this interior solution into usoln for plotting purposes:
# (recall boundary conditions in usoln are already set)

usoln[1:-1, 1:-1] = uvec.reshape( (m,m) )

# using Linf norm of spectral solution good to 10 significant digits
umax_true = 0.07367135328
umax = usoln.max()
abs_err = abs(umax - umax_true)
rel_err = abs_err/umax_true
print "m = {0}".format(m)
print "||u||_inf = {0}, ||u_true||_inf={1}".format(umax,umax_true)
print "Absolute error = {0:10.3e}, relative error = {1:10.3e}".format(abs_err,rel_err)

show_result = True
if (show_result):
# plot results:
    ax = Axes3D(pylab.gcf())
    ax.plot_surface(X,Y,usoln, rstride=1, cstride=1, cmap=pylab.cm.jet)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    #pylab.axis([a, b, a, b])
    #pylab.daspect([1 1 1])
    pylab.title('Surface plot of computed solution')
    pylab.show()
