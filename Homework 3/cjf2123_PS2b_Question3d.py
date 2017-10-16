"""
Chris Fong cjf2123
PS #2b
Question 3d

This script will run the Poisson's equation on a rectangular grid 
with hx = hy = h. 

The mesh for this scipt is preset for x = [0,2] x [0,1] 
and a Mx = bx*15 by  My = by/bx*Mx mesh
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pylab
from mpl_toolkits.mplot3d import Axes3D

show_result = True
a = 0.0
bx = 2.0
by = 1.0
resolution = 15             #Point per units spacing
mx = np.ceil(bx*resolution)                     # number of interior points in each direction
my = np.ceil(by/bx*mx)                            # number of interior points in each direction

h = (bx-a)/(mx+1)
#err2norm = np.zeros((len(h),1))
#umax = np.zeros((len(h),1))
#for i in range(len(mx)):
x = np.arange(a,bx,h)   # grid points x including boundaries
y = np.arange(a,by,h)    # grid points y including boundaries
if len(y) < my+2:
    y1 = np.zeros((my+2))
    y1[0:len(y)] = y
    y1[-1] = by
    y = y1
if len(x) < mx+2:
    x1 = np.zeros((mx+2))
    x1[0:len(x)] = x
    x1[-1] = bx
    x = x1
    
X,Y = np.meshgrid(x,y)     # 2d arrays of x,y values
X = X.T                    # transpose so that X(i,j),Y(i,j) are
Y = Y.T                    # coordinates of (i,j) point

Xint = X[1:-1,1:-1]        # interior points
Yint = Y[1:-1,1:-1]

def f(x,y):
    return 1.25*np.exp(x+y/2)

rhs = f(Xint,Yint)         # evaluate f at interior points for right hand side
                       # rhs is modified below for boundary conditions.

# set boundary conditions around edges of usoln array:
utrue = np.exp(X+Y/2)
usoln = utrue
rhs[:,0] -= usoln[1:-1,0] / h**2
rhs[:,-1] -= usoln[1:-1,-1] / h**2
rhs[0,:] -= usoln[0,1:-1] / h**2
rhs[-1,:] -= usoln[-1,1:-1] / h**2  

# convert the 2d grid function rhs into a column vector for rhs of system:
rhsT = rhs.T
F = rhsT.reshape((mx*my,1))

# form matrix A:
I = sp.eye(my,my)
g = np.ones(mx)
T = sp.spdiags([g,-4.*g,g],[-1,0,1],mx,mx)
I2 = sp.eye(mx,mx) 
S = sp.spdiags([g,g],[-1,1],my,my)
A = (sp.kron(I,T) + sp.kron(S,I2)) / (h**2)
A = A.tocsr()

# Solve the linear system:
uvec = spsolve(A, F)

# reshape vector solution uvec as a grid function and
# insert this interior solution into usoln for plotting purposes:
# (recall boundary conditions in usoln are already set)

usoln[1:-1, 1:-1] = np.reshape(uvec,(mx,my),order='F')

# using Linf norm of spectral solution good to 10 significant digits
umax_true = utrue.max()
#        umax = usoln.max()
umax = uvec.max()
err2norm = h*np.linalg.norm(umax-umax_true)


if (show_result):
# plot results:
    ax = Axes3D(pylab.gcf())
    ax.plot_surface(X,Y,usoln, rstride=1, cstride=1, cmap=pylab.cm.jet)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    pylab.title('Problem 3d: Surface plot of computed solution on rectangular mesh')
    pylab.show()  




print "Domain = ({0},{1}) x ({0},{2}) ".format(a,bx,by)
print "mx = {0},my = {1}, h = {2} ".format(mx,my,h)
print "Error in the 2-Norm = {0}".format(err2norm)
print "\r"
#

    
