"""
Chris Fong cjf2123
PS #2b
Question 3d
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pylab
from mpl_toolkits.mplot3d import Axes3D

show_result = True
a = 0.0
b = 1.0
mx = np.array([256])                     # number of interior points in each direction
my = np.array([256])                     # number of interior points in each direction
m = np.array([16,32,64,128,256])  
hx = (b-a)/(mx+1)
hy = (b-a)/(my+1)
h = (b-a)/(m+1)
err2norm = np.zeros((len(h),1))
umax = np.zeros((len(h),1))
for i in range(len(m)):
    x = np.linspace(a,b,m[i]+2)   # grid points x including boundaries
    y = np.linspace(a,b,m[i]+2)   # grid points y including boundaries
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
    rhs[:,0] -= usoln[1:-1,0] / h[i]**2
    rhs[:,-1] -= usoln[1:-1,-1] / h[i]**2
    rhs[0,:] -= usoln[0,1:-1] / h[i]**2
    rhs[-1,:] -= usoln[-1,1:-1] / h[i]**2  
    
    # convert the 2d grid function rhs into a column vector for rhs of system:
    F = rhs.reshape((m[i]*m[i],1))
    
    # form matrix A:
    I = sp.eye(m[i],m[i])
    g = np.ones(m[i])
    T = sp.spdiags([g,-4.*g,g],[-1,0,1],m[i],m[i])
    S = sp.spdiags([g,g],[-1,1],m[i],m[i])
    A = (sp.kron(I,T) + sp.kron(S,I)) / (h[i]**2)
    A = A.tocsr()
    
    # Solve the linear system:
    uvec = spsolve(A, F)

    # reshape vector solution uvec as a grid function and
    # insert this interior solution into usoln for plotting purposes:
    # (recall boundary conditions in usoln are already set)
    usoln = np.zeros((len(x),len(y)))
    usoln[1:-1, 1:-1] = uvec.reshape( (m[i],m[i]) )

    # using Linf norm of spectral solution good to 10 significant digits
    umax_true = utrue.max()
#        umax = usoln.max()
    umax[i] = uvec.max()
    err2norm[i] = h[i]*np.linalg.norm(umax[i]-umax_true)
    

    if (show_result) & (i == 1):
    # plot results:
        ax = Axes3D(pylab.gcf())
        ax.plot_surface(X,Y,usoln, rstride=1, cstride=1, cmap=pylab.cm.jet)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        pylab.axis([a, b, a, b])
        pylab.title('Problem 3b: Surface plot of computed solution. M = 32')
        pylab.show()  



err2norm_log = np.log10(err2norm)
hlog = np.log10(h)
(p2norm,bA) = np.polyfit(hlog,err2norm_log,1)

#Display results
fig = pylab.figure()
eh1x = fig.add_subplot(1,1,1)
pylab.title("Problem 3c: Log-Log Plot of h vs Error in the 2-norm")
line1 = eh1x.plot(h,err2norm)
eh1x.set_yscale('log')
eh1x.set_xscale('log')
eh1x.legend(["Error in the 2-Norm"],loc="best")
eh1x.set_xlabel("Step Size (h)")
eh1x.set_ylabel("Error (e)")
pylab.annotate('Order of convergence = {0}'.format(p2norm),(.45,.15),xycoords='axes fraction')
pylab.show()



#print "m = {0}".format(m)
#print "||u||_inf = {0}, ||u_true||_inf={1}".format(umax,umax_true)
#print "Absolute error = {0}, relative error = {1}".format(abs_err,rel_err)
#print "\r"
##

    
