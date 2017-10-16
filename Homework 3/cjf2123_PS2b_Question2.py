"""
Chris Fong cjf2123
PS2b
Question 2
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pylab
from mpl_toolkits.mplot3d import Axes3D

a = 0.0
b = 1.0
m = np.array([16,32,64,128,256])                     # number of interior points in each direction
#m = np.array([32,128])
h = (b-a)*(1/(m+1.0))
abs_err = np.zeros((len(h),1))
rel_err = np.zeros((len(h),1))
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
        return -np.ones(x.shape)

    rhs = f(Xint,Yint)         # evaluate f at interior points for right hand side
                           # rhs is modified below for boundary
    usoln = np.zeros(X.shape) 
    
#        rhs[:,0] -= usoln[1:-1,0] / h**2
#        rhs[:,-1] -= usoln[1:-1,-1] / h**2
#        rhs[0,:] -= usoln[0,1:-1] / h**2
#        rhs[-1,:] -= usoln[-1,1:-1] / h**2                       
    # set boundary conditions around edges of usoln array:
    
    
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
#        usoln[1:-1, 1:-1] = uvec.reshape( (m,m) )
    # reshape vector solution uvec as a grid function and
    # insert this interior solution into usoln for plotting purposes:
    # (recall boundary conditions in usoln are already set)

    usoln[1:-1, 1:-1] = uvec.reshape( (m[i],m[i]) )
    umax = usoln.max()
#    umax[i] = uvec.max()
#    abs_err[i] = abs(umax[i] - umax_true)
#    rel_err[i] = abs(umax[i] - umax_true)/umax_true


#Question 2b - find rate of convergence p
abs_err_log = np.log10(abs_err)
rel_err_log = np.log10(rel_err)

hlog = np.log10(h)

(pAbs,bA) = np.polyfit(hlog,abs_err_log,1)
(pRel,bR) = np.polyfit(hlog,rel_err_log,1)  
#
#print "m = {0}\r\r".format(m)
#print "||u||_inf = {0},\r\r ||u_true||_inf={1} \r\r".format(umax,umax_true)
#print "Absolute error = {0},\r\r Relative error = {1} \r\r".format(abs_err,rel_err)
#print "Order of Convergence: Absolute Error = {0},\r\r Relative error = {1} \r\r".format(pAbs,pRel)

#Display results
#fig = pylab.figure()
#eh1x = fig.add_subplot(1,1,1)
#pylab.title("Problem 2_b: Log-Log Plot of h vs Error")
#line1 = eh1x.plot(h,abs_err,h,rel_err)
#eh1x.set_yscale('log')
#eh1x.set_xscale('log')
#eh1x.legend(["Absolute Error","Relative Error"],loc="best")
#eh1x.set_xlabel("Step Size (h)")
#eh1x.set_ylabel("Error (e)")
#pylab.annotate('p_absolute = {0}'.format(pAbs),(.45,.25),xycoords='axes fraction')
#pylab.annotate('p_relative = {0}'.format(pRel),(.45,.52),xycoords='axes fraction')
#pylab.show()

#print "\r"

show_result = True
if show_result:
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
    
