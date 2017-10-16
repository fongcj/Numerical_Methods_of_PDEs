# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:31:53 2012

@author: cjf2123
"""
import numpy as np
import pylab
from scipy.optimize import newton

dt = .01
l = -.8
uo = 1
a = 0
b = 1
k = np.linspace(a,b,(b-a)/dt+1)
z = l*k
print '--------------------------------------------------------------------'
nptsx = 501
nptsy = 501
xa = -3 
xb = 3
ya = -4 
yb = 4
x = np.linspace(xa,xb,nptsx)
y = np.linspace(ya,yb,nptsy)
[X,Y] = np.meshgrid(x,y)
Z = X + 1j*Y
#----------------------------------------------------------------------------
#Evaluation function
def feval(funcName, *args):
    return eval(funcName)(*args)
#True solution
def expo(z):
    R = np.exp(z)
    return R
utrue = expo(z)
def steps(k):
    t = np.round(5.0/k)
    return t
#----------------------------------------------------------------------------
#Forward Euler
def Rfe(z): 
    R = (1 + z)
    return R
RFE = Rfe(z)                    #Part a            
Rvalfe = np.abs(feval('Rfe',Z)) #Part b
Rvalfe[Rvalfe<1] = 0            #Part c
Rvalfe[Rvalfe>0] = 1.0
Xt = X[Rvalfe==0.0]
Yt = Y[Rvalfe==0.0]
dist =  np.sqrt(Xt**2+Yt**2)
Zmaxfe = np.max(dist)
print 'Max step - Forward Euler: ', Zmaxfe
def rminfe(z):                  #Part d
    r = np.abs(Rfe(z) - expo(z))/np.abs(expo(z))
    return r
rmin_fe = newton(rminfe,.001,tol=1e-6,maxiter=40)
print 'Min step for objective function - Forward Euler:', rmin_fe
step_fe = steps(rmin_fe)        #Part d2
print 'Number of steps to reach t = 5/lambda:', step_fe
print '--------------------------------------------------------------------'
#----------------------------------------------------------------------------
#Backward Euler
def Rbe(z):
    R = 1./(1-z)
    return R
RBE = Rbe(z)                    #Part a 
Rvalbe = np.abs(feval('Rbe',Z)) #Part b 
Rvalbe[Rvalbe<1] = 0            #Part c
Rvalbe[Rvalbe>0] = 1
Xt = X[Rvalbe==0]
Yt = Y[Rvalbe==0]
dist =  np.sqrt(Xt**2+Yt**2)
Zmaxbe = np.max(dist)
print 'Max step - Backward Euler: ', Zmaxbe
def rminbe(z):                  #Part d
    r = np.abs(Rbe(z)-expo(z))/np.abs(expo(z))
    return r
rmin_be = newton(rminbe,.001,tol=1e-6,maxiter=40)
print 'Min step for objective function - Backward Euler:', rmin_be
step_be = steps(rmin_be)        #Part d2
print 'Number of steps to reach t = 5/lambda:', step_be
print '--------------------------------------------------------------------'
#----------------------------------------------------------------------------
#Midpoint 
def Rmid(z):
    R = (1 + z + .5*z**2)
    return R
RMID = Rmid(z)                  #Part a 
Rvalmid = np.abs(feval('Rmid',Z))   #Part b
Rvalmid[Rvalmid<1] = 0          #Part b
Rvalmid[Rvalmid>0] = 1
XX = X[Rvalmid==0]
YY = Y[Rvalmid==0]
dist =  np.sqrt(XX**2+YY**2)
Zmaxmid = np.max(dist)
print 'Max step - Midpoint: ', Zmaxmid
def rminmid(z):                  #Part d
    r = np.abs(Rmid(z)-expo(z))/np.abs(expo(z))
    return r
rmin_mid = newton(rminmid,.001,tol=1e-6,maxiter=40)
print 'Min step for objective function - Midpoint:', rmin_mid
step_mid = steps(rmin_mid)        #Part d2
print 'Number of steps to reach t = 5/lambda:', step_mid
print '--------------------------------------------------------------------'
#----------------------------------------------------------------------------
#Trapezoidal
def Rt(z):
    R = (1+z/2)/(1-z/2)
    return R
RT= Rt(z)                       #Part a 
Rvalt = np.abs(feval('Rt',Z))   #Part b
Rvalt[Rvalt<1] = 0              #Part c
Rvalt[Rvalt>0] = 1
XX = X[Rvalt==0]
YY = Y[Rvalt==0]
dist =  np.sqrt(XX**2+YY**2)
Zmaxt = np.max(dist)
print 'Max step - Trapezoidal: ', Zmaxt
def rmint(z):                  #Part d
    r = np.abs(Rt(z)-expo(z))/np.abs(expo(z))
    return r
rmin_t = newton(rmint,.001,tol=1.0e-6,maxiter=40)
print 'Min step for objective function - Trapezoidal:', rmin_t
step_t = steps(rmin_t)        #Part d2
print 'Number of steps to reach t = 5/lambda:', step_t
print '--------------------------------------------------------------------'
#----------------------------------------------------------------------------
#Runge Kutta
def Rrk(z):
    R = (z**4)/24 + (z**3)/6 + (z**2)/2 + z + 1
    return R
RRK = Rrk(z)                    #Part a 
Rvalrk = np.abs(feval('Rrk',Z)) #Part b
Rvalrk[Rvalrk<1] = 0            #Part c
Rvalrk[Rvalrk>0] = 1
XX = X[Rvalrk==0]
YY = Y[Rvalrk==0]
dist =  np.sqrt(XX**2+YY**2)
Zmaxrk = np.max(dist)
print 'Max step - Runge-Kutta: ', Zmaxrk
def rminrk(z):                  #Part d
    r = np.abs(Rrk(z)-expo(z))/np.abs(expo(z))
    return r
rmin_rk = newton(rminrk,.001,tol=1.0e-6,maxiter=40)
print 'Min step for objective function - Runge-Kutta:', rmin_rk
step_rk = steps(rmin_rk)        #Part d2
print 'Number of steps to reach t = 5/lambda:', step_rk
print '--------------------------------------------------------------------'
#----------------------------------------------------------------------------
fig = pylab.figure()
eh1x = fig.add_subplot(1,1,1)
pylab.title("Problem 1b: Comparison of R(z) and exp(z)")
line1 = eh1x.plot(k,utrue,k,RFE,k,RBE,k,RMID,k,RT,k,RRK)
eh1x.legend(["Utrue","Euler (F)","Euler (B)","Midpoint","Trapezoidal","Runge-Kutta"],loc="best")
eh1x.set_xlabel("Time (k)")
eh1x.set_ylabel("Solution (U)")
pylab.show()

fig1 = pylab.figure()
eh = fig1.add_subplot(3,2,1)
line1 = eh.imshow(Rvalfe,extent=[xa,xb,ya,yb])
pylab.title("Region of absolute stability - Forward Euler")
eh.grid()

eh2 = fig1.add_subplot(3,2,2)
line2 = eh2.imshow(Rvalbe,extent=[xa,xb,ya,yb])
pylab.title("Region of absolute stability - Backward Euler")
eh2.grid()

eh3 = fig1.add_subplot(3,2,3)
line3 = eh3.imshow(Rvalmid,extent=[xa,xb,ya,yb] )
pylab.title("Region of absolute stability - Midpoint")
eh3.grid()

eh4 = fig1.add_subplot(3,2,4)
line4 = eh4.imshow(Rvalt,extent=[xa,xb,ya,yb])
pylab.title("Region of absolute stability - Trapezoidal")
eh4.grid()

eh5 = fig1.add_subplot(3,2,5)
line5 = eh5.imshow(Rvalrk,extent=[xa,xb,ya,yb])
pylab.title("Region of absolute stability - Runge-Kutta")
eh5.grid()
pylab.show()

