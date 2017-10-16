# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 17:08:19 2012

@author: cjf2123
"""
#Chris Fong
#APMA 4301 
#Homework 2a Problem 1

import scipy as sp
import numpy as np
import pylab
import diffMatrix as dm



#N = array([8, 16, 32, 64, 128, 256, 512, 1024])
#M = array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
N = array([128])
M = array([128.0])
stencil_size = array([3,5])
h = 1./M
eh13 = sp.zeros((len(N),len(h)))
eh23 = sp.zeros((len(N),len(h)))
eh15 = sp.zeros((len(N),len(h)))
eh25 = sp.zeros((len(N),len(h)))
ehQ1_13 = sp.zeros((len(N),len(h)))
ehQ1_23 = sp.zeros((len(N),len(h)))
for i in range(len(N)):
    
    x = np.linspace(0,1,N[i]+1)
    f = x**2 + np.sin(4*pi*x)
    fp = 2*x + 4*(np.pi)*(np.cos(4*pi*x))
    fpp = 2 - 16*(np.pi**2)*(np.sin(4*pi*x))
    D1f3 = dm.setDn(1,x,3)*f
    D2f3 = dm.setDn(2,x,3)*f
    D1f5 = dm.setDn(1,x,5)*f
    D2f5 = dm.setDn(2,x,5)*f
    
    #compute error between problem 1 and 2  
    fppp = -64*(np.pi**3)*(np.cos(4*pi*x))
    for j in range(len(h)):
    #compute absolute mesh error
        eh13[i,j] = np.sqrt(h[j])*(np.linalg.norm(D1f3-fp))
        eh23[i,j] = np.sqrt(h[j])*(np.linalg.norm(D2f3-fpp))
        eh15[i,j] = np.sqrt(h[j])*(np.linalg.norm(D1f5-fp))
        eh25[i,j] = np.sqrt(h[j])*(np.linalg.norm(D2f5-fpp))
               
        #compute error between problem 1 and 2  
        error_Q1_1D = (h[j]**3)/6*fppp      #First error term
        error_Q1_2D = (h[j]**2)/12*fppp
        ehQ1_13[i,j] = np.sqrt(h[j])*(np.linalg.norm(error_Q1_1D))
        ehQ1_23[i,j] = np.sqrt(h[j])*(np.linalg.norm(error_Q1_2D))
    
eh13 = np.transpose(eh13)         #N increases as down the rows
eh23 = np.transpose(eh23)
eh15 = np.transpose(eh15)         
eh25 = np.transpose(eh25)  
ehQ1_13 = np.transpose(ehQ1_13)         
ehQ1_23 = np.transpose(ehQ1_23)

#Problem 2 (a) (i)
fig = pylab.figure()
eh1x = fig.add_subplot(2,1,1)
pylab.title("Problem 2_a_i: 1st derivative- Log-Log Plot of h vs eh")
eh2x = fig.add_subplot(2,1,2)
pylab.title("Problem 2_a_i: 2nd derivative: Log-Log Plot of h vs eh")
line1 = eh1x.plot(h,eh13)
line2 = eh2x.plot(h,eh23)
eh1x.set_yscale('log')
eh1x.set_xscale('log')
eh2x.set_yscale('log')
eh2x.set_xscale('log')
eh1x.legend(["N=8","N=16","N=32","N=64","N=128","N=256","N=512","N=1024"],loc="best")
eh2x.legend(["N=8","N=16","N=32","N=64","N=128","N=256","N=512","N=1024"],loc="best")
eh1x.set_xlabel("Step Size (h)")
eh2x.set_xlabel("Step Size (h)")
eh1x.set_ylabel("Error (eh)")
eh2x.set_ylabel("Error (eh)")
pylab.show()

#Problem 2 (a) (ii)
hlog = np.log10(h)
eh13log = np.log10(eh13)
eh23log = np.log10(eh23)
(p13,b13) = np.polyfit(hlog,eh13log,1)
(p23,b23) = np.polyfit(hlog,eh23log,1)

print 'Problem 2_a_ii: Order of convergence of the Error, 1st derivative, : ', str(p13), ' for h = ', str(h),'respectively. \r'
print 'Problem 2_a_ii: Order of convergence of the Error, 2nd derivative, : ', str(p23), ' for h = ', str(h),'respectively. \r'


#Problem 2 (a) (iii)
#The first error term for the centered difference we found from problem 1 was
#h^3/6*u'''(x) for the first derivative and 
#h^2/12*u'''(x) for the second derivative
#The 2-norm of this vector will be compared with our calculated setD*f - f',f''
fig = pylab.figure()
eh1x = fig.add_subplot(2,1,1)
pylab.title("Problem 2_a_iii: 1st derivative- Log-Log Plot of h vs eh")
eh2x = fig.add_subplot(2,1,2)
pylab.title("Problem 2_a_iii: 2nd derivative: Log-Log Plot of h vs eh")
line1 = eh1x.plot(h,ehQ1_13)
line2 = eh2x.plot(h,ehQ1_23)
eh1x.set_yscale('log')
eh1x.set_xscale('log')
eh2x.set_yscale('log')
eh2x.set_xscale('log')
eh1x.legend(["N=8","N=16","N=32","N=64","N=128","N=256","N=512","N=1024"],loc="best")
eh2x.legend(["N=8","N=16","N=32","N=64","N=128","N=256","N=512","N=1024"],loc="best")
eh1x.set_xlabel("Step Size (h)")
eh2x.set_xlabel("Step Size (h)")
eh1x.set_ylabel("Error (eh)")
eh2x.set_ylabel("Error (eh)")


#Problem 2 (b) (i)
fig = pylab.figure()
eh1x = fig.add_subplot(2,1,1)
pylab.title("Problem 2_b_i: 1st derivative- Log-Log Plot of h vs eh")
eh2x = fig.add_subplot(2,1,2)
pylab.title("Problem 2_b_i: 2nd derivative: Log-Log Plot of h vs eh")
line1 = eh1x.plot(h,eh15)
line2 = eh2x.plot(h,eh25)
eh1x.set_yscale('log')
eh1x.set_xscale('log')
eh2x.set_yscale('log')
eh2x.set_xscale('log')

eh1x.legend(["N=8","N=16","N=32","N=64","N=128","N=256","N=512","N=1024"],loc="best")
eh2x.legend(["N=8","N=16","N=32","N=64","N=128","N=256","N=512","N=1024"],loc="best")
eh1x.set_xlabel("Step Size (h)")
eh2x.set_xlabel("Step Size (h)")
eh1x.set_ylabel("Error (eh)")
eh2x.set_ylabel("Error (eh)")
pylab.show()

#Problem 2 (b) (ii)
hlog = np.log10(h)
eh15log = np.log10(eh15)
eh25log = np.log10(eh25)
(p15,b15) = np.polyfit(hlog,eh15log,1)
(p25,b25) = np.polyfit(hlog,eh25log,1)

print 'Problem 2_b_ii: Order of convergence of the Error, 1st derivative, : ', str(p15), ' for h = ', str(h),'respectively. \r'
print 'Problem 2_b_ii: Order of convergence of the Error, 2nd derivative, : ', str(p25), ' for h = ', str(h),'respectively. \r'


