# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:19:36 2012

@author: cjf2123
"""
import numpy as np
#import scipy as sp
import sympy as syp
[xj,t,x] = syp.symbols('xj,t,x')
h = 0.5
N1 = 1-t
N2 = t
x = xj-h*t
X = np.array((0,.5,1.0))
ftrue = X*(1-X)/2.0
fext = x*(1-x)/2.0
f1 = syp.integrate(N1*fext,(t,0,1))
f2 = syp.integrate(N2*fext,(t,0,1))
f1e = f1.evalf(subs={xj:0.0})
f3e = f2.evalf(subs={xj:1.0})
f2e = f1.evalf(subs={xj:0.5})+f2.evalf(subs={xj:0.5})

fi = np.array((f1e,f2e,f3e))
M = 0.5*np.matrix('.33333 .16666 0.0 ; .16666 .66666 .16666 ; 0.0 .16666 .333333')
w = np.linalg.solve(M,fi)

e2 = np.linalg.norm(w-ftrue)
print e2