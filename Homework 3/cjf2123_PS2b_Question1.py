# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:12:09 2012

@author: cjf2123
"""
import scipy
import scipy.sparse as sp
import numpy as np
#from scipy.sparse.linalg import spsolve
#import pylab
#from mpl_toolkits.mplot3d import Axes3D

a = 0.0
b = 1.0
N = 100
h = 1/(N-1.0)
n = 2
m = 2
I = sp.eye(N,N)
g = np.ones(N)
T = sp.spdiags([g,-4*g,g],[-1,0,1],N,N)
S = sp.spdiags([g,g],[-1,1],N,N)
A = (sp.kron(I,T) + sp.kron(S,I)) / (h**2)
A = A.tocsr()
u = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        u[i,j] = np.sin(m*np.pi*i*h)*np.sin(n*np.pi*j*h)
uvec = u.reshape((N*N,1))
duvec = A*uvec
du = duvec.reshape((N,N))   
p=0

        