# -*- coding: utf-8 -*-
"""
showjacobispectrum.py
Created on Thu Oct 18 11:08:41 2012

Quicky script to plot spectrum of iteration matrix for Jacobi and damped
Jacobi preconditioned Richardson for the canonical problem 

    \Delsq u = f with dirichlet conditions on a uniform grid over [0,1]x[0,1]

@author: mspieg
"""

from pylab import *

# number of grid points in each direction
m=20
h=1./(m+1) # grid spacing
omega = 2./3. # damping factor

# horizontal and vertical wave numbers
p = linspace(1,m,m)  
q = linspace(1,m,m)

P,Q =meshgrid(p,q)

# Eigenvalues of A, G = (I - D\inv A) and G_omega = (I - omega*D\inv A) 
lambdaA = 2./h/h*(cos(P*pi*h)+cos(Q*pi*h) - 2)
lambdaG = 1. + h**2/4.*lambdaA
lambdaGdamped = 1. + omega*h**2/4.*lambdaA

# plot it all out
figure()
contourf(p,q,lambdaG)
xlabel('p')
ylabel('q')
title('eigenspectrum of Jacobi iteration matrix (I-M^{-1}A)')
colorbar()

figure()
contourf(p,q,lambdaGdamped)
xlabel('p')
ylabel('q')
title('eigenspectrum of damped Jacobi iteration matrix (I-omega*M^{-1}A) omega=2/3')
colorbar()

show()

