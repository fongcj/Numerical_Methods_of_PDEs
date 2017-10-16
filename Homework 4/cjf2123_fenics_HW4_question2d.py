"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1
"""

# Copyright (C) 2007-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-08-16
# Last changed: 2011-06-28

# Begin demo
import numpy as np
import pylab
from dolfin import *
P_elements = 1
x0 = 0
y0 = 0
x1 = 1
y1 = 3
Nx = 32
Ny = 48

E = 0.0
# Create mesh and define function space
mesh = Rectangle(x0,y0,x1,y1,Nx,Ny) #rectangle function [0,1]x[0,3] that's 32x48 in size
V = FunctionSpace(mesh, "Lagrange", P_elements) #piece was linear, quadratic, cubic #elements . This is for part c

# Define Dirichlet boundary (x = 0 or x = 1) figures if x is on boundary
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Define boundary condition. If on the boundary set u to 0
u0 = Expression("exp(x[0]+x[1]/2)") #Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u_e = Expression("exp(x[0]+x[1]/2)")
u = TrialFunction(V)
v = TestFunction(V) # F = int(v(Lu-f)dx). Try to make this zero. Least square problem
f = Expression("1.25*exp(x[0]+x[1]/2)") #what needs #to be appended. The is the forcing term .
#g = Expression("sin(5*x[0])") #g is the gradient
a = inner(grad(u), grad(v))*dx
L = f*v*dx      #+ g*v*ds       # linear form. See notes for continuous equation can't change #is variable

# Compute solution
u = Function(V)
solve(a == L, u, bc)
error = (u-u_e)**2*dx
E = sqrt(assemble(error))
#L2 norm = sqrt(int((u-utrue)^2dx)) dx is dA for 2d and dV for 3d look for examples #and modify
    
# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
plot(u, interactive=True, axes=True, 
         title='surf plot of u',colorbar='on',)




