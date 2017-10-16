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
P_elements = 2
N = np.array([8,16,32,64,128])
h = (1.0)/(N+1.0)
u32 = np.zeros((N[2],N[2]))
E = np.zeros((len(N),1))
for i in range(len(N)):
    # Create mesh and define function space
    mesh = UnitSquare(N[i], N[i]) #rectangle function here mesh cube for 3d
    #mesh = Mesh("mesh/blob.xml")
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
    
    if (i == 2):
        u32 = u
    error = (u-u_e)**2*dx
    E[i] = sqrt(assemble(error))
    #L2 norm = sqrt(int((u-utrue)^2dx)) dx is dA for 2d and dV for 3d look for examples #and modify
    
# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
plot(u32, interactive=True, axes=True, 
         title='surf plot of u',colorbar='on',)

err2norm_log = np.log10(E)
hlog = np.log10(h)
(p2norm,bA) = np.polyfit(hlog,err2norm_log,1)

#Display results
fig = pylab.figure()
eh1x = fig.add_subplot(1,1,1)
pylab.title("Problem 2c: Log-Log Plot of h vs Error in the 2-norm")
line1 = eh1x.plot(h,E)
eh1x.set_yscale('log')
eh1x.set_xscale('log')
eh1x.legend(["Error in the 2-Norm"],loc="best")
eh1x.set_xlabel("Step Size (h)")
eh1x.set_ylabel("Error (e)")
pylab.annotate('Order of convergence = {0}'.format(p2norm),(.45,.15),xycoords='axes fraction')
pylab.show()


