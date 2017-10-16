# This demo solves the time-dependent diffusion equation using FEniCS
#
# Modified from dolfin advection-Diffusion demo.;y


__author__ = "Marc Spiegelman (mspieg@ldeo.columbia.edu)"
__date__ = "21 Oct 2009 09:59:56"
__copyright__ = "Copyright (C) 2009 Marc Spiegelman"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# creat Mesh on unit interval
mesh = UnitSquare(100,100)

# Create FunctionSpaces
V = FunctionSpace(mesh, "CG", 1)


# set initial condition and project onto u0
amp = 1.;
x0 = 0.5
sigma = .05
gaussian = "%f*exp(-((x[0]-.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))/pow(%f,2))" % (amp,sigma)
f  = Expression(gaussian)
# Functions
u0 = Function(V)
u1 = Function(V)

#initial condition
u1.interpolate(f)
#u1 = project(f,V)

# Parameters
T = 0.01
beta = 4
k = (2*mesh.hmin()**2)
print "dt=%g" % k
t = k


# Test and trial functions
v = TestFunction(V)
u = TrialFunction(V)


# Variational problem
a = v*u*dx + 0.5*k*inner(grad(v),grad(u))*dx
L = v*u0*dx - 0.5*k*inner(grad(v),grad(u0))*dx

# # Set up boundary condition
# g  = Constant(mesh, 1.0)
# bc = DirichletBC(V, g, sub_domains, 1)

# Assemble matrix
A = assemble(a)

# Output file
out_file = File("temperature.pvd","compressed")

# output initial condition
out_file << u1

# Time-stepping
while t < T:

    # Copy solution from previous interval
    u0.assign(u1)

    # Assemble vector and apply boundary conditions
    b = assemble(L)
    #bc.apply(A, b)

    # Solve the linear system
    solve(A, u1.vector(), b)

    # plot solutions
    #plot(u1)

    # Save the solution to file
    out_file << u1

    # Move to next interval
    t += k

