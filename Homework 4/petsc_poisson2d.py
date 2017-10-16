# -*- coding: utf-8 -*-
"""
petsc_poisson2d.py
Created on Tue Sep  4 22:01:39 2012

@author: mspieg
"""

# Summary
#     Basic use of distributed arrays communication data structures in PETSc
#     to test a wide range of solvers on the basic poisson problem
#
#     Delsq u = -1 u=0 on boundaries of [0,1]x[0,1]
# 
# Examples
#     Direct solve:
#     $ python petsc_poisson2d.py -ksp_monitor -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package umfpack
# 
#     Iterative solve:
#     $ python petsc_poisson2d.py -ksp_monitor -ksp_type cg -pc_type ilu
# 
# Description
#     DAs are extremely useful when working simulations that are discretized
#     on a structured grid. DAs don't actually hold data; instead, they are 
#     templates for distributing and communicating information (matrices and 
#     vectors) across a parallel system.
# 
#     In this example, we set up a simple 2D Poisson equation with dirichlet
#     boundary conditions. The solution, given unit forcing, 
#     is solved using a ksp object.
# 
#
# For more information, consult the PETSc user manual.
# Also, look at the petsc4py/src/PETSc/DA.pyx file.                    

# check to see if xrange is implemented
try: range = xrange
except: pass


import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from matplotlib import pylab
import numpy as np

class Poisson2D(object):

    def __init__(self, da):
        assert da.getDim() == 2
        self.da = da
        # create solution vector
        self.x = da.createGlobalVec()
        # create RHS vector
        self.b = da.createGlobalVec()
        # create the Operator Matrix
        self.A = da.getMatrix('aij')
       
    def formRHS(self):
        # set the RHS to -1.
        self.b.set(-1.)
        return self.b
            
    def formOperator(self):
        mx, my = self.da.getSizes()
        hx, hy = [1.0/m for m in [mx, my]]
        ihx2 = 1./hx/hx
        ihy2 = 1./hy/hy
        (xs, xe), (ys, ye) = self.da.getRanges()
        # Stencil objects make it easy to set the values of the matrix elements.
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        # Set matrix elements to correct values.
        (i0, i1), (j0, j1) = self.da.getRanges()
        for j in range(j0, j1):
            for i in range(i0, i1):
                row.index = (i, j)
                for index, value in [((i, j), -2.*(ihx2 + ihy2)),
                             ((i-1, j), ihx2),
                             ((i+1, j), ihx2),
                             ((i, j-1), ihy2),
                             ((i, j+1), ihy2)]:
                  col.index = index
                  self.A.setValueStencil(row, col, value) # Sets a single matrix element.
                            
        self.A.assemblyBegin() # Make matrices useable.
        self.A.assemblyEnd()
        
    def setupSolver(self):
        self.ksp = PETSc.KSP().create()
        self.pc = self.ksp.getPC()
        self.pc.setDM(self.da)
        self.formOperator()
        self.ksp.setOperators(self.A)
        self.ksp.setConvergenceHistory(length=1000)
        self.ksp.setFromOptions()
        # set (pc,ksp) name
        self.solver_name='({0},{1})'.format(self.pc.getType(),self.ksp.getType())

    def solve(self,b,x):
        self.ksp.solve(b,x)
        return x
        
def solvePoisson(nx,ny):
    # Create the DA.
    
    da = PETSc.DA().create([nx, ny], stencil_width=1, boundary_type=('ghosted', 'ghosted'))
    
    pde = Poisson2D(da)                        
    b = pde.formRHS()
    pde.setupSolver()
    print "Solving using", pde.solver_name
        
    # Solve!
    tic = PETSc.Log().getTime()
    pde.solve(b, pde.x)
    toc = PETSc.Log().getTime()
    time = toc-tic
    
    # output performance/convergence information
    its = pde.ksp.getIterationNumber()
    rnorm = pde.ksp.getResidualNorm()
    print "Nx Ny its    ||r||_2     ElapsedTime (s)    Elapsed Time/N_tot"    
    print nx,ny, its, rnorm, time, time/nx/ny
    
    rh = pde.ksp.getConvergenceHistory()
    
    filename = '{0}_{1}x{2}.npz'.format(pde.solver_name,nx,ny)
    np.savez(filename,rhist=rh,npts=np.array([nx,ny]),name=pde.solver_name,time=time)
    return pde, time

def main():
    # get Dimensions of the 2D grid.
    OptDB = PETSc.Options()
    
    n  = OptDB.getInt('n', 33)
    nx = OptDB.getInt('nx', n)
    ny = OptDB.getInt('ny', n)

    # run solution
    pde, time = solvePoisson(nx,ny)

    # Plot solution
    show_solution = OptDB.getBool('plot_solution',0)
    if show_solution:
        pylab.figure()
        pylab.contourf(pde.da.getVecArray(pde.x)[:])
        pylab.colorbar()
    
    # plot convergence behavior 
    pylab.figure()
    rh = pde.ksp.getConvergenceHistory()
    pylab.semilogy(range(len(rh)),rh,'b-o')
    pylab.xlabel('Iterations')
    pylab.ylabel('||r||_2')
    pylab.title('Convergence Behavior {0}, time={1:8.6f} s'.format(pde.solver_name,time))
    pylab.grid()
    pylab.show()

if __name__ == '__main__':
    main()
