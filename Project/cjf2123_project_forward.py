# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 23:29:45 2012

@author: cjf2123
"""
#Steady state forward problem is built and computed here
import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse.linalg import dsolve
from matplotlib import pylab
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import cjf2123_project_plotData as pd

class output_struct:
    U    = None
    Usq = None
    left    = None
    right    = None
    upper    = None
    lower    = None
    C = 0.0
    M = None
    time = 0.0
    timePerGrid = 0.0
    its = 0.0
    rnorm = 0.0
#-----------------------------------------------------------------------------
def build_A(inputd,self):
    n       = inputd.n
    lx      = inputd.lx
    ly      = inputd.ly
    nx      = int(inputd.nx)
    ny      = int(inputd.ny)
    dx      = inputd.dx
    dy      = inputd.dy
    x       = inputd.x
    y       = inputd.y
    D       = inputd.D
    mua     = inputd.mua
    qin     = inputd.qin
    Reff    = inputd.Reff
    C       = inputd.C    
    kspType = inputd.kspType
    
    nz      = nx*ny*5-2*nx-2*ny
    row     = np.zeros(nz)
    col     = np.zeros(nz)
    val     = np.zeros(nz)
    R       = np.zeros(nx*ny)
    ind     = 0

    M = None
    for i in range(nx): #Rows
        for j in range(ny): #Columns
            row_num = int((i)*ny + j)
            if i>0 and i<nx-1 and j>0 and j<ny-1:       #Non-boundary case
                row[ind:ind+5]	= row_num
                col[ind:ind+5]	= [row_num-ny, row_num+ny,row_num-1, row_num+1, row_num]
                val[ind]        = -dy/dx*(D[i,j]+D[i-1,j])/2.0
                val[ind+1]      = -dy/dx*(D[i,j]+D[i+1,j])/2.0
                val[ind+2]      = -dx/dy*(D[i,j]+D[i,j-1])/2.0
                val[ind+3]      = -dx/dy*(D[i,j]+D[i,j+1])/2.0
                val[ind+4]      = mua[i,j]*dx*dy - sum(val[ind:ind+5])             
                ind             = ind + 5
            elif i==0 and j>0 and j<ny-1:          #Top row, except the corners This is where the source is illuminating
                row[ind:ind+4]  = row_num
                col[ind:ind+4]  = [row_num+ny, row_num-1,row_num+1, row_num]
                val[ind]        = -dy/dx*(D[i,j]+D[i+1,j])/2.0
                val[ind+1]      = -dx/2.0/dy*(D[i,j]+D[i,j-1])/2.0
                val[ind+2]      = -dx/2.0/dy*(D[i,j]+D[i,j+1])/2.0
                val[ind+3]      = mua[i,j]*dx*dy/2.0 + dy*C - sum(val[ind:ind+4])
                R[row_num]      = 4.0*dy*C*qin[i,j]
                ind             = ind + 4
            elif i==nx-1 and j>0 and j<ny-1:        #Bottom row, except the corners This is where the detectors are
                row[ind:ind+4]  = row_num
                col[ind:ind+4]  = [row_num-ny, row_num-1, row_num+1, row_num]
                val[ind]        = -dy/dx*(D[i,j]+D[i-1,j])/2.0
                val[ind+1]      = -dx/2.0/dy*(D[i,j]+D[i,j-1])/2.0
                val[ind+2]      = -dx/2.0/dy*(D[i,j]+D[i,j+1])/2.0
                val[ind+3]      = mua[i,j]*dx*dy/2.0 + dy*C - sum(val[ind:ind+4])    
                R[row_num]      = 4.0*dy*C*qin[i,j]
                ind             = ind + 4
            elif j==0 and i>0 and i<nx-1:          #First column, except corners. 
                row[ind:ind+4]  = row_num
                col[ind:ind+4]  = [row_num-ny, row_num+ny, row_num+1, row_num]
                val[ind]        = -dy/dx/2.0*(D[i,j]+D[i-1,j])/2.0
                val[ind+1]      = -dy/dx/2.0*(D[i,j]+D[i+1,j])/2.0
                val[ind+2]      = -dx/dy*(D[i,j]+D[i,j+1])/2.0
                val[ind+3]      = mua[i,j]*dx*dy/2.0 + dx*C - sum(val[ind:ind+4])
                R[row_num]      = 4.0*dx*C*qin[i,j]
                ind             = ind + 4
            elif j==ny-1 and i>1 and i<nx-1:         #Last column, except corners. 
                row[ind:ind+4]  = row_num
                col[ind:ind+4]  = [row_num-ny, row_num+ny, row_num-1, row_num]
                val[ind]        = -dy/dx/2.0*(D[i,j]+D[i-1,j])/2.0
                val[ind+1]      = -dy/dx/2.0*(D[i,j]+D[i+1,j])/2.0
                val[ind+2]      = -dx/dy*(D[i,j]+D[i,j-1])/2.0
                val[ind+3]      = mua[i,j]*dx*dy/2.0 + dx*C - sum(val[ind:ind+4])
                R[row_num]      = 4.0*dx*C*qin[i,j]
                ind             = ind + 4
            elif i==0 and j==0:                 #Upper left corner
                row[ind:ind+3]  = row_num
                col[ind:ind+3]  = [row_num+ny, row_num+1, row_num]
                val[ind]        = -dy/dx/2.0*(D[i,j]+D[i+1,j])/2.0
                val[ind+1]      = -dx/dy/2.0*(D[i,j]+D[i,j+1])/2.0
                val[ind+2]      = mua[i,j]*dx*dy/4.0 + (dx+dy)/2.0*C - sum(val[ind:ind+3])
                R[row_num]      = 2.0*(dx+dy)*C*qin[i,j]
                ind             = ind + 3
            elif i==0 and j==ny-1:                #Upper right corner
                row[ind:ind+3]  = row_num
                col[ind:ind+3]  = [row_num+ny, row_num-1, row_num]
                val[ind]        = -dy/dx/2.0*(D[i,j]+D[i+1,j])/2.0
                val[ind+1]      = -dx/dy/2.0*(D[i,j]+D[i,j-1])/2.0
                val[ind+2]      = mua[i,j]*dx*dy/4.0 + (dx+dy)/2.0*C - sum(val[ind:ind+3])
                R[row_num]      = 2.0*(dx+dy)*C*qin[i,j]
                ind             = ind + 3
            elif i==nx-1 and j==0:                #Lower left corner
                row[ind:ind+3]  = row_num
                col[ind:ind+3]  = [row_num-ny, row_num+1, row_num]
                val[ind]        = -dy/dx/2.0*(D[i,j]+D[i-1,j])/2.0
                val[ind+1]      = -dx/dy/2.0*(D[i,j]+D[i,j+1])/2.0
                val[ind+2]      = mua[i,j]*dx*dy/4.0 + (dx+dy)/2.0*C - sum(val[ind:ind+3])
                R[row_num]      = 2.0*(dx+dy)*C*qin[i,j]
                ind             = ind + 3
            else:                               #Lower right corner
                row[ind:ind+3]  = row_num
                col[ind:ind+3]  = [row_num-ny, row_num-1, row_num]
                val[ind]        = -dy/dx/2.0*(D[i,j]+D[i-1,j])/2.0
                val[ind+1]      = -dx/dy/2.0*(D[i,j]+D[i,j-1])/2.0
                val[ind+2]      = mua[i,j]*dx*dy/4.0 + (dx+dy)/2.0*C - sum(val[ind:ind+3])
                R[row_num]      = 2.0*(dx+dy)*C*qin[i,j]
                ind             = ind + 3
            if kspType != 'sd':
                PETSc.Vec.setValue(self.b,row_num,R[row_num])
            
    M = sps.csr_matrix((val,(row,col)),shape=(nx*ny,nx*ny))
    return(M,R,row,col,val)
#-----------------------------------------------------------------------------
def solve_forward(inputd):
    nx = inputd.nx
    ny = inputd.ny
    C = inputd.C
    # run solution
    [pde, time, timePerGrid,its, rnorm, U, M] = solveDiffusion(nx,ny,inputd) 

    output = output_struct
    if inputd.kspType == 'sd':
        output.U = U
        output.M = M
    else:
        output.U = pde.x.getArray()
        output.M = pde.A
    output.time = time
    output.timePerGrid = timePerGrid
    output.its = its
    output.rnorm = rnorm

    output.Usq      = np.reshape(output.U,(ny,nx))    
    output.left    = C*output.Usq[:,0] 
    output.right    = C*output.Usq[:,nx-1]
    output.upper    = C*output.Usq[ny-1,:] 
    output.lower    = C*output.Usq[0,:]
    
    return(output)
#-----------------------------------------------------------------------------     
def solveDiffusion(nx,ny,inputd):
    # Create the DA and setup the problem    
    if inputd.kspType == 'sd':
        U = 0.0
        self = 0.0
        [M,R,row,col,val] = build_A(inputd,self) 
        tic = PETSc.Log().getTime()
        U = dsolve.spsolve(M, R,use_umfpack=True)           #Direct solver   
        toc = PETSc.Log().getTime()
        its = 1.0
        rnorm = np.linalg.norm(M*U-R)
        pde = 0.0
    else:    
        da = PETSc.DA().create([nx, ny], stencil_width=1, boundary_type=('ghosted', 'ghosted'))    
        pde = diffusion2D(da)                        
        pde.setupSolver(inputd)   
        print "Solving using", pde.solver_name
        #Now solve
        tic = PETSc.Log().getTime()
        pde.solve(pde.b, pde.x)   
        toc = PETSc.Log().getTime()
        its = pde.ksp.getIterationNumber()
        rnorm = pde.ksp.getResidualNorm()
        M = 0.0
        U = M
     
    # output performance/convergence information
    time = toc-tic
    timePerGrid = time/nx/ny
    
    #rnorm = np.linalg.norm(M*U-R)
    
    
    print "Nx Ny its    ||r||_2     ElapsedTime (s)    Elapsed Time/N_tot"    
    print nx,ny, its, rnorm, time, timePerGrid    

    return pde, time, timePerGrid, its, rnorm, U, M
#-----------------------------------------------------------------------------    
class diffusion2D(object):

    def __init__(self, da):
        assert da.getDim() == 2
        self.da = da
        # create solution vector
        self.x = da.createGlobalVec()
        # create RHS vector
        self.b = da.createGlobalVec()
        # create the Operator Matrix
        self.A = da.getMatrix('aij')
        self.row = da.createGlobalVec()
        self.col = da.createGlobalVec()
                  
    def formOperator(self,inputd):
        mx, my = self.da.getSizes()
        hx, hy = [1.0/m for m in [mx, my]]
        ihx2 = 1./hx/hx
        ihy2 = 1./hy/hy
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        [M,R,row,col,val] = build_A(inputd,self) 
        
        row = row.astype(int)
        col = col.astype(int)
        
        for i in range(len(row)):          
            jj = row[i]
            ii = col[i]
            PETSc.Mat.setValue(self.A,ii,jj,val[i])
        PETSc.Mat.setValue(self.A,0,0,val[2])    
        self.A.assemblyBegin() # Make matrices useable.
        self.A.assemblyEnd()
        
    def setupSolver(self,inputd):
        self.ksp = PETSc.KSP().create()
        self.pc = self.ksp.getPC()
        self.pc.setDM(self.da)
        self.formOperator(inputd)
        self.ksp.setOperators(self.A)
        self.ksp.setConvergenceHistory(length=1000)
        self.ksp.setFromOptions()
        self.pc.setType(inputd.pcType)
        self.ksp.setType(inputd.kspType)
        self.solver_name='({0},{1})'.format(self.pc.getType(),self.ksp.getType())
        
    def solve(self,b,x):
        self.ksp.solve(b,x)
        return x


