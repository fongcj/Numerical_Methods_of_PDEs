# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 22:01:39 2012

@author: christopher Fong
"""
try: range = xrange
except: pass


import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


from matplotlib import pylab
import numpy as np

#Files
import cjf2123_project_createp as crp
import cjf2123_project_tomography_routines as tomo
import cjf2123_project_plotData as pd
#import cjf2123_project_forward as forward

"""
-----------------------------------------------------------------------------
Parameters for tomography are initialized here
"""      
class input_struct:
    sourcePos = np.array((0,1,2,3,4,5))
    n     = 1.313
    lx    = 5.0        #This code must be a square grid or else qin wont work. Fix qin 
    ly    = lx
    dense = 50.0
    nx    = np.int(np.round(dense*lx))
    ny    = np.int(np.round(nx*ly/lx))
    dx    = lx/(nx-1.0)
    dy    = ly/(ny-1.0)
    x     = np.linspace(0,lx,nx)
    y     = np.linspace(0,ly,ny)
    D     = 0.08*np.ones((ny,nx))
    Reff    = -1.4399*n**-2+ 0.7099*n**-1+ 0.6681+ 0.0636*n
    C       = (1.0-Reff)/2.0/(1.0+Reff)      #Partial current coefficient
    mua   = None
    index = 1
    plotTomo = 'n'
    pcType = 'none'     #'none' - No PC, 'ilu' - incomplete LU factorization, 'hypre' (high performance precond)
    kspType = 'sd'   #'cg'-conjugate gradient,'gmres'-generalized minimial residual method 'sd'-sparse direct
"""
-----------------------------------------------------------------------------
Parameters for performance analysis are initialized here
"""    
class final_struct:
    mua = None 
    experiments = None   
    time = None
    timePerGrid = None
    its = None
    rnorm = None
    denseArray = np.array([5,25,100,250])   #i index
    Nwidth = 0.0
    pcArray = ['none','ilu','hypre']        #k index 
    kspArray = ['sd','cg','gmres']          #j index
"""
-----------------------------------------------------------------------------
"""    
def main():
    # get Dimensions of the 2D grid.
    inputd = input_struct   
    inputd.mua = crp.find_mua(inputd,0.5,1.0)
    inputd.mua   = np.reshape(inputd.mua,inputd.nx*inputd.ny,1)
    lx = inputd.lx
    
    final = final_struct
    final.mua = np.zeros((inputd.mua.size,1))
    final.experiments = np.ones((inputd.nx,len(inputd.sourcePos)))
    #Try different Matrix sizes maybe 5
    #Different preconditioners: None, ILU, hypre (high performance precond)
    #Different solvers: CG, GMRES, Sparse direct, Richardson used thousands of iterations. Do not include
    sdDone = 0
    denseArray = final.denseArray
    final.Nwidth = denseArray*lx
    pcArray = final.pcArray
    kspArray = final.kspArray
    final.time          = np.zeros((len(denseArray),len(kspArray),len(pcArray)))
    final.timePerGrid   = np.zeros((len(denseArray),len(kspArray),len(pcArray)))
    final.its           = np.zeros((len(denseArray),len(kspArray),len(pcArray)))
    final.rnorm         = np.zeros((len(denseArray),len(kspArray),len(pcArray)))
    for j in range(len(kspArray)):
        inputd.kspType = kspArray[j]
        for k in range(len(pcArray)):
            inputd.pcType = pcArray[k]
            for i in range(len(denseArray)):
                inputd.dense = denseArray[i]
                if j == 0 and k == 0:
                    results = tomo.tomography(inputd,final)
                    if i == len(denseArray)-1:
                        sdDone = 1
                else:               
                    if sdDone == 1 and j > 0:
                        hfafda = 0
                        print hfafda
                        results = tomo.tomography(inputd,final)           
                final.time[i,j,k] = np.mean(results.time)
                final.timePerGrid[i,j,k] = np.mean(results.timePerGrid)
                final.its[i,j,k] = np.round(np.mean(results.its))
                final.rnorm[i,j,k] = np.mean(results.rnorm)
    #Plots to show here:
    #-time of convergance vs mesh size for all 5 plots
    #-time per grid point vs mesh size
    #-Error norm vs mesh size
    #-Iterations vs mesh size
    #
    pd.plotData(final)
    endhere = 1
    
#    final.experiments = results.detectors  
#    initial_guess = crp.find_mua(inputd,0.75,0.75)  #initial guess
#    Acon = np.ones(input.nx*input.ny)  #A constraint
#    bcon = 4.125*np.ones(input.nx*input.ny,1)
#    initial_guess   = reshape(initial_guess,input.nx*input.ny,1)
#    #----------------------------------------
#    #-Minimize this function here
#    #f = @(initial_guess)inverse(input,final,initial_guess)
#    [PHI,grad] = inverse(input,final,initial_guess)
#    #- [x,fval] = fminunc(f,initial_guess, optimset(...
#    #-    'MaxIter',20))
#    #- final_D.fval = fval
#    #- final_D.x = x
#    #----------------------------------------

if __name__ == '__main__':
    main()

