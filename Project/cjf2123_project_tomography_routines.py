# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:16:48 2012

@author: cjf2123
"""
#cjf2123_project_tomography

import numpy as np
import cjf2123_project_createp as crp
import cjf2123_project_forward as fwd
import cjf2123_project_plotData as pd

"""
-----------------------------------------------------------------------------
"""
class results_struct:
    detectors = None
    U = None
    grad = None
    time = None
    timePerGrid = None
    its = None
    rnorm = None
"""
-----------------------------------------------------------------------------
"""
def tomography(inputd,final):
    nx = inputd.nx
    ny = inputd.ny
    inputd.mua = np.reshape(inputd.mua,(nx,ny))   
    
    sourcePos = inputd.sourcePos
    results = results_struct
    results.detectors = np.zeros((len(sourcePos),nx))
    results.U = np.zeros((nx*ny,len(sourcePos)))
    results.its = np.zeros(len(sourcePos))
    results.rnorm = np.zeros(len(sourcePos))
    results.time = np.zeros(len(sourcePos))
    results.timePerGrid = np.zeros(len(sourcePos))
    results.grad = 1
    for p in range(len(sourcePos)): 
        inputd.qin   = crp.find_qin(inputd,sourcePos[p],0.1)  #Define location of the source
        inputd.index = p
        output = fwd.solve_forward(inputd)
#        
        results.time[p] = output.time
        results.timePerGrid[p] = output.timePerGrid
        results.its[p] = output.its
        results.rnorm[p] = output.rnorm      
        results.detectors[p,:] = output.lower
        results.U[:,p] = output.U
        
        if inputd.plotTomo == 'y':
            pd.imageData(inputd,output)
            pd.plotTomo(inputd,output)
        #grad = gradient(inputd,output,final);
        #results.grad[p] = grad;
    return results
"""
-----------------------------------------------------------------------------
"""
def inverse(inputd,final,initial_guess):
    inputd.mua = initial_guess
    [result] = tomography(inputd,final)
    final.mua = inputd.mua
    [PHI] = objective_function(result.detectors,final.experiment)
    grad = result.grad
    return(PHI,grad)
"""
-----------------------------------------------------------------------------
"""    
def objective_function(result,experiment):
    PHI = sum(sum((result-experiment)**2))
    return(PHI)

"""
-----------------------------------------------------------------------------
"""
def gradient(inputd,output,final):
    sources = inputd.sourcePos
    n       = inputd.n
    lx      = inputd.lx
    ly      = inputd.ly
    nx      = inputd.nx
    ny      = inputd.ny
    dx      = inputd.dx
    dy      = inputd.dy
    x       = inputd.x
    y       = inputd.y
    D       = inputd.D
    mua     = inputd.mua
    p = inputd.index
    #Uses adjoint difference to compute the gradient for the inverse problem
    u = reshape(output.U,output.U.size,1)
    A = output.M
    Meas = final.experiments[:,p]
    grad = 1
    #A*lambda = -1/M^2*Q'*(M-Q*u), where u is the solution given inputd ua, Q is
    #a matrix that produces the same measurements locations as Meas
    Q = zeros((nx,nx*ny))
    for i in range(nx):
        j = (i)*nx+1
        Q[i,j] = 1    
#    lambda1 = A\(-1/(sum(Meas.^2))*Q'*(Meas-Q*u))
    lambda1 = 1
    delf = lambda1*u
    grad = delf
    return(grad)