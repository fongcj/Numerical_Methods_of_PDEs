# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:41:48 2012

@author: cjf2123
"""
import numpy as np
from matplotlib import pylab
from matplotlib import colors

def imageData(inputd,output):
    nx      = inputd.nx
    ny      = inputd.ny
    lx      = inputd.lx
    ly      = inputd.ly    
    source  = inputd.sourcePos
    index   = inputd.index
    v = np.linspace(np.log(output.U.min()),np.log(output.U.max()),15,endpoint=True) 
    v = np.round(v,1)
    pylab.figure()
    pylab.title('Diffuse Optical Tomography: Laser at x={0}'.format(inputd.sourcePos[index]))
    pylab.imshow(np.log(output.Usq), origin='lower', extent=[0,ly,0,lx])
    pylab.xlabel("X (cm)")
    pylab.ylabel("Y (cm)")
    clr = pylab.colorbar(ticks=v)
    clr.set_label('Light Intensity (AU)') 
    print clr
    pylab.show()
    
def plotTomo(inputd,output):
    nx      = inputd.nx
    ny      = inputd.ny
    lx      = inputd.lx
    ly      = inputd.ly    
    source  = inputd.sourcePos
    index   = inputd.index
    x = np.linspace(0,lx,nx)
    pylab.figure()
    pylab.title('Diffuse Optical Tomography: Detector with laser at x={0}'.format(inputd.sourcePos[index]))
    pylab.plot(x,np.log(output.lower))
    pylab.grid(True)
    pylab.ylim([-30,-15])
    pylab.xlabel("X (cm)")
    pylab.ylabel("Partial Current log(C*U)")
    pylab.show()


def plotData(final):
    timeVec = final.time
    timePerGridVec = final.timePerGrid
    itsVec = final.its
    rnormVec = final.rnorm   
    Nwidth = final.Nwidth
    
    pylab.figure(1)
    pylab.title('Diffuse Optical Tomography: Elapsed Time vs N')
    pylab.plot(Nwidth,timeVec[:,0,0],Nwidth,timeVec[:,1,0],Nwidth,timeVec[:,1,1],Nwidth,timeVec[:,1,2],Nwidth,timeVec[:,2,0],Nwidth,timeVec[:,2,1],Nwidth,timeVec[:,2,2])
    pylab.xscale('log')
    pylab.legend(["Sparse Direct","KSP:CG, PC:None","KSP:GMRES, PC:None","KSP:CG, PC:ILU","KSP:GMRES, PC:ILU","KSP:CG, PC:HYPRE","KSP:GMRES, PC:HYPRE"],loc="best")
    pylab.xlabel("Square Matrix Width (N)")
    pylab.ylabel("Elapsed Time (s)")
    pylab.grid()

    
    pylab.figure(2)
    pylab.title('Diffuse Optical Tomography: Elapsed Time per Grid Points vs N')
    pylab.plot(Nwidth,timePerGridVec[:,0,0],Nwidth,timePerGridVec[:,1,0],Nwidth,timePerGridVec[:,1,1],Nwidth,timePerGridVec[:,1,2],Nwidth,timePerGridVec[:,2,0],Nwidth,timePerGridVec[:,2,1],Nwidth,timePerGridVec[:,2,2])
    pylab.xscale('log')
    pylab.legend(["Sparse Direct","KSP:CG, PC:None","KSP:GMRES, PC:None","KSP:CG, PC:ILU","KSP:GMRES, PC:ILU","KSP:CG, PC:HYPRE","KSP:GMRES, PC:HYPRE"],loc="best")
    pylab.xlabel("Square Matrix Width (N)")
    pylab.ylabel("Elapsed Time/N_total (s/pixel)")
    pylab.grid()
    
    pylab.figure(3)
    pylab.title('Diffuse Optical Tomography: Number of Iterations vs N')
    pylab.plot(Nwidth,itsVec[:,0,0],Nwidth,itsVec[:,1,0],Nwidth,itsVec[:,1,1],Nwidth,itsVec[:,1,2],Nwidth,itsVec[:,2,0],Nwidth,itsVec[:,2,1],Nwidth,itsVec[:,2,2])
    pylab.xscale('log')
    pylab.legend(["Sparse Direct","KSP:CG, PC:None","KSP:GMRES, PC:None","KSP:CG, PC:ILU","KSP:GMRES, PC:ILU","KSP:CG, PC:HYPRE","KSP:GMRES, PC:HYPRE"],loc="best")
    pylab.xlabel("Square Matrix Width (N)")
    pylab.ylabel("Iterations")
    pylab.grid()

    pylab.figure(4)
    pylab.title('Diffuse Optical Tomography: Relative Norm Error vs N')
    pylab.plot(Nwidth,rnormVec[:,0,0],Nwidth,rnormVec[:,1,0],Nwidth,rnormVec[:,1,1],Nwidth,rnormVec[:,1,2],Nwidth,rnormVec[:,2,0],Nwidth,rnormVec[:,2,1],Nwidth,rnormVec[:,2,2])
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.legend(["Sparse Direct","KSP:CG, PC:None","KSP:GMRES, PC:None","KSP:CG, PC:ILU","KSP:GMRES, PC:ILU","KSP:CG, PC:HYPRE","KSP:GMRES, PC:HYPRE"],loc="best")
    pylab.xlabel("Square Matrix Width (N)")
    pylab.ylabel("Relative Norm Error")
    pylab.grid()
    pylab.show()
    
