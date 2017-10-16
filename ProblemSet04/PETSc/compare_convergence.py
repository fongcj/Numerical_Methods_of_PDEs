# -*- coding: utf-8 -*-
"""
CompareConvergence
Created on Mon Oct 15 22:52:45 2012

quicky script to read and plot convergence behavior for difference petsc runs
output from petsc_poisson2d.py expected in a npz file with keys

['npts', 'time', 'name', 'rhist']

@author: mspieg
"""

import numpy as np
import pylab
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages

def plotfiles(files):
    legend=[]
    fig = pylab.figure()
    for file in files:
        s=np.load(file)
        rh = s['rhist']
        pylab.semilogy(range(len(rh)),rh/rh[0])
        legend += [str(s['name'])]
        pylab.hold(True)
    
    pylab.xlabel('# of iterations')
    pylab.ylabel('||r||_2/||r_0||_2')
    pylab.grid()
    pylab.title('Convergence comparison {0}x{1}'.format(s['npts'][0],s['npts'][1]))    
    print legend
    pylab.legend(legend,loc='best')
    return fig

def makeplots():
    # set problem size for comparison
    N = 65

    # read make plots by different ksp's
    plot1 = plotfiles(glob('*richardson*.npz'))
    plot2 = plotfiles(glob('*cg*.npz'))
    plot3 = plotfiles(glob('*gmres*.npz'))
    plot4 = plotfiles(glob('*optimal*.npz')+glob('*cg*.npz')+glob('*gmres*.npz'))
    
    # save to pdf
    pp = PdfPages('ConvergenceComparison.pdf')
    pp.savefig(plot1)
    pp.savefig(plot2)
    pp.savefig(plot3)
    pp.savefig(plot4)
    pp.close()
    pylab.show()

if __name__ == '__main__':
    makeplots()

    
