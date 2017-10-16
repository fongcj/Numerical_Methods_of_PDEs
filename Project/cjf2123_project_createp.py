# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 01:17:33 2012

@author: cjf2123
"""
import numpy as np 

def find_mua(inputd,mua1,mua2):
    nx = inputd.nx
    ny = inputd.ny
    v  = np.zeros((ny,nx))
    for i in range(ny):
        for j in range(nx):
            if j >= nx/2:
                v[i,j] = mua1
            else:
                v[i,j] = mua2
    return(v)


def find_qin(inputd,qx,width):
    nx = inputd.nx
    ny = inputd.ny
    dx = inputd.dx
    v  = np.zeros((ny,nx))
    y  = inputd.y
    indexX = round(qx/dx)
    if indexX == 0:
        indexX = 1
        width = dx
    indexY = len(y)
    widthIndex = round(width/dx)
    midWidth = np.floor(widthIndex/2)
    if midWidth == 0:
        v[indexY-1][indexX-1] = 1
    else:
        v[indexY-1][indexX-midWidth-1:indexX+midWidth-1] = 1
        
    return(v)