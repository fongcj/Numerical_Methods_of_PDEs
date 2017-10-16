# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:12:56 2012

@author: cjf2123
"""
#Chris Fong
#APMA 4301 
#Homework 2a Problem 1
import scipy
import sympi

from fdcoeffV import fdcoeffV


k = [1,2];
h = 1;
xbar = [0*h,0.5*h,1*h]
x = [0*h,1*h,2*h]
c = scipy.zeros((len(k)*len(xbar),len(x)))
for i in range(len(k)):
    for j in range(len(xbar)):
        c[(i)*len(xbar)+(j),:] = fdcoeffV(k[i],xbar[j],x)
        if i == 0:
            print 'First derivative, xbar = %2.1f: ' %xbar[j] ,str(c[(i)*len(xbar)+(j),:]),'\r'
        else:
            print 'Second derivative, xbar = %2.1f: '