# -*- coding: utf-8 -*-
"""
Simple Numpy + pylab script to demonstrate simple matlab like commands
Created on Tue Sep 11 22:34:41 2012

@author: mspieg
"""

#import the modules numpy and pylab keeping their namespaces
from numpy import *
from pylab import *

x = linspace(0,1,100)
y = sin(4*pi*x)

figure()
plot(x,y)
title('Simple plot')
xlabel('x')
ylabel('f(x)')
grid()
show()


