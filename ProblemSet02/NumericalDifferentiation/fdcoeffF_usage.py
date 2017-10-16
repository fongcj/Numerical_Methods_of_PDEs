from fdcoeffF import *

print "Coefficients for second derivative at x=0, stencil [-1,0,1]"
print fdcoeffF(2,0,[-1,0,1])

print "Coefficients for fourth derivative at x=0, stencil [-2,-1,0,1,2]"
print fdcoeffF(4,0,[-2,-1,0,1,2])
