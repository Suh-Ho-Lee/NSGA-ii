#!/usr/bin/env python

def interpGridToPoint(xg,yg,data,xp,yp):
   '''
   Explain
    interpolate grid data to point value.

   Usage
    out = interpGridToPoint(xg,yg,data,xp,yp)
   '''
   import numpy as np
   import pandas, scipy
   import os 

   # check input format.
   s1 = len(xg)
   s2 = len(yg)
   s  = np.shape(data)
   if (s1 != s[0]) | (s2 != s[1]): #np.shape(data) != np.array((s1,s2)):
      raise Exception('ERROR: dimension of data is {}. len(xg), len(yg) = ({},{})',
            np.shape(data),len(xg),len(yg))

   # prepare interpolation.
   if isinstance(xp,float) & isinstance(yp,float):
      xp = np.array([xp])
      yp = np.array([yp])

   if 1:
      #out = interp(xp,yp)
      out = scipy.interpolate.interpn((xg,yg),data,(xp,yp))
   else:
      interp = scipy.interpolate.RectBivariateSpline(xg,yg,data)
      out = np.zeros((len(xp),))
      for i, _x, _y in zip(range(len(xp)),xp,yp):
         out[i] = interp(_x,_y)

   return out
