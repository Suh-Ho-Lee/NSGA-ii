#!/usr/bin/env python
import numpy as np
import os, sys

def interpPointToGrid(x,y,z,xi,yi,method='IDW', # {{{
      options=None,verbose=False):
   '''
   Explain
    inteprolate Point To Grid.

   Options
    method      - method for interpolation
                 IDW    - inverse distance weighting method.

    options     - define parameters for each method.

   Reference

   IDW code is from 
    https://gist.github.com/Majramos/5e8985adc467b80cccb0cc22d140634e
   '''
   if len(x) != len(y):
      raise Exception('ERROR: length of x and y are not same.')
   if len(x) != len(z):
      raise Exception('ERROR: length of x and z are not same.')

   if method.lower()=='idw':
      if not options:
         power = 1
      else:
         power = options['power']

      if verbose:
         print('   2D to 1D column.')

      size = np.shape(xi)
      # 2D to 1D column...
      xi = np.ravel(xi)
      yi = np.ravel(yi)

      # do interpolation
      out = simple_idw(x,y,z,xi,yi,power=power)
      
      return np.reshape(out,size)
   else:
      raise Exception('ERROR: %s is not supported yet.'%(method))
   # }}}

# IDW fucntions.
def distance_matrix(x0, y0, x1, y1): # {{{
    """ Make a distance matrix between pairwise observations.
    Note: from <http://stackoverflow.com/questions/1871536>
    """

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    # calculate hypotenuse
    return np.hypot(d0, d1)
# }}}
def simple_idw(x, y, z, xi, yi, power=1): # {{{
   """ Simple inverse distance weighted (IDW) interpolation
   Weights are proportional to the inverse of the distance, so as the distance
   increases, the weights decrease rapidly.
   The rate at which the weights decrease is dependent on the value of power.
   As power increases, the weights for distant points decrease rapidly.
   """

   dist = distance_matrix(x,y, xi,yi)

   # In IDW, weights are 1 / distance
   weights = 1.0/(dist+1e-12)**power

   # Make weights sum to one
   weights /= weights.sum(axis=0)

   # Multiply the weights for each interpolated point by all observed Z-values
   return np.dot(weights.T, z)
# }}}

if __name__ == '__main__':
   print('   test interpPointToGrid with IDW')
