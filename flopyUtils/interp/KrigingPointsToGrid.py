#!/usr/bin/env python

# KrigingPointsToGrid # {{{
def KrigingPointsToGrid(x,y,zval,xg,yg,
      method='universal',
      variogram_model='spherical',
      variogram_parameters=None,
      enable_plotting=None,
      verbose=0,
      ):
   import pykrige
   import numpy as np

   # check method
   if not (method in ['ordinary','universal']):
      raise Exception('Kriging method(%s) are ordinar and universal.'%(method))

   # check input grid information.
   nx = len(xg)
   ny = len(yg)
   print('(nx,ny) = ({},{})'.format(nx,ny))

   # make ordinary kriging
   if method == 'ordinary':
      kriging = pykrige.ok.OrdinaryKriging(x,y,zval,variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            verbose=False,enable_plotting=enable_plotting)
   elif method == 'universal':
      kriging = pykrige.uk.UniversalKriging(x,y,zval,variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            verbose=False,enable_plotting=enable_plotting)

   # interpolateion
   [data_interp, ss] = kriging.execute('grid',xg,yg)

   return data_interp.T
   # }}}
