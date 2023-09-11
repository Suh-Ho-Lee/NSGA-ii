#!/usr/bin/env python
import numpy as np
import os, sys, platform
from .. import geometry

def ExportGeotiff(mf,data,dx,dy,fname,epsg=None,verbose=0): # {{{
   '''
   Explain
    Export given data with geotiff.

   Usage
    flopyUtils.ExportGeotiff(mf,data,dx,dy,fname):

   See also...
   test_export.py
   '''

   try:
      import rasterio
   except ImportError:
      print('ERROR: we cannot import rasterio library.')
   import warnings

   # import grid..
   x, y = geometry.GetXy(mf)

   # flip y values...
   y = np.flip(y)

   # check ascending order in y values.
   if np.any(np.diff(y) < 0):
      raise Exception('ERROR: y value should be ascending..')

   # transform input data to grid.
   data = np.flipud(data).T

   # set xy grid for geotiff
   nx = int(abs(x[-1] - x[0])/dx)
   ny = int(abs(y[-1] - y[0])/dy)

   # interpolate to new grid....
   import scipy
   interp = scipy.interpolate.RegularGridInterpolator((x,y),data)

   xi = np.linspace(x[0], x[-1], nx)
   yi = np.linspace(y[0], y[-1], ny)
   xg, yg = np.meshgrid(xi,yi)
   if verbose:
      print('check shape of interpolation')
      print('   xmin, xmax = %f, %f'%(x[0], x[-1]))
      print('   ymin, ymax = %f, %f'%(y[0], y[-1]))
      print('   (nx,ny) = (%d,%d)'%(nx,ny))
      print('   xg = {} / yg = {}'.format(np.shape(xg), np.shape(yg)))
   data_new = interp((xg.ravel(), yg.ravel()))
   data_new = np.reshape(data_new,np.shape(xg))

   from rasterio.transform import Affine
   transform = Affine.translation(x[0] - dx/2, y[0] - dy/2) * Affine.scale(dx, dy)

   s = np.shape(data_new) # size of array.
   dtype = data_new.dtype

   # check epsge of modflow..
   if not mf.modelgrid._epsg:
      warnings.warn('WARN: we cannot find any EPSG at mf.modelgrid._epsg...')
   if (not mf.modelgrid._epsg) & (not epsg):
      raise Exception('ERROR: check epsg of input model.')
   if epsg: # update model grid epsg.
      mf.modelgrid._epsg = epsg

   # check mf field.
   with rasterio.open(fname,'w',
       driver='GTiff',
       height=s[0],
       width=s[1],
       count=1,
       dtype=dtype,
       crs=mf.modelgrid._epsg,
       transform=transform,) as fid:
      fid.write(data_new, 1)
   # }}}
