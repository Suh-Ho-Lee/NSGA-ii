#!/usr/bin/env python3
import os

def interpFromGridToGrid(xg,yg,zval,XG,YG,method='linear',debug=False,
      ncpus=1):# {{{
   '''
   Explain
    interpFromGridToGrid

   Inputs
    xg, yg : input grid information
    zval   : interp z values
    XG,YG  : output grid

   Options
    method   - interpolation method (default: linear). linear, nearest are available.
    np       - number of process for multiprocessing.
   '''
   import numpy as np
   import scipy.interpolate

   if debug:
      print('number of core: %d'%(ncpus))
      print('initialize interpolater.')
   interp = scipy.interpolate.RegularGridInterpolator((xg,yg),zval,method=method)
   #if method == 'linear':
   #   interp = scipy.interpolate.RectBivariateSpline(xg,yg,zval)
   #elif method == 'nearest':
   #else:
   #   raise Exception('ERROR: we cannot interpolate given data with %s method. "linear", "nearest" are available.'%(method))

   try:
      if len(np.shape(XG)) == 1:
         XG,YG = np.meshgrid(XG,YG)
      nx, ny = np.shape(XG)

      xmin = np.amin(xg)
      xmax = np.amax(xg)
      ymin = np.amin(yg)
      ymax = np.amax(yg)

      if ncpus > 1: # multiprocessing.
         import parmap
         XG = XG.ravel()
         YG = YG.ravel()

         # find xy in boundary.
         pos = (xmin<=XG) & (XG<=xmax) & (ymin<=YG) & (YG<=xmax)

         xy = [(_x,_y) for _x,_y in zip(XG[pos],YG[pos])]

         if debug:
            print(' do interpolation.')
         data = parmap.map(interp, xy, pm_pbar=True, pm_processes=ncpus)
         # reshape to 2d grid
         data = np.reshape(data,(nx,ny))
      else: # use single cpus.
         print('use single cpu for interpolation.')
         # find xy in boundary.
         pos = (xmin<=XG) & (XG<=xmax) & (ymin<=YG) & (YG<=ymax)

         data = np.empty(np.shape(XG))
         data[pos] = interp((XG[pos],YG[pos]))
      return data
   except:
      raise Exception('ERROR: we cannot interpolate given data set..')
   # }}}
