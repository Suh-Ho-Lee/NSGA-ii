#!/usr/bin/env python3

__all__ = ['GetModelGridPolygon']
def GetModelGridPolygon(mf,debug=0):
   '''
   Explain
    Get each grid polygon from modflow model.

   Usage

   '''
   from .grid import flopyGetXY, GetDxDy
   import shapely
   import numpy as np
   # get model grid
   xc, yc = flopyGetXY(mf,center=1)
   dx, dy = GetDxDy(mf)
   nx = len(dx)
   ny = len(dy)

   # set each grid point as polygon for searching intersection.
   polys = []
   for j, y in enumerate(yc):
       for i, x in enumerate(xc):
          p = [ [x-dx[i]/2, y-dy[j]/2],
                [x+dx[i]/2, y-dy[j]/2],
                [x+dx[i]/2, y+dy[j]/2],
                [x-dx[i]/2, y+dy[j]/2],
                [x-dx[i]/2, y-dy[j]/2]]
          polys.append(shapely.geometry.Polygon(p))

   # reshape polys
   polys = np.reshape(polys, (ny,nx))

   return polys
