#!/usr/bin/env python3

__all__ = ['ModelIntersectLine']
def ModelIntersectLine(mf,lines): # {{{
   '''
   Explain
    Find intersection of modflow grid with lines.
   '''
   from .GetModelGridPolygon import GetModelGridPolygon
   import shapely
   import numpy as np

   # get number of grid.
   ny, nx = mf.dis.nrow, mf.dis.ncol

   mask   = np.zeros((ny,nx))
   gpolys = GetModelGridPolygon(mf)

   # check inputs
   if isinstance(lines,shapely.geometry.LineString):
      lines = [lines]

   for line in lines:
      pos = line.intersects(gpolys)
      mask[pos] = 1

   return mask
   # }}}

