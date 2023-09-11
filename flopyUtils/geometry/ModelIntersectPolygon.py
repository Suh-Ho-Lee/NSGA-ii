#!/usr/bin/env python

__all__ = ['ModelIntersectPolygon']

def INTERSECTION(poly, poly_in):
   #print(type(poly),type(poly_in))
   if poly.intersection(poly_in).area > 0:
      return 1
   else:
      return 0

def ModelIntersectPolygon(mf,poly,verbose=1):
   '''
   Explain
    Get intersection between polygon and model.

   Usage

    import flopy
    mf = flopy.modflow.Modflow('mf',model_ws='Models/Temp',version='mf2005',exe_name='mf2005')
    pos = ModelIntersectPolygon(mf,poly)
   '''
   import numpy as np
   import multiprocessing
   import geopandas, shapely
   from .grid import flopyGetXY

   # check input polygon...
   if isinstance(poly,geopandas.geodataframe.GeoDataFrame):
      pos = np.zeros((len(poly),),dtype=bool)
      # check each geometry in polygon
      for i,d in enumerate(poly.geometry):
         if isinstance(d,shapely.geometry.polygon.Polygon):
            pos[i] = True
      poly = poly.geometry[pos]
      #print(type(poly))
   elif isinstance(poly,shapely.geometry.polygon.Polygon):
      poly = [poly]
   else:
      raise Exception('ERROR: current type of polygon({}) is not supported'.format(type(poly)))

   # get model grid
   xg, yg = flopyGetXY(mf)
   nx = len(xg)
   ny = len(yg)

   # get model grid information.
   dx = np.diff(xg)
   dy = np.diff(yg)
   dx = np.concatenate(([dx[0]],dx,[dx[-1]]))/2
   dy = np.concatenate(([dy[0]],dy,[dy[-1]]))/2

   # set each grid point as polygon for searching intersection.
   polys = []
   for j, y in enumerate(yg):
       for i, x in enumerate(xg):
          p = [[x-dx[i], y-dy[j]],
                [x+dx[i+1],y-dy[j]],
                [x+dx[i+1],y+dy[j+1]],
                [x-dx[i],y+dy[j+1]],
                [x-dx[i], y-dy[j]]]
          polys.append(shapely.geometry.Polygon(p))

   # searching polygon intersections...
   mask = np.zeros((ny,nx),dtype=bool)
   for geometry in poly:
      print('Masking...')
      with multiprocessing.Pool()as pool:
         #print('do multi processing...')
         pos = pool.starmap(INTERSECTION,[(geometry, p) for p in polys])
      pos = (np.reshape(pos,(ny,nx))>0)
      mask[pos] = True

   return mask
