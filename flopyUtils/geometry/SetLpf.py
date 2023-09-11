#!/usr/bin/env python

def SetLpf(mf,shpname,hk,ss,sy,verbose=0,epsg:int=None, # {{{
      ):
   '''
   Explain
    set specific hydraulic property distribution depending on shapefile.

   Inputs

   '''
   import numpy as np
   import geopandas, os, flopy
   from .find_bc import find_bc
   from .grid import getXyGrid
   from ..utils.print import print_
   from .PointsInPolygon import PointsInPolygon

   if isinstance(shpname,str):
      shpname = [shpname]
   if isinstance(hk,float) | isinstance(hk,int):
      hk = [hk]
   if isinstance(ss,float) | isinstance(ss,int):
      ss = [ss]
   if isinstance(sy,float) | isinstance(sy,int):
      sy = [sy]

   if verbose:
      print('SetLPF:')
      print('length of shpname = {}'.format(np.shape(shpname)))
      print('length of hk      = {}'.format(np.shape(hk)))
      print('length of ss      = {}'.format(np.shape(ss)))
      print('length of sy      = {}'.format(np.shape(sy)))

   # get model grid... 
   xg,yg = getXyGrid(mf,center=1)

   # get material properties.
   HK = mf.lpf.hk.array
   SS = mf.lpf.ss.array
   SY = mf.lpf.sy.array

   data = []
   for _shp, _hk, _ss, _sy in zip(shpname, hk, ss, sy):
      print_('SetProperty: load {}'.format(_shp),debug=verbose)

      if not os.path.isfile(_shp):
         raise Exception('ERROR: we cannot find %s'%(_shp))

      # get points in polygon
      if epsg:  # change coordinate system.
         print_('SetLPF: change coordinate system to EPSG')
         _shp = geopandas.read_file(_shp).to_crs('epsg:%d'%(epsg))

      pos = np.where((PointsInPolygon(_shp,xg,yg)>0))
      print(pos)
      for l in range(mf.nlay):
         HK[l,pos[0],pos[1]] = _hk
         SS[l,pos[0],pos[1]] = _ss
         SY[l,pos[0],pos[1]] = _sy

   print_('SetLpf: set material properties...',debug=verbose)
   mf.lpf.hk = HK
   mf.lpf.ss = SS
   mf.lpf.sy = SY

   return mf

