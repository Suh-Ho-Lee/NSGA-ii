#!/usr/bin/env python

def SetChd(mf,shpname,shead,ehead=None,verbose=0,epsg:int=None, # {{{
      istype=0):
   '''
   Explain
    set specific head distribution depending on boundary.

   Inputs
    epsg   - EPSG system for shape file.
    istype - (str of list) how to determine the starting head along boundary condition
             'constant' - constant head
             'subtract' - shead = (top - DTW) values.

   '''
   import numpy as np
   import geopandas, os, flopy
   from .find_bc import find_bc
   from .grid import getXyGrid
   from ..utils.print import print_
   from .PointsInPolygon import PointsInPolygon

   if isinstance(shpname,str):
      shpname = [shpname]
   if isinstance(shead,float) | isinstance(shead,int):
      shead = [shead]
   if ehead:
      if isinstance(ehead,float) | isinstance(ehead,int):
         ehead = [ehead]
   else:
      ehead = shead

   if isinstance(istype,str):
      istype = [istype]

   # check type for setting CHD
   for _type in istype:
      if not _type in ['constant','subtract']:
         raise Exception('ERROR: SetCHD: current type "{}" is not supported. Assign constant or subtract string in type.'%{_type})

   if verbose:
      print('SetCHD:')
      print('length of shpname = {}'.format(np.shape(shpname)))
      print('length of shead   = {}'.format(np.shape(shead)))
      print('length of ehead   = {}'.format(np.shape(ehead)))

   # first remove chd package
   if 'CHD' in mf.get_package_list():
      mf.remove_package('CHD')

   # get top elevation
   top = mf.dis.top.array

   # get boundary
   ibc = find_bc(mf)
   xg,yg = getXyGrid(mf,center=1)

   data = []
   for _shp, _shead, _ehead, _istype in zip(shpname, shead, ehead, istype):
      print_('SetChd: load {}'.format(_shp),debug=verbose)

      if not os.path.isfile(_shp):
         raise Exception('ERROR: we cannot find %s'%(_shp))

      # get points in polygon
      if epsg:  # change coordinate system.
         _shp = geopandas.read_file(_shp).to_crs('epsg:%d'%(epsg))

      ibc_chd = (PointsInPolygon(_shp,xg,yg) >= 1) & (ibc >= 1)
      pos = np.where(ibc_chd)
      for r,c in zip(pos[0],pos[1]):
         if _istype=='constant':
            data.append([mf.nlay-1,r,c,_shead,_ehead])
         elif _istype=='substract':
            data.append([mf.nlay-1,r,c,top[r,c]-_shead,top[r,c]-_ehead])

   print_('SetChd: set head boundary...',debug=verbose)
   flopy.modflow.ModflowChd(mf,stress_period_data={0:data})

   return mf
   # }}}
