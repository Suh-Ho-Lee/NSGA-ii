#!/usr/bin/env python

def GetAreas(mf):
   '''
   Explain
    get area of given modoels.

   Usage
    areas = flopyUtils.GetAreas(mf)
   '''
   import flopy
   import numpy as np

   dy = mf.dis.delc.array
   dx = mf.dis.delr.array
   dx,dy = np.meshgrid(dx,dy)
   areas=dx*dy
   return areas

if __name__ == '__main__':
   import flopy
   import numpy as np
   mf = flopy.modflow.Modflow.load('mf.nam',model_ws='../../Data/WNS')
   areas = GetAreas(mf) 

   # check shape of areas
   print('nrow = {}/ncol = {}'.format(mf.nrow, mf.ncol))
   print(np.shape(areas))
