#!/usr/bin/env python
from .GetAreas import GetAreas
import numpy as np
import flopy

def GetTotalAreas(mf):
   # check each areas.
   areas = GetAreas(mf)

   # check masking...
   ibound = mf.bas6.ibound
   areas[ibound==0] == 0

   return np.sum(areas)

if __name__ == '__main__':
   mf = flopy.modflow.Modflow.load('mf.nam',model_ws='../../Data/WNS')
   area = GetTotalAreas(mf)

   print('   total area = {} m^2'.format(area))
