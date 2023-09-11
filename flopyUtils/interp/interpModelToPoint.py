#!/usr/bin/env python
from .interpGridToPoint import interpGridToPoint
from ..geometry.GetXy import GetXy
import numpy as np

def interpModelToPoint(mf,data,xp,yp):
   '''
   Explain
    interpolate grid from modflow to point values.

   Usage
    top = mf.dis.top.array
    out = interpModelToPoint(mf,top,xp,yp)
   '''

   XG,YG = GetXy(mf,center=1)
   YG  = np.flipud(YG)
   data = np.flipud(data).T

   return interpGridToPoint(XG,YG,data,xp,yp)
