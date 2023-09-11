#!/usr/bin/env python
import numpy as np
import flopy

__all__ = ['LoadInitModflow']
def LoadInitModflow(**kwargs):
   '''
   Explain
    get pre-scribed model.

   Usage
    mf = flopyUtils.test.LoadInitModflow()

   Ref
    https://flopy.readthedocs.io/en/latest/Notebooks/mf_tutorial01.html
   '''
   # initialize model
   mf = flopy.modflow.Modflow('mf',exe_name='mf2005',model_ws='./Models/Test')

   # initialize grid
   Lx   = 1e+3
   Ly   = 1e+3/2
   ztop = 0.
   zbot = -50.
   nlay = 1
   nrow = 50
   ncol = 25
   delr = Lx/ncol
   delc = Ly/nrow
   delv = (ztop-zbot)/nlay
   botm = np.linspace(ztop,zbot,nlay+1)

   # initialize grid.
   dis = flopy.modflow.ModflowDis(
         mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm[1:],
         )

   return mf

