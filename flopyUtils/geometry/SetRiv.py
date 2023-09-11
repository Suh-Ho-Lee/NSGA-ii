#!/usr/bin/env python

def SetRiv(mf,rstage,rbotm,hk_riv,verbose=False,
      shpname=None):
   '''
   Explain
    Set river stage and botm elevation from give data values.

   Inputs
   '''
   import flopy
   import numpy as np
   from ..utils.print import print_

   # check rstage and rbotm
   if rstage < rbotm:
      raise Exception('ERROR: relative stage (=%f) is lower than relative botm elevation(=%f)'%(rstage,rbotm))

   if not shpname:
      top = mf.dis.top.array
      botm = mf.dis.botm.array

      delr = mf.dis.delr.array
      delc = mf.dis.delc.array
      riv = mf.riv.stress_period_data[0]

      print_(mf,debug=verbose)
      print_('   size of delr = {}'.format(np.shape(delr)),debug=verbose)
      print_('   size of delc = {}'.format(np.shape(delc)),debug=verbose)
      print_('{}'.format(riv.dtype),debug=verbose)

      lay, row, col = riv['k']-1, riv['i']-1, riv['j']-1
   else:
      raise Exception('ERROR: SetRiv package is not available with give shpname option.')

   # initialize stage value.
   _stage = np.zeros((len(lay),),dtype=float)
   _botm  = np.zeros((len(lay),),dtype=float)
   for i, r, c in zip(range(len(row)),row,col):
      _stage[i] = top[r,c] + rstage
      _botm[i]  = top[r,c] + rbotm

      # update layer information.
      for l in range(mf.nlay):
         if l == 0:
            __top  = top
            __botm = botm[0,:,:]
         else:
            __top  = botm[l-1,:,:]
            __botm = botm[l,:,:]
         if (__botm[r,c] < _botm[i]) & ( _botm[i] < __top[r,c]):
            lay[i] = l

   # initialize conductance value
   conduct = np.zeros((len(lay),),dtype=float)
   for i in range(len(conduct)):
      conduct[i] = delr[col[i]]*delc[row[i]]*hk_riv

   # assign each value
   riv['k']  = lay
   riv['cond'] = conduct
   riv['stage'] = _stage
   riv['rbot'] = _botm

   # set river package.
   flopy.modflow.ModflowRiv(mf,stress_period_data={0:riv})
   return mf

if __name__ == '__main__':
   import flopy
   mf = flopy.modflow.Modflow.load('mf.nam',model_ws='../../../../Model/WNSdefault/')

   mf = SetRiv(mf,-1,-2,.1)
   mf.riv.check()
