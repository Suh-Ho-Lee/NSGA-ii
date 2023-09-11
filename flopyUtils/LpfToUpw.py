#!/usr/bin/env python

def LpfToUpw(mf,verbose=0):
   import flopy, warnings
   from .utils.print import print_
   '''
   Explain
    change groundwater flow package from LPF to UPW.

   Usage
    mf = LpfToUpw(mf)
   '''
   print_('   change LPF > UPW package for preventing dry cell..',debug=verbose)

   if mf.has_package('upw'):
      warnings.warn('WARN: current modflow contains UPW package.')
      return mf

   # LPF > UPW
   upw = flopy.modflow.ModflowUpw(mf,
            hk=mf.lpf.hk,vka=mf.lpf.vka,ss=mf.lpf.ss,sy=mf.lpf.sy,chani=mf.lpf.chani,
            laytyp=mf.lpf.laytyp,
                        )

   # use different solver for NWT.
   nwt = flopy.modflow.ModflowNwt(mf)

   print_('   remove lpf, pcg package.',debug=verbose)
   mf.remove_package('lpf')
   mf.remove_package('pcg')

   # return value.
   return mf
