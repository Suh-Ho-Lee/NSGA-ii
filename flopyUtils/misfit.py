#!/usr/bin/env python
import numpy as np
import pandas
import os, sys, platform

def HobMisfit(mf):
   '''
   Explain
    calculate misfit between modelled and observed head.

   Usage
    misfit, rmse = HobMisfit(mf)
   '''
   from utils.ReadHobOut import ReadHobOut

   # check flopy model.
   print(mf.get_name_file_entries())
   if mf.has_package('hob'):
      print('this modflow has "hob"...')
      fname = os.path.join(mf.model_ws, mf.name+'.hob.out')
      data = ReadHobOut(fname)
      mod  = data['mod']
      obs  = data['obs']
   else:
       
   return misfit

if __name__ == '__main__':
   import flopy
   mf = flopy.modflow.Modflow.load('mf.nam',model_ws='../../../Models/pest-ies_result')

   HobMisfit(mf)
