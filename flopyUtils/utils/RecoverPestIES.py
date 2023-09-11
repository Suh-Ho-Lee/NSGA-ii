#!/usr/bin/env python


def RecoverPestIES(model_base,model_ws,pest_name='pesties',mf_name='mf'):
   import numpy as np
   import pandas, re, glob, pyemu
   import os, sys, platform

   # initialize file directory.
   print('generate %s'%(model_ws))
   if os.path.isdir(model_ws):
      os.system('rm -rf %s'%(model_ws))
   # copy directory....
   os.system('cp -r %s %s'%(model_base, model_ws))

   # search last result.
   regex = re.compile(r'%s/%s.(?P<ne>\d+).par.csv'%(model_base,pest_name))
   print(regex)
   lists = np.sort(glob.glob('%s/%s.[!regected]*.par.csv'%(model_base,pest_name)))
   ne = []
   for l in lists:
      ne.append(regex.search(l).group('ne'))
   pos = np.argmax(np.array(ne,dtype=int))
   fname = lists[pos]
   print(fname)

   # load pest.
   pst = pyemu.Pst(os.path.join(model_base,'pesties.pst'))

   # load final result.
   result = pandas.read_csv(fname,index_col=0)
   pe = pyemu.ParameterEnsemble(pst,result)

   # update estimated parameter values.
   parval1 = pe.loc['base',:].to_numpy()
   pst.parameter_data.loc[:,'parval1'] = parval1
   pst.control_data.noptmax = 0

   # build new pst data.
   pst.write(os.path.join(model_ws,'base_N.pst'))

   # generate ensemble results.
   pyemu.os_utils.run("pestpp-ies base_N.pst",cwd=model_ws)

if __name__ == '__main__':
   import os
   RecoverPestIES(model_base='../../../../Models/pest-ies_master',
         model_ws='../../../../Models/pest-ies_result')
