#!/usr/bin/env python
import numpy as np
import os
import pandas

def ReadHobOut(fname): # {{{

   # check file exists
   if not os.path.isfile(fname):
      raise Exception("ERROR: we cannot find %s"%(fname))

   obs  = []
   mod  = []
   name = []
   with open(fname,'r') as fid:
      line = fid.readline() # skip first row
      while line:
         line = fid.readline().replace('\n','').split(' ')
         line = list(filter(None,line))
         #print(line)
         if not line:
            break
         mod.append(float(line[0]))
         obs.append(float(line[1]))
         name.append(line[-1])

   obs = np.array(obs)
   mod = np.array(mod)
   name = np.array(name,dtype=object)

   return pandas.DataFrame({'obs':obs,'mod':mod, 'name':name})
   # }}}

