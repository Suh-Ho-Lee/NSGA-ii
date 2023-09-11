#!/usr/bin/env python
import numpy as np

def get_oc2time(mf):
   '''
   Explain
    get save data time from oc file.

   Usage
    time = get_oc2time(mf)
   '''
   totim  = []
   nper   = mf.dis.nper
   nstp   = mf.dis.nstp.array
   perlen = mf.dis.perlen.array
   tsmult = mf.dis.tsmult.array
   t = 0.0

   # get oc key...
   oc_key = np.array(list(mf.oc.stress_period_data.keys()))

   for kper in range(nper):
       m = tsmult[kper]
       p = float(nstp[kper])
       dt = perlen[kper]
       if m > 1:
           dt *= (m - 1.0) / (m**p - 1.0)
       else:
           dt = dt / p
       for kstp in range(nstp[kper]):
           t += dt
           if np.any(np.sum(np.array(oc_key) == (kper,kstp),axis=1)==2):
              totim.append(t)
           if m > 1:
               dt *= m
   totim = np.array(totim, dtype=float)

   return totim

