#!/usr/bin/env python
import flopy, json
import numpy as np

def PathlineToJson(mf,fname,oname,epsg=None):
   '''
   Explain
    convert path line file to json format.

   Options
    mf    - modflow structure from flopy.
    epsg  - geocode information. If not defined, epsg is set as None value.
    fname - input "mppth" file. 
    oname - output json format file.
   '''
   # check input format.
   if not isinstance(mf,flopy.modflow.mf.Modflow):
      raise Exception('ERROR: input "mf"(={}) variable is not "flopy.modflow.mf.Modflow"'.format(type(mf)))

   # get xy offset.
   xoffset = mf.modelgrid.xoffset
   yoffset = mf.modelgrid.yoffset

   pobj = flopy.utils.PathlineFile(fname)
   data = {} # structure for json
   # initialize EPSG
   data['epsg'] = None
   if epsg:
      data['epsg'] = epsg
   data['maxid'] = pobj.get_maxid().tolist()
   for idx, d in enumerate(pobj.get_alldata()):
      print('   processing = %d/%d'%(idx+1,pobj.get_maxid()+1),end='\r')
      data[idx] = {}
      data[idx]['particleid'] = idx
      data[idx]['x'] = np.array(xoffset+d['x']).tolist()
      data[idx]['y'] = np.array(yoffset+d['y']).tolist()
      data[idx]['z'] = d['z'].tolist()
      data[idx]['k'] = d['k'].tolist()
      data[idx]['time'] = d['time'].tolist()
   print('')

   print('======== save pathline file to json format =========')
   with open(oname,'w') as fid:
      json.dump(data,fid,indent=3)
