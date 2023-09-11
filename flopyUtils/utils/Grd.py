#!/usr/bin/env python
'''
Explain
 read surfer type grd file format.

Reference
 https://surferhelp.goldensoftware.com/topics/ascii_grid_file_format.htm 
'''
def ReadGrd(fname,verbose=0): # {{{
   import numpy as np
   import os
   if not os.path.isfile(fname):
      raise Exception('ERROR: %s does not exist.'%(fname))

   if verbose:
      print('ReadGrd: load %s.'%(fname))

   with open(fname,'r') as fid:
      # read line
      ID  = fid.readline().strip()
      tmp = fid.readline().strip().split()
      nx = int(tmp[0])
      ny = int(tmp[1])
      if verbose:
         print('ID         : {}'.format(ID))
         print('(nx,ny)    : (%d,%d)'%(nx,ny))

      # get xlim
      tmp = fid.readline().strip().split()
      xlo = float(tmp[0])
      xhi = float(tmp[1])

      # get ylim
      tmp = fid.readline().strip().split()
      ylo = float(tmp[0])
      yhi = float(tmp[1])

      # get zlim
      tmp = fid.readline().strip().split()
      zlo = float(tmp[0])
      zhi = float(tmp[1])
      if verbose:
         print('xmin/xmax  : {}/{}'.format(xlo,xhi))
         print('ymin/ymax  : {}/{}'.format(ylo,yhi))
         print('zmin/zmax  : {}/{}'.format(zlo,zhi))


      tmp = fid.readlines()
      zval = []
      for l in tmp:
         tmp = l.strip().split()
         if tmp:
           zval.extend(tmp)
      zval = np.array(zval,dtype=float)
      zval = np.reshape(zval,(ny,nx))
      xg   = np.linspace(xlo,xhi,nx)
      yg   = np.linspace(ylo,yhi,ny)

   if not ID == 'DSAA':
      raise Exception('ERROR; current %s ID is %s. "DSAA" is supported.'%(fname, ID))

   if verbose:
      print('data shape : {}'.format(np.shape(zval)))

   return {'id':ID,'xmin':xlo, 'xmax':xhi,
         'xg':xg,'yg':yg,
         'ymin':ylo, 'ymax':yhi,
         'zmin':zlo, 'zmax':zhi,
         'zval':zval}
   # }}}
def WriteGrd(fname,xg,yg,zval,verbose=0,grd_format='DSAA'): # {{{
   import numpy as np

   # initialize grid information.
   nx = len(xg)
   ny = len(yg)
   xmin = np.amin(xg)
   xmax = np.amax(xg)
   ymin = np.amin(yg)
   ymax = np.amax(yg)
   zmin = np.amin(zval)
   zmax = np.amax(zval)

   # open file for writing data...
   fid = open(fname,'w')

   fid.write('%s\n'%(grd_format))
   fid.write('%d %d\n'%(nx,ny))
   fid.write('%f %f\n'%(xmin,xmax))
   fid.write('%f %f\n'%(ymin,ymax))
   fid.write('%f %f\n'%(zmin,zmax))
   for _z in zval:
      for __z in _z:
         fid.write('%f '%(__z))
      fid.write('\n')

   fid.close()
   # }}}

if __name__ == '__main__':
   import matplotlib.pyplot as plt
   fname = '../../Data/Surfer/surfer-grid.grd'
   data = ReadGrd(fname,verbose=1)

   xg   = data['xg']
   yg   = data['yg']
   zval = data['zval']

   print('   Test WriteGrd.')
   fname = '../../temp/temp.grd'
   WriteGrd(fname,xg,yg,zval)

   #fig, ax = plt.subplots(facecolor='w')
   #ax.pcolormesh(data['xg'],data['yg'],data['zval'])
   #plt.show()
