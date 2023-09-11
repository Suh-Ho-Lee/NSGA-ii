#!/usr/bin/env python

from ..utils.varargout import varargout
from ..utils.print import print_
import inspect,flopy
import numpy as np

def flopyGetXY(mf,center=1,debug=False,isoffset=1): # {{{
   import warnings
   """
   Explain
   """
   warnings.warn('WARN: flopyGetXY will be replaced to "GetXy".')
   x,y,z = GetXy(mf,center=1,debug=debug,isoffset=1)
   return varargout(x,y,z)
   # }}}

def GetXy(mf,center=1,debug=False,isoffset=1): # {{{
    """
    Explain
     Get x,y grid points from modflow model.
     :math:$x R^n$
     :math:$y R^m$

    Usage
    x,y = flopyGetXY(mf)

    # get centered x,y poistion
    x,y   = flopyGetXY(mf)
    x,y,z = flopyGetXY(mf)

    Options
    isoffset    - consider xul and yul value in modflow.
                    ul = uppler left
    """
    # define function name.
    f_name = inspect.currentframe().f_code.co_name

    if not isinstance(mf,flopy.modflow.mf.Modflow):
        raise Exception('Error: input file is not type(flopy.modflow.Modflow.mf')
    nlay = mf.dis.nlay # number of z dir
    nrow = mf.dis.nrow # number of y dir
    ncol = mf.dis.ncol # number of x dir

    # [nlay, nrow, ncol]
    dx = mf.dis.delr.array # x direction
    dy = mf.dis.delc.array # y direction
    x = np.zeros(dx.shape[0]+1)
    y = np.zeros(dy.shape[0]+1)
    #x[1:] = dx.cumsum() # cumulative sum for calculating x based on dx.
    #y[1:] = dy.cumsum() # cumulative sum for calculating y based on dy.
    x[1:] = np.add.accumulate(dx)
    y[1:] = np.add.accumulate(dy)

    # get model domain.
    Lx = 0
    Ly = np.sum(dy)

    # calculate centered coorinates, because modflow use block centered method.
    if center:
        x = (x[0:-1]+x[1:])/2
        y = (y[0:-1]+y[1:])/2

    # update x y value with Lx, and Ly of model domain.
    x = Lx + x
    y = Ly - y

    # get coorner coordinates
    print_('   %s: get global coordinate'%(f_name),debug=debug)
    if flopy.__version__ <= '3.3.3':
        print_('   {}: flopy version = {}'.format(f_name,flopy.__version__),debug=debug)
        xul = mf.dis._sr.xul # upper left corner grid
        yul = mf.dis._sr.yul # upper left corder grid
    elif flopy.__version__ >= "3.3.4":
        print_('   {}: flopy version = {}'.format(f_name,flopy.__version__),debug=debug)
        xul = mf.modelgrid.xoffset # upper left corner grid
        yul = mf.modelgrid.yoffset # upper left corder grid
    else:
        print_('   current version(flopy {}) is not available'.format(flopy.__version__),
                debug=1)

    print_('   {}: xul = {}'.format(f_name,xul),debug=debug)
    print_('   {}: yul = {}'.format(f_name,yul),debug=debug)

    # calibarte global coordinate with xul, yul.
    print_('   {}: len x = {}'.format(f_name,np.shape(x)),debug=debug)
    print_('   {}: len y = {}'.format(f_name,np.shape(y)),debug=debug)
    if isoffset:
        if xul:
            print_('   xul = %f'%(xul),debug=debug)
            x = x + xul
        if yul:
            print_('   yul = %f'%(yul),debug=debug)
            y = y + yul

    # get z elevation from botm and top elevation.
    z    = np.zeros((nlay,nrow,ncol),dtype=float)
    botm = mf.dis.botm.array
    top  = mf.dis.top.array

    # calculate z elevation based on block centered method.
    print_(['top  array shape = ',np.shape(top)],debug=debug)
    print_(['botm array shape = ',np.shape(botm[0,:,:])],debug=debug)

    # top elevation
    z[0,:,:] =  (botm[0,:,:]+top)/2
    # bottom elevation
    for i in range(1,nlay):
        z[i,:,:] = np.mean(botm[i:i+1,:,:],axis=0)

    return varargout(x,y,z)
    # }}}
