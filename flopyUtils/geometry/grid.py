#!/usr/bin/env python

import flopy, inspect, numpy
import numpy as np
from ..utils.varargout import varargout
from ..utils.print import print_

__all__ = ['getXyGrid','GetDxDy']

def flopyGetXY(mf,center=1,debug=False,isoffset=1): # {{{
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
    x = numpy.zeros(dx.shape[0]+1)
    y = numpy.zeros(dy.shape[0]+1)
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

def flopyGetXyz(mf,center=1,debug=False): # {{{
    r'''
    Explain
     Get x y z grid coordinates from "dis" package.

     :math:`x \in R^{n \times m}`
     :math:`y \in R^{n \times m}`

    Usage
     x,y,z = flopyGetXyz(mf)
    '''
    x,y,z = flopyGetXY(mf,center=center,debug=debug)
    return x,y,z
# }}}

def flopyGetXyzGrid(mf,center=0,debug=False): # {{{
    '''
    Explain
     Get x,y meshgrid points from modflow model.

    Usage
     # get un-centered x,y grid position
     xg,yg = flopyGetXyzGrid(mf)

     # get centered x,y grid poistion
     xg,yg,zg = flopyGetXyzGrid(mf,center=1)

    '''
    if not isinstance(mf,flopy.modflow.mf.Modflow):
        raise Exception('Error: input file is not type(flopy.modflow.Modflow.mf')

    if 0:
        # old version for get x,y coordinates {{{
        nlay = mf.dis.nlay
        ncol = mf.dis.ncol
        nrow = mf.dis.nrow

        # get coorner coordinates
        xul = mf.dis._sr.xul # upper left corner grid
        yul = mf.dis._sr.yul # upper left corder grid

        # [nlay, nrow, ncol]
        dx = mf.dis.delr.array
        dy = mf.dis.delc.array
        x = numpy.zeros(dx.shape[0]+1)
        y = numpy.zeros(dy.shape[0]+1)
        x[1:] = dx.cumsum()
        y[1:] = dy.cumsum()

        # calculate centered coorinates, because modflow use block centered method.
        if center:
            x = (x[0:-1]+x[1:])/2
            y = (y[0:-1]+y[1:])/2

        # calibarte global coordinate with xul, yul.
        if debug:
            print('len x = ',x.shape)
            print('len y = ',y.shape)

        if xul:
            x = x + xul
        if yul:
            y = -y + yul

        # get z elevation from geometry
        bot = mf.dis.botm.array
        top = mf.dis.top.array
        top = top.reshape((1,nrow,ncol))
        mfPrint(['   nlay = ',mf.dis.nlay] ,debug=debug)
        mfPrint(['   bot = ',np.shape(bot)],debug=debug)
        mfPrint(['   top = ',np.shape(top)],debug=debug)
        z = np.concatenate((top,bot),axis=0)
        # }}}
    else:
        x,y,z = flopyGetXyz(mf,center=center,debug=debug)

    # generate mesh grid.
    x,y = np.meshgrid(x,y)

    return varargout(x,y,z)
# }}}

def flopyGetXyGrid(mf,center=0,debug=False): # {{{
    '''
    Explain
     get 2d xy grid array.
    '''
    xg,yg = flopyGetXyzGrid(mf,center=center,debug=debug)
    return varargout(xg,yg)
# }}}

def getXyGrid(mf,center=0,debug=False): # {{{
   return flopyGetXyGrid(mf,center=center,debug=debug)
   # }}}

def GetDxDy(mf,debug=0): # {{{
   '''
   Explain
    Get dx, dy value in flopy. "delr" and "delr" in modflow are dx, dy, respectively.

   Usage
    mf = flopy.modflow.Modflow()
    dx, dy = GetDxDy(mf)
   '''
   return mf.dis.delr.array, mf.dis.delc.array
   # }}}
