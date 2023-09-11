#!/usr/bin/env python
import numpy as np
import os,sys,flopy
#from ..utils.print import print_
from ..utils.print import *
from .GetXy import *

def xy2index(mf,x,y,debug=0): # {{{
    '''
    Explain
     Get specific row and column from x,y coordinates. This function is alternative to flopyXyToIndex.

    Usage
     row, col = xy2index(mf,welx,wely)
    '''

    # check inputs
    if isinstance(x,float) | isinstance(x,int):
       x = np.array([x],dtype=float)
    if isinstance(y,float) | isinstance(x,int):
       y = np.array([y],dtype=float)

    # get model grid
    xg,yg = mf.modelgrid.xycenters

    # consider model coordinate system from EPSG.
    xg = xg + mf.modelgrid.xoffset
    yg = yg + mf.modelgrid.yoffset

    print_('   obs xy -> obs row, col data',debug=debug)
    row = []  # y direction
    col = []  # x direction
    for x_, y_ in zip(x,y):
        col.append(np.argmin(abs(xg-x_)))
        row.append(np.argmin(abs(yg-y_)))

    return row, col
    # }}}
def xyz2index(mf,x,y,z,debug=0): # {{{
    '''
    Explain
     Get specific row and column from x,y,z coordinates. This function is alternative to flopyXyzToIndex.

    Usage
     lay, row, col = xyz2index(mf,welx,wely,welz)
    '''

    # get model grid
    xg,yg = mf.modelgrid.xycenters

    # consider model coordinate system from EPSG.
    xg = xg + mf.modelgrid.xoffset
    yg = yg + mf.modelgrid.yoffset

    print_('   obs xy -> obs row, col data',debug=debug)
    lay = np.zeros(np.shape(z))# z direction
    row = []  # y direction
    col = []  # x direction
    for x_, y_ in zip(x,y):
        col.append(np.argmin(abs(xg-x_)))
        row.append(np.argmin(abs(yg-y_)))
    row = np.array(row)
    col = np.array(col)

    for i, z_, r_, c_  in zip(range(len(z)),z,row,col):
        for layer in range(mf.nlay):
            check = (mf.modelgrid.top_botm[layer,r_,c_]
                    >= z_
                    >= mf.modelgrid.top_botm[layer+1,r_,c_])
            if check:
                lay[i] = layer

    return lay, row, col
    # }}}
def index2xyz(mf,lays,rows,cols,debug=False):# {{{
    '''
    Explain
     Get x y coordinates from (nlay, nrow, ncol) array.
                               z      y    x

    Usage
     wel = mf.wel.stress_period_data[0]
     x,y = flopyIndexToXy(mf,wel[2],wel[1],wel[0])

     # others
     x,y,z = flopyIndexToXy(mf,lay,row,col)

    Options
     debug - show process of current function.
    '''

    # get x,y,z coordinates in mf
    xi,yi,zi = flopyGetXY(mf,center=1)

    # get model information
    nrow, ncol, nlay, _ = mf.get_nrow_ncol_nlay_nper()

    if isinstance(cols,int):
       cols = [cols]
    if isinstance(rows,int):
       rows = [rows]
    if isinstance(lays,int):
       lays = [lays]

    # force dtype as int
    if not isinstance(cols,list):
        cols = cols.astype(int)
    if not isinstance(rows,list):
        rows = rows.astype(int)
    if not isinstance(lays,list):
        lays = lays.astype(int)

    mfPrint('   cols = {0}'.format(cols),debug=debug)
    mfPrint('   rows = {0}'.format(rows),debug=debug)

    if not np.shape(lays) or not np.any(lays):
        mfPrint('   Force lays',debug=debug)
        mfPrint('   check shape of cols = {0}'.format(np.shape(cols)),debug=debug)
        lays = np.zeros(np.shape(cols))

    # check length of array
    if np.shape(lays) != np.shape(rows) or np.shape(lays) != np.shape(cols) or np.shape(rows) != np.shape(cols):
    #if len(lays) != len(rows) or len(lays) != len(cols) or len(rows) != len(cols):
        raise Exception('ERROR: check length of x,y,z.')
    
    # check lay, row, col boundary.
    if np.amax(lays) > nlay-1:
        raise Exception('Error max index of lays(=%d) is outside of boundary'%(np.amax(lays)))
    if np.amax(rows) > nrow-1:
        raise Exception('Error max index of rows(=%d) is outside of boundary'%(np.amax(rows)))
    if np.amax(cols) > ncol-1:
        raise Exception('Error max index of cols(=%d) is outside of boundary'%(np.amax(cols)))

    # initialize x,y points
    x = np.zeros(np.shape(cols))
    y = np.zeros(np.shape(rows))
    z = np.zeros(np.shape(lays))
    if not np.shape(cols):
        x = xi[int(cols)]
        y = yi[int(cols)]
        z = zi[int(lays), int(rows), int(cols)]
    else:
        for i in range(np.shape(cols)[0]):
            x[i] = xi[int(cols[i])]
            y[i] = yi[int(rows[i])]
            z[i] = zi[int(lays[i]), int(rows[i]), int(cols[i])]

    return varargout(x,y,z)
# }}}
def index2xy(mf,rows,cols,debug=False):# {{{
    '''
    Explain
     Get x y coordinates from (nlay, nrow, ncol) array.
                               z      y    x

    Usage
     wel = mf.wel.stress_period_data[0]
     x,y = flopyIndexToXy(mf,wel[2],wel[1],wel[0])

     # others
     x,y,z = flopyIndexToXy(mf,lay,row,col)

    Options
     debug - show process of current function.
    '''

    # set  lays...
    lays = np.zeros(np.shape(rows),dtype=int)

    x,y,_ = index2xyz(mf,lays,rows,cols,debug=debug)

    return varargout(x,y)
# }}}

