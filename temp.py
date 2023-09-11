#!/usr/bin/env python

# import modules {{{
# system modules
import sys
import glob
import os
import platform
import copy
import shutil
import math
from scipy.stats import cauchy
import flopy.utils.binaryfile as bf
from tempfile import TemporaryDirectory

import flopyUtils
#import flopyUtils

if 'Linux' in platform.platform():
    sys.path.append('./code/flopy-utils')
else:
    sys.path.append("C:\\Users\\SUHHO LEE\\Desktop\\flopy\\code\\flopy-utils")
    sys.path.append("C:\\Users\\SUHHO LEE\\Desktop\\flopy")
# inwoo's packages
from organizer import *
from flopyUtils import *
import modules #load user defined function

import shapely
import numpy as np
import flopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pareto
import geopandas
import random
# Gif making packages
import glob
from PIL import Image

# change EPSG
import osgeo
from osgeo import osr
import flopy.discretization.grid as fdg

# import csv file
import pandas

# parallel computing
import multiprocessing
import threading
# }}}
# Tools for GW-GA
import deap
from deap import creator
from deap import base
from deap import tools
import time
# }}}
# Performance metric
import hypervolume
import imageio
# }}}

figno = 1
steps = [27,28,29,30,31,32,33,34,35,36,37]
org = organizer(steps=steps)

def evalObjFuncMoney2(mf, mt, i, workdir, welc, welr, welp_1, welp_2, welp_3, weight_conc,queue):  # {{{
    mf.name = 'mf'
    mf.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mf._model_ws):
        os.mkdir(mf._model_ws)
    mf = ChangeOutputName(mf)

    mt.name = 'mt'
    mt.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mt._model_ws):
        os.mkdir(mt._model_ws)
    mt = ChangeOutputName(mt)
    #print('mt3d name = %s' % (mt.name))

    # get pumping time.
    perlen = mf.dis.perlen.array[0]

    # change well location
    stress_period_data = {0: [[0, welr, welc, -welp_1], [0, 9, 8, -welp_2], [0, 3, 8, -welp_3]]}
    flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

    # LMT Linkage with MT3DMS for multi-species mass transport modeling
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

    # set mt 3d variables.
    ssm_data1 = []
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    pos = np.where(mf.bas6.ibound.array == 0)
    #print(np.shape(pos)[1])
    for index in range(np.shape(pos)[1]):
        ssm_data1.append([pos[0][i], pos[1][i], pos[2][i], 0, -1])

    # set pumping/injection well
    # for i in range(len(row)):
    ssm_data1.append([0, welr, welc, 0, itype['WEL']])

    stress_period_data = {0: ssm_data1}
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=stress_period_data)

    # change ftl file name for flopy
    mt.ftlfilename = 'mt3d_link.ftl'

    # run modflow model
    mf.write_input()
    mf.run_model(silent=1)

    # run mt3d model
    mt.write_input()
    mt.run_model(silent=1)

    # get solute mass
    # use flopy package tool.
    mas = mt.load_mas(mt.model_ws + '/MT3D001.MAS')
    mas = np.array(mas.tolist())
    total_mass = mas[:,6]
    # define concentration
    cmin = 0.03
    cobj = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D001.UCN')
    c = cobj.get_data(idx= 10, totim=cobj.get_times()[-1])
    pos = np.where(c > cmin)
    c_new = np.zeros(np.shape(c))
    c_new[pos] = c[pos] - cmin

    # get area
    vol = mf.dis.get_cell_volumes()
    c_vol = c_new * vol

    # cost function
    #  money for total pumping rate + remaining contaminant cost.
    #  m3/day * rho_water
    #print('%d th  total_mass = %f'%(i,total_mass))
    cost = np.zeros((3,),dtype=float)
    cost[0] = 1000000
    cost[1] = 0.17*(85.4 * (welp_1+welp_2+welp_3)*perlen)
    cost[2] = 10*np.sum(c_vol)

    # output results
    queue.put([i, np.sum(cost), cost])
#}}}
def evalObjNSGA(mf, mt, i, workdir,welp_1,welp_2,welp_3,welp_4,welp_5,welp_6,welp_7,welp_8,welp_9,welp_10,welp_13,welp_14,welp_15,welp_16,welp_17,welp_18,queue,processlock):  # {{{
    # start processing
    processlock.acquire()

    mf.name = 'mf'
    mf.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mf._model_ws):
        os.mkdir(mf._model_ws)
    mf = ChangeOutputName(mf)

    mt.name = 'mt'
    mt.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mt._model_ws):
        os.mkdir(mt._model_ws)
    mt = ChangeOutputName(mt)
    #print('mt3d name = %s' % (mt.name))

    # get pumping time.
    perlen = mf.dis.perlen.array[0]

    # change well location
    # order of the well name -> MW_1, MW_2, DJ_1, DJ_2, MW_3, MW_4, MW_5, BH_1, BH_2, BH_3, NDMW_09, NDMW_10, NDMW_11, NDMW_12, NDMW_13, NDMW_14
    stress_period_data = {0: [[1, 43, 56, -welp_1], [1, 50, 63, -welp_2], [1, 61, 46, -welp_3], [1, 62, 45, -welp_4],[1, 72, 40, -welp_5], [1, 79, 62, -welp_6], [1, 95, 60, -welp_7], [1, 103, 33, -welp_8], [1, 114, 64, -welp_9], [1, 108, 103, -welp_10], [1, 94, 66, -welp_13], [1, 87, 101, -welp_14], [1, 81, 82, -welp_15], [1, 102, 91, -welp_16], [1, 67, 65, -welp_17], [1, 119, 61, -welp_18]]}
    flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)
    #change recharge rate for transient model
    #flopy.modflow.ModflowRch(mf, nrchop=3, rech=1.3e-5)

    # LMT Linkage with MT3DMS for multi-species mass transport modeling
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

    # set mt 3d variables.
    #ssm_data = []
    #itype = flopy.mt3d.Mt3dSsm.itype_dict()
    #pos = np.where(mf.bas6.ibound.array == 0)
    # print(np.shape(pos)[1])
    #for index in range(np.shape(pos)[1]):
        #ssm_data.append([pos[0][i], pos[1][i], pos[2][i], 0, -1])

    # set pumping/injection well
    # for i in range(len(row)):
    # ssm_data1.append([0, welr, welc, 0, itype['WEL']])

    #stress_period_data = {0: ssm_data}
    #flopy.mt3d.Mt3dSsm(mt, stress_period_data=stress_period_data)

    # change ftl file name for flopy
    mt.ftlfilename = 'mt3d_link.ftl'

    # remove files, such as *.MAS and *.UCN
    # MT3D001 is PCE and MT3D002 is TCE
    if os.path.isdir(mt.model_ws):
        if os.path.isfile(mt.model_ws + '/MT3D001.UCN'):
            os.remove(mt.model_ws + '/MT3D001.UCN')
        if os.path.isfile(mt.model_ws + '/MT3D001.MAS'):
            os.remove(mt.model_ws + '/MT3D001.MAS')

    # run modflow model
    mf.write_input()
    mf.run_model(silent=1)

    # run mt3d model
    mt.write_input()
    mt.run_model(silent=1)

    # cost function
    #  money for total pumping rate + remaining contaminant cost.
    #  m3/day * rho_water
    #print('%d th  total_mass = %f'%(i,total_mass))
    cost1 = np.zeros((2,),dtype=float)
    cost1[0] = 1000000
    cost1[1] = 85.4*(welp_1+welp_2+welp_3+welp_4+welp_5+welp_6+welp_7+welp_8+welp_9+welp_10+welp_13+welp_14+welp_15+welp_16+welp_17+welp_18)*500
    cost1 = np.sum(cost1)

    # get total concentration for second objective func.
    # use flopy package tool.
    cobj_1 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D001.UCN')
    cobj_2 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D002.UCN')
    c_1 = cobj_1.get_data(totim=cobj_1.get_times()[-1], mflay=1)
    c_2 = cobj_2.get_data(totim=cobj_2.get_times()[-1], mflay=1)
    mas_1 = mt.load_mas(mt.model_ws + '/MT3D001.MAS')
    mas_1 = np.array(mas_1.tolist())
    mas_2 = mt.load_mas(mt.model_ws + '/MT3D002.MAS')
    mas_2 = np.array(mas_2.tolist())
    #print(np.shape(mas))
    #print(mas[:,6])
    cost2 = mas_1[:, 6][-1] + mas_2[:, 6][-1]
    #cost2 = np.zeros((2,), dtype=float)
    #for i in range(32,90):
        #cost2[0] = c_2[132,i]
    #cost2[1] = c_1[51,87]+ c_1[52,88]+c_1[53,89]+c_1[54,90]+c_1[55,91]+c_1[56,92]+c_1[57,93]+c_1[58,94]
    #cost2 = np.sum(cost2)
    #print('cost1 = {}, cost2 = {}'.format(cost1,cost2[-1]))

    # output results
    queue.put([i, cost1, cost2])

    # done process
    processlock.release()
#}}}
def evalObjNSGA_2(mf, mt, i, workdir, welc_1, welr_1, welc_2, welr_2, welc_3, welr_3, welc_4, welr_4, welc_5, welr_5, welp_1, welp_2, welp_3,welp_4,welp_5,queue,processlock):  # {{{
    # start processing
    processlock.acquire()

    workdir = './NSGA_2/'
    mf.name = 'mf'
    mf.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mf._model_ws):
        os.mkdir(mf._model_ws)
    mf = ChangeOutputName(mf)

    mt.name = 'mt'
    mt.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mt._model_ws):
        os.mkdir(mt._model_ws)
    mt = ChangeOutputName(mt)
    #print('mt3d name = %s' % (mt.name))

    # get pumping time.
    perlen = mf.dis.perlen.array[0]

    # injection and pumping well
    stress_period_data = {0: [[1, welr_1, welc_1, -welp_1], [1, welr_2, welc_2, -welp_2], [1, welr_3, welc_3, -welp_3], [1, welr_4, welc_4, -welp_4], [1, welr_5, welc_5, -welp_5]]}
    flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

    # LMT Linkage with MT3DMS for multi-species mass transport modeling
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

    # set mt 3d variables.
    #ssm_data1 = []
    #itype = flopy.mt3d.Mt3dSsm.itype_dict()
    #pos = np.where(mf.bas6.ibound.array == 0)
    # print(np.shape(pos)[1])
    #for index in range(np.shape(pos)[1]):
        #ssm_data1.append([pos[0][i], pos[1][i], pos[2][i], 0, -1])

    # set pumping/injection well
    # for i in range(len(row)):
    # ssm_data1.append([0, welr, welc, 0, itype['WEL']])

    #stress_period_data = {0: ssm_data1}
    #ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=stress_period_data, crch=0.0)

    # change ftl file name for flopy
    mt.ftlfilename = 'mt3d_link.ftl'

    # remove files, such as *.MAS and *.UCN
    if os.path.isdir(mt.model_ws):
        if os.path.isfile(mt.model_ws + '/MT3D001.UCN'):
            os.remove(mt.model_ws + '/MT3D001.UCN')
        if os.path.isfile(mt.model_ws + '/MT3D001.MAS'):
            os.remove(mt.model_ws + '/MT3D001.MAS')

    # run modflow model
    mf.write_input()
    mf.run_model(silent=1)

    # run mt3d model
    mt.write_input()
    mt.run_model(silent=1)

    # cost function
    #  money for total pumping rate + remaining contaminant cost.
    #  m3/day * rho_water
    #print('%d th  total_mass = %f'%(i,total_mass))
    cost1 = np.zeros((2,),dtype=float)
    cost1[0] = 1000000
    cost1[1] = 85.4 * (welp_1 + welp_2 + welp_3 + welp_4 + welp_5) * perlen
    cost1 = np.sum(cost1)

    # get total concentration for second objective func.
    # use flopy package tool.
    cobj_1 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D001.UCN')
    cobj_2 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D002.UCN')
    c_1 = cobj_1.get_data(totim=cobj_1.get_times()[-1], mflay=1)
    c_2 = cobj_2.get_data(totim=cobj_2.get_times()[-1], mflay=1)
    mas_1 = mt.load_mas(mt.model_ws + '/MT3D001.MAS')
    mas_1 = np.array(mas_1.tolist())
    mas_2 = mt.load_mas(mt.model_ws + '/MT3D002.MAS')
    mas_2 = np.array(mas_2.tolist())
    #print(np.shape(mas))
    #print(mas[:,6])
    cost2 = mas_1[:,6][-1]+mas_2[:,6][-1]
    # output results
    queue.put([i, cost1, cost2])

    # done process
    processlock.release()
#}}}
def evalObjNSGA_3(mf, mt, i, workdir,welp_1,welp_2,welp_3,welp_4,welp_5,welp_6,welp_7,welp_8,welp_9,welp_10,welp_11,welp_12,welp_13,welp_14,welp_15,welp_16,welp_17,welp_18,welp_19,queue,processlock):  # {{{
    # start processing
    processlock.acquire()

    mf.name = 'mf'
    mf.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mf._model_ws):
        os.mkdir(mf._model_ws)
    mf = ChangeOutputName(mf)

    mt.name = 'mt'
    mt.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mt._model_ws):
        os.mkdir(mt._model_ws)
    mt = ChangeOutputName(mt)
    #print('mt3d name = %s' % (mt.name))

    # get pumping time.
    perlen = mf.dis.perlen.array[0]

    # change well location
    # order of the well name -> KDMW-7, MW-23, KDMW-6, GW-7, MW-12, GW-18, KDPW-11, KDPW-7, KDPW-10, KDPW-8, KDPW-9, KDMW-9, KDMW-10, KDMW-12, KDMW-8, KDMW-13, KDMW-14, KDMW-15, KDMW-16
    wel_sp1 = [[0, 61, 29, -welp_1], [0, 65, 32, -welp_2], [0, 76, 36, -welp_3], [0, 82, 47, -welp_4],[0, 63, 45, -welp_5], [0, 62, 37, -welp_6], [0, 60, 35, -welp_7], [0, 59, 33, -welp_8], [0, 57, 30, -welp_9], [0, 56, 25, -welp_10], [0, 55, 23, -welp_11], [0, 49, 26, -welp_12], [0, 46, 28, -welp_13], [0, 54, 31, -welp_14], [0, 53, 34, -welp_15], [0, 52, 38, -welp_16], [0, 50, 41, -welp_17], [0, 47, 42, -welp_18], [0, 43, 43, -welp_19]]
    wel_sp2 = [[0, 61, 29, 0], [0, 65, 32, 0], [0, 76, 36, 0], [0, 82, 47, 0],[0, 63, 45, 0], [0, 62, 37, 0], [0, 60, 35, 0], [0, 59, 33, 0], [0, 57, 30, 0], [0, 56, 25, 0], [0, 55, 23, 0], [0, 49, 26, 0], [0, 46, 28, 0], [0, 54, 31, 0], [0, 53, 34, 0], [0, 52, 38, 0], [0, 50, 41, 0], [0, 47, 42, 0], [0, 43, 43, 0]]
    stress_period_data = {0:wel_sp2, 46:wel_sp1}
    flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)
    #change recharge rate for transient model
    #flopy.modflow.ModflowRch(mf, nrchop=3, rech=1.3e-5)

    # LMT Linkage with MT3DMS for multi-species mass transport modeling
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

    # set mt 3d variables.
    #ssm_data = []
    #itype = flopy.mt3d.Mt3dSsm.itype_dict()
    #pos = np.where(mf.bas6.ibound.array == 0)
    # print(np.shape(pos)[1])
    #for index in range(np.shape(pos)[1]):
        #ssm_data.append([pos[0][i], pos[1][i], pos[2][i], 0, -1])

    # set pumping/injection well
    # for i in range(len(row)):
    # ssm_data1.append([0, welr, welc, 0, itype['WEL']])
    # ssm = flopy.mt3d.Mt3dSsm(mt, mxss=12000)

    # change ftl file name for flopy
    mt.ftlfilename = 'mt3d_link.ftl'

    # remove files, such as *.MAS and *.UCN
    # MT3D001 is PCE and MT3D002 is TCE
    if os.path.isdir(mt.model_ws):
        if os.path.isfile(mt.model_ws + '/MT3D001.UCN'):
            os.remove(mt.model_ws + '/MT3D001.UCN')
        if os.path.isfile(mt.model_ws + '/MT3D001.MAS'):
            os.remove(mt.model_ws + '/MT3D001.MAS')
    # run modflow model
    mf.write_input()
    mf.run_model(silent=1)

    # run mt3d model
    mt.write_input()
    mt.run_model(silent=1)

    # cost function
    #  money for total pumping rate + remaining contaminant cost.
    #  m3/day * rho_water
    #print('%d th  total_mass = %f'%(i,total_mass))
    cost1 = np.zeros((2,),dtype=float)
    cost1[0] = 19000000
    cost1[1] = 85.4*(welp_1+welp_2+welp_3+welp_4+welp_5+welp_6+welp_7+welp_8+welp_9+welp_10+welp_11+welp_12+welp_13+welp_14+welp_15+welp_16+welp_17+welp_18+welp_19)*450
    cost1 = np.sum(cost1)

    # get total concentration for second objective func.
    # use flopy package tool.
    mas = mt.load_mas(mt.model_ws + '/MT3D001.MAS')
    mas = np.array(mas.tolist())
    cobj = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D001.UCN')
    c = cobj.get_data(totim=cobj.get_times()[-1], mflay=0)
    #a = c[20,3]
    #print(a)
    #print(np.shape(mas))
    #print(mas[:,6])
    cost2 = mas[:,6][-1]
    #print('cost1 = {}, cost2 = {}'.format(cost1,cost2))

    # output results
    queue.put([i, cost1, cost2])

    # done process
    processlock.release()
#}}}
def evalObjNSGA_4(mf, mt, i, workdir,welp_1, welp_2, queue,processlock):  # {{{
    # start processing
    processlock.acquire()

    mf.name = 'mf'
    mf.model_ws = workdir + '/GA%d' % (i)
    if not os.path.isdir(mf._model_ws):
        os.mkdir(mf._model_ws)
    mf = ChangeOutputName(mf)

    mt.name = 'mt'
    mt.model_ws = workdir + '/GA%d' % (i)
    if not os.path.isdir(mt._model_ws):
        os.mkdir(mt._model_ws)
    mt = ChangeOutputName(mt)
    # print('mt3d name = %s' % (mt.name))

    # get pumping time.
    perlen = mf.dis.perlen.array[0]

    #Well location
    wel_sp1 = [[0, 49, 49, welp_1], [2, 49, 44, -welp_2]]
    wel_sp2 = [[0, 49, 49, 0], [2, 49, 44, 0]]
    stress_period_data = {0: wel_sp2, 39: wel_sp1, 45: wel_sp2, 51: wel_sp1}
    flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

    # LMT Linkage with MT3DMS for multi-species mass transport modeling
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

    # change ftl file name for flopy
    mt.ftlfilename = 'mt3d_link.ftl'

    # remove files, such as *.MAS and *.UCN
    # MT3D001 is PCE and MT3D002 is TCE
    if os.path.isdir(mt.model_ws):
        if os.path.isfile(mt.model_ws + '/MT3D001.UCN'):
            os.remove(mt.model_ws + '/MT3D001.UCN')
        if os.path.isfile(mt.model_ws + '/MT3D001.MAS'):
            os.remove(mt.model_ws + '/MT3D001.MAS')
    # run modflow model
    mf.write_input()
    mf.run_model(silent=1)

    # run mt3d model
    mt.write_input()
    mt.run_model(silent=1)

    #Cost function
    cost1 = np.zeros((2,),dtype=float)
    cost1[0] = 1000000
    cost1[1] = 85.4*(welp_1+welp_2)*24
    cost1 = np.sum(cost1)

    #Concentration function
    cobj = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D001.UCN')
    c_1 = cobj.get_data(totim=59.0, mflay=0)
    c_2 = cobj.get_data(totim=44.0, mflay=0)
    cost2 = c_1[49,44]

    # output results
    queue.put([i, cost1, cost2])

    # done process
    processlock.release()
#}}}
def evalObjNSGA_5(mf, mt, i, workdir, welr_1, welc_1, welr_2, welc_2, welr_3, welc_3, welp_1,welp_2,welp_3, queue, processlock):
    # start processing
    processlock.acquire()

    mf.name = 'mf'
    mf.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mf._model_ws):
        os.mkdir(mf._model_ws)
    mf = ChangeOutputName(mf)

    mt.name = 'mt'
    mt.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mt._model_ws):
        os.mkdir(mt._model_ws)
    mt = ChangeOutputName(mt)
    #print('mt3d name = %s' % (mt.name))

    # get pumping time.
    perlen = mf.dis.perlen.array[0]

    # change well location
    stress_period_data = {0: [[1, welr_1, welc_1, -welp_1],[1, welr_2, welc_2, -welp_2],[1, welr_3, welc_3, -welp_3]]}
    flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

    # LMT Linkage with MT3DMS for multi-species mass transport modeling
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

    # set mt 3d variables.
    '''ssm_data1 = []
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    pos = np.where(mf.bas6.ibound.array == 0)
    #print(np.shape(pos)[1])
    for index in range(np.shape(pos)[1]):
        ssm_data1.append([pos[0][i], pos[1][i], pos[2][i], 0, -1])

    # set pumping/injection well
    # for i in range(len(row)):
    ssm_data1.append([0, welr, welc, 0, itype['WEL']])

    stress_period_data = {0: ssm_data1}
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=stress_period_data)'''

    # change ftl file name for flopy
    mt.ftlfilename = 'mt3d_link.ftl'

    # run modflow model
    mf.write_input()
    mf.run_model(silent=1)

    # run mt3d model
    mt.write_input()
    mt.run_model(silent=1)

    # get solute mass
    # define concentration
    cmin_1 = 10
    #cmin_2 = 0.03
    cobj_1 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D001.UCN')
    #cobj_2 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D002.UCN')
    c_1 = cobj_1.get_data(totim=cobj_1.get_times()[-1], mflay=1)
    #c_2 = cobj_2.get_data(totim=cobj_1.get_times()[-1], mflay=1)
    pos_1 = np.where(c_1 > cmin_1)
    #pos_2 = np.where(c_2 > cmin_2)
    c_new_1 = np.zeros(np.shape(c_1))
    #c_new_2 = np.zeros(np.shape(c_2))
    c_new_1[pos_1] = c_1[pos_1] - cmin_1
    #c_new_2[pos_2] = c_2[pos_2] - cmin_2

    #print(c_new_1)
    #print(c_new_2)
    #print(c_new_1[pos_1])
    #print(c_new_2[pos_2])

    # get area
    #vol = mf.dis.get_cell_volumes()
    #c_vol_1 = c_new_1 * vol
    #c_vol_2 = c_new_2 * vol

    # cost function
    #  money for total pumping rate + remaining contaminant cost.
    #  m3/day * rho_water
    #print('%d th  total_mass = %f'%(i,total_mass))
    cost = np.zeros((3,),dtype=float)
    cost[0] = 1000000
    cost[1] = 85.4*((welp_1+welp_2+welp_3) * 3600)
    cost[2] = 8000*(np.sum(c_new_1[pos_1]))

    # output results
    queue.put([i, np.sum(cost), cost])

    # done process
    processlock.release()
#}}}
def evalObjNSGA_6(mf, mt, i, workdir, welp_1, welp_2, welp_3, welp_4, welp_5, welp_6, welp_7, welp_8, welp_9, welp_10, welp_11, welp_12, welp_13, welp_14, queue, processlock):
    # start processing
    processlock.acquire()

    mf.name = 'mf'
    mf.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mf._model_ws):
        os.mkdir(mf._model_ws)
    mf = ChangeOutputName(mf)

    mt.name = 'mt'
    mt.model_ws = workdir+'/GA%d' % (i)
    if not os.path.isdir(mt._model_ws):
        os.mkdir(mt._model_ws)
    mt = ChangeOutputName(mt)
    #print('mt3d name = %s' % (mt.name))

    # get pumping time.
    perlen = mf.dis.perlen.array[0]

    # change well location -> MW-1, MW-2, DJ-1, DJ-2, MW-3, MW-4, BH-1, MW-5, NDMW-9, BH-2, NDMW-12, BH-3, NDMW-11, NDMW-13
    wel_sp1 = [[1, 28, 39, -welp_1],[1, 33, 44, -welp_2],[1, 40, 32, -welp_3], [1, 41, 31, -welp_4], [1, 49, 28, -welp_5], [1, 54, 43, -welp_6], [1, 71, 23, -welp_7], [1, 65, 42, -welp_8], [1, 64, 47, -welp_9], [1, 81, 45, -welp_10], [1, 70, 65, -welp_11], [1, 76, 76, -welp_12], [1, 55, 58, -welp_13], [1, 44, 46, -welp_14]]
    wel_sp2 = [[1, 28, 39, 0],[1, 33, 44, 0],[1, 40, 32, 0],  [1, 41, 31, 0], [1, 49, 28, 0], [1, 54, 43, 0], [1, 71, 23, 0], [1, 65, 42, 0], [1, 64, 47, 0], [1, 81, 45, 0], [1, 70, 65, 0], [1, 76, 76, 0], [1, 55, 58, 0], [1, 44, 46, 0]]
    stress_period_data = {0:wel_sp2, 5:wel_sp1, 8:wel_sp2, 17:wel_sp1, 20:wel_sp2, 29:wel_sp1, 32:wel_sp2, 41:wel_sp1, 44:wel_sp2, 53:wel_sp1, 56:wel_sp2, 65:wel_sp1, 68:wel_sp2, 77:wel_sp1, 80:wel_sp2, 89:wel_sp1, 92:wel_sp2, 101:wel_sp1, 104:wel_sp2, 113:wel_sp1, 116:wel_sp2, 125:wel_sp1, 128:wel_sp2, 137:wel_sp1, 140:wel_sp2, 149:wel_sp1, 152:wel_sp2, 161:wel_sp1, 164:wel_sp2, 173:wel_sp1, 176:wel_sp2, 185:wel_sp1, 188:wel_sp2, 197:wel_sp1, 200:wel_sp2}
    flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

    # LMT Linkage with MT3DMS for multi-species mass transport modeling
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')

    # set mt 3d variables.
    '''ssm_data1 = []
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    pos = np.where(mf.bas6.ibound.array == 0)
    #print(np.shape(pos)[1])
    for index in range(np.shape(pos)[1]):
        ssm_data1.append([pos[0][i], pos[1][i], pos[2][i], 0, -1])

    # set pumping/injection well
    # for i in range(len(row)):
    ssm_data1.append([0, welr, welc, 0, itype['WEL']])

    stress_period_data = {0: ssm_data1}
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=stress_period_data)'''

    # change ftl file name for flopy
    mt.ftlfilename = 'mt3d_link.ftl'

    # run modflow model
    mf.write_input()
    mf.run_model(silent=1)

    # run mt3d model
    mt.write_input()
    mt.run_model(silent=1)

    # get solute mass
    #print(c_new_1)
    #print(c_new_2)
    #print(c_new_1[pos_1])
    #print(c_new_2[pos_2])

    # get area
    #vol = mf.dis.get_cell_volumes()
    #c_vol_1 = c_new_1 * vol
    #c_vol_2 = c_new_2 * vol

    # cost function
    #  money for total pumping rate + remaining contaminant cost.
    #  m3/day * rho_water
    #print('%d th  total_mass = %f'%(i,total_mass))
    cost1 = np.zeros((2,), dtype=float)
    cost1[0] = 1000000
    cost1[1] = 85.4*((welp_1+welp_2+welp_3+welp_4+welp_5+welp_6+welp_7+welp_8+welp_9+welp_10+welp_11+welp_12+welp_13+welp_14) * 1080)
    cost1 = np.sum(cost1)

    cobj_1 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D001.UCN')
    cobj_2 = flopy.utils.binaryfile.UcnFile(mt.model_ws + '/MT3D002.UCN')
    c_1 = cobj_1.get_data(totim=cobj_1.get_times()[-1], mflay=1)
    c_2 = cobj_2.get_data(totim=cobj_2.get_times()[-1], mflay=1)
    mas_1 = mt.load_mas(mt.model_ws + '/MT3D001.MAS')
    mas_1 = np.array(mas_1.tolist())
    mas_2 = mt.load_mas(mt.model_ws + '/MT3D002.MAS')
    mas_2 = np.array(mas_2.tolist())
    #cost2 = mas_1[:,6][-1]+mas_2[:,6][-1]
    cost2 = c_1[55,45]+c_1[55,46]+c_1[55,47]+c_1[55,48]+c_1[55,49]+c_1[55,50]+c_1[55,51]+c_1[55,52]+c_1[55,53]+c_1[55,54]+c_1[55,55]+c_2[91,37]+c_2[91,38]+c_2[91,39]+c_2[91,40]+c_2[91,41]+c_2[91,42]+c_2[91,43]+c_2[91,44]+c_2[91,45]+c_2[91,46]+c_2[91,47]+c_2[91,48]
    # output results
    queue.put([i, cost1, cost2])

    # done process
    processlock.release()
#}}}
if perform(org, 'GA optimization with original'):  # {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './TransientWonju2/mf_MODFLOW_text/'
    mt_ws = './TransientWonju2/mt_MT3DUSGS'

    # set working directory
    workdir = './genetic_algorithm3/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0

    # depending on machine system, executio file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3d-usgs_1.1.0_64.exe'

    # remove existed GA results {{{
    f = workdir  # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)

    for f in glob.glob(workdir + '/GA*'):
        if os.path.isdir(f):
            shutil.rmtree(f)
    for f in glob.glob(workdir + '/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir + '/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 1
    perlen = [100]
    nstp = [200]
    steady = [False]

    CXPB = 0.85  # probability of cross in genes.
    MUTPB = 0.05  # probability of mutation
    NGEN = 20   # number of generation for GA.
    npop = 20   # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load(namemf + '.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load(namemt3d + '.nam', version='mt3d-usgs',
            exe_name=exe_name_mt3d, model_ws=mt_ws,modflowmodel=copy.deepcopy(mf))

    # load model grid information
    nrow, ncol, nlay, nper = mf.get_nrow_ncol_nlay_nper()

    # update Dis for time domain.
    updateDis(mf, nper=1, perlen=perlen, nstp=nstp, steady=steady)

    # get xy grid.
    xg, yg = flopyGetXY(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_222/Data_2/ContaminantArea/Boundary_Boundary_polys.shp'
    polygons = geopandas.read_file(shpname)

    # load well locations.
    filename = './NSGA_222/Data_2/Well/well_location.xlsx'
    df = pandas.read_excel(filename, sheet_name='source')
    welx = df['X']
    wely = df['Y']
    welz = np.zeros(np.shape(welx))
    welc, welr, well = flopyXyzToIndex(mf, welx, wely, welz)
    print(welc)
    # get well index
    print('wel x,y shape = ', np.shape(welx))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welr_low = 0
    welr_up  = nrow-1
    welc_low = 0
    welc_up  = ncol-1
    welp_low = 1
    welp_up = 50
    toolbox.register("attr_welr", np.random.randint, welc_low, welc_up)
    toolbox.register("attr_welc", np.random.randint, welc_low, welc_up)
    toolbox.register("attr_welp", random.randint, welp_low, welp_up)

    # Structure initializers: define 'individual' to be an individual
    #                         consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welr, toolbox.attr_welc, toolbox.attr_welp), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_5)

    # register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welr_low, welc_low, welp_low]
    low = np.matlib.repmat(low, 1, 1)
    up  = [welr_up, welc_up, welp_up]
    up = np.matlib.repmat(up,1,1)
    toolbox.register("mutate", tools.mutUniformInt, low=low, up=up, indpb=0.2)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=2)

    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    proc = []
    for i, ind in enumerate(pop):
        ind = np.reshape(ind,(3,1))
        welr = ind[:,0]
        welc = ind[:,1]
        welp = ind[:,2]
        packages = {'welr':welr,'welc':welc,'welp':welp}
        p = threading.Thread(target=evalObjNSGA_5,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, packages, queue))
        proc.append(p)
        p.start()

    costs = np.zeros((npop,3),dtype=float)
    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses[i], costs[i,:] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses = fitnesses[np.argsort(order)]
    costs     = costs[np.argsort(order),:]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses[i],)

    g = 0
    with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
        fid.write('index,welx,wely,welp,total_cost,cost1,cost2,cost3\n')
        for i in range(npop):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], fitnesses[i], costs[i,0], costs[i,1], costs[i,2]))

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    best_ind = tools.selBest(pop, 1)[0]

    print("  Evaluated %i individuals" % len(pop))
    avgFit   = [sum(fits)/len(fits)]
    avgStd   = [std]
    minFit   = [min(fits)]
    maxFit   = [max(fits)]
    bestWelx = [welx[best_ind[0]]]
    bestWely = [wely[best_ind[0]]]
    bestWelp = [best_ind[1]]
    bestCosts  = np.zeros((NGEN+1,3),dtype=float)
    bestCosts[0,:]  = costs[np.argmin(fits),:]

    # Begin the evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            print('ind[0], ind[1] = %d, %f'%(ind[0], ind[1]))
            print('welc = %d, welr = %d, welp = %f'%(welc[ind[0]], welr[ind[0]], ind[1]))
            p = threading.Thread(target=evalObjFuncMoney2, args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, packages, queue))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses[i], costs[i,:] = queue.get()

        fitnesses = fitnesses[np.argsort(order)]
        costs     = costs[np.argsort(order),:]

        print('show fitnesses values')
        print(fitnesses)

        print('show values in pop.')
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        print("Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        print(pop)
        for i in range(npop):
            print(pop[i].fitness.values)
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        best_ind = tools.selBest(pop, 1)[0]

        avgFit.append(mean)
        avgStd.append(std)
        minFit.append(min(fits))
        maxFit.append(max(fits))

        bestWelx.append(welx[best_ind[0]])
        bestWely.append(wely[best_ind[0]])
        bestWelp.append(best_ind[1])
        bestOrder = np.argmin(fits)
        #print('where is argmin of fits?',bestOrder, costs[bestOrder,:])
        bestCosts[g+1,:] = costs[np.argmin(fits),:]

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print(bestWelx, bestWely, bestWelp)

        # save results in text format
        with open('./%s/ga_gen%03d.txt' % (resultsdir, g + 1), 'w') as fid:
            fid.write('index,welx,wely,welp,total_cost,cost1,cost2,cost3\n')
            for i in range(npop):
                fid.write('%d,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], fitnesses[i], costs[i,0], costs[i,1], costs[i,2]))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is R,C,Pumping = %s, total_conc = %s" % (best_ind, best_ind.fitness.values))

    # save results in text format
    with open('./%s/ga_results.txt' % (resultsdir), 'w') as fid:
        fid.write('generation,welx,wely,welp,avgFit,avgStd,minFit,MaxFit,cost1,cost2,cost3\n')
        for i in range(len(avgFit)):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
            i, bestWelx[i], bestWely[i], bestWelp[i], avgFit[i], avgStd[i], minFit[i], maxFit[i],bestCosts[i,0], bestCosts[i,1], bestCosts[i,2]))
    # }}}

    # }}}

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'GA optimization with conceptual'):  # {{{
    namemf = 'conceptual_12'
    namemt3d = 'conceptual_12'
    mf_ws = './conceptual/conceptual_12_MODFLOW_text/'
    mt_ws = './conceptual/conceptual_12_MT3DUSGS'

    # set working directory
    workdir = './conceptual/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0

    # depending on machine system, executio file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3d-usgs_1.1.0_64.exe'
    # depending on machine system, executio file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3dms='mt3dms'
    else: # window system.
        exe_name_mt3dms='mt3dms5b.exe'

    # remove existed GA results {{{
    f = workdir  # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)

    for f in glob.glob(workdir + '/GA*'):
        if os.path.isdir(f):
            shutil.rmtree(f)
    for f in glob.glob(workdir + '/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir + '/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 100
    perlen = [1 for i in range(100)]
    nstp = [1 for j in range(100)]
    steady = [False]

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 30   # number of generation for GA.
    npop = 100   # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load(namemf + '.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load(namemt3d + '.nam', version='mt3d-usgs',
            exe_name=exe_name_mt3d, model_ws=mt_ws,modflowmodel=copy.deepcopy(mf))

    # load model grid information
    nrow, ncol, nlay, nper = mf.get_nrow_ncol_nlay_nper()

    # update Dis for time domain.
    updateDis(mf, nper=100, perlen=perlen, nstp=nstp, steady=steady)

    # get xy grid.
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './conceptual/Data/grid.shp'
    polygons = geopandas.read_file(shpname)

    # load well locations.
    filename = './conceptual/Data/well_location.xlsx'
    df = pandas.read_excel(filename, sheet_name='source')
    welx = df['X']
    wely = df['Y']
    welz = np.zeros(np.shape(welx))
    welc, welr, well = flopyXyzToIndex(mf, welx, wely, welz)
    #print(welc)
    # get well index
    print('wel x,y shape = ', np.shape(welx))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_1_low = 0
    welp_1_up = 2000
    welp_2_low = 100
    welp_2_up = 2000
    toolbox.register("attr_welindx", np.random.randint, 0, np.shape(welc)[0]-1)
    toolbox.register("attr_welp_1", random.randint, welp_1_low, welp_1_up)
    toolbox.register("attr_welp_2", random.randint, welp_2_low, welp_2_up)

    # Structure initializers: define 'individual' to be an individual
    #                         consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welindx, toolbox.attr_welp_1, toolbox.attr_welp_2), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjFuncMoney2)

    # register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [0, welp_1_low, welp_2_low]
    up = [np.shape(welx)[0]-1, welp_1_up, welp_2_up]
    toolbox.register("mutate", tools.mutUniformInt, low=low, up=up, indpb=0.2)


    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=2)

    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    proc = []
    for i, ind in enumerate(pop):
        p = threading.Thread(target=evalObjFuncMoney2,
                             args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welc[ind[0]], welr[ind[0]], ind[1],
                                   ind[2], queue))
        proc.append(p)
        p.start()

    costs = np.zeros((npop, 4), dtype=float)
    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses[i], costs[i, :] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses = fitnesses[np.argsort(order)]
    costs     = costs[np.argsort(order),:]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses[i],)

    g = 0
    with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
        fid.write('index,welx,wely,welp_1,welp_2,total_cost,cost1,cost2,cost3,cost4\n')
        for i in range(npop):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], fitnesses[i], costs[i,0], costs[i,1], costs[i,2],costs[i,3]))

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    best_ind = tools.selBest(pop, 1)[0]

    print("  Evaluated %i individuals" % len(pop))
    avgFit   = [sum(fits)/len(fits)]
    avgStd   = [std]
    minFit   = [min(fits)]
    maxFit   = [max(fits)]
    bestWelx = [welx[best_ind[0]]]
    bestWely = [wely[best_ind[0]]]
    bestWelp_1 = [best_ind[1]]
    bestWelp_2 = [best_ind[2]]
    bestCosts  = np.zeros((NGEN+1,4),dtype=float)
    bestCosts[0,:]  = costs[np.argmin(fits),:]


    # Begin the evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            print('ind[0], ind[1], ind[2] = %d, %f, %f'%(ind[0], ind[1], ind[2]))
            print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2]))
            p = threading.Thread(target=evalObjFuncMoney2, args=(
                    copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welc[ind[0]],welr[ind[0]],ind[1],ind[2],queue))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses[i], costs[i,:] = queue.get()

        fitnesses = fitnesses[np.argsort(order)]
        costs     = costs[np.argsort(order),:]

        print('show fitnesses values')
        print(fitnesses)

        print('show values in pop.')
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        print("Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        print(pop)
        for i in range(npop):
            print(pop[i].fitness.values)
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        best_ind = tools.selBest(pop, 1)[0]

        avgFit.append(mean)
        avgStd.append(std)
        minFit.append(min(fits))
        maxFit.append(max(fits))

        bestWelx.append(welx[best_ind[0]])
        bestWely.append(wely[best_ind[0]])
        bestWelp_1.append(best_ind[1])
        bestWelp_2.append(best_ind[2])
        bestOrder = np.argmin(fits)
        #print('where is argmin of fits?',bestOrder, costs[bestOrder,:])
        bestCosts[g+1,:] = costs[np.argmin(fits),:]

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print(bestWelx, bestWely, bestWelp_1, bestWelp_2)

        # save results in text format
        with open('./%s/ga_gen%03d.txt' % (resultsdir, g + 1), 'w') as fid:
            fid.write('index,welx,wely,welp_1,welp_2,total_cost,cost1,cost2,cost3,cost4\n')
            for i in range(npop):
                fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], fitnesses[i], costs[i,0], costs[i,1], costs[i,2],costs[i,3]))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is R,C,Pumping = %s, total_conc = %s" % (best_ind, best_ind.fitness.values))

    # save results in text format
    with open('./%s/ga_results.txt' % (resultsdir), 'w') as fid:
        fid.write('generation,welx,wely,welp_1,welp_2,avgFit,avgStd,minFit,MaxFit,cost1,cost2,cost3,cost4\n')
        for i in range(len(avgFit)):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
            i, bestWelx[i], bestWely[i], bestWelp_1[i], bestWelp_2[i], avgFit[i], avgStd[i], minFit[i], maxFit[i],bestCosts[i,0], bestCosts[i,1], bestCosts[i,2],bestCosts[i,3]))
    # }}}

    # }}}

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'GA optimization with genetic_algorithm'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './genetic_algorithm/mf_MODFLOW_text/'
    mt_ws = './genetic_algorithm/mf_MT3DUSGS'

    # set working directory
    workdir = './genetic_algorithm/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3d-usgs_1.1.0_64.exe'


    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)

    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    if 1:
        nper = 1
        perlen = 1820
        nstp = 200
        steady = False

        CXPB = 0.8  # probability of cross in genes.
        MUTPB = 0.1  # probability of mutation
        NGEN = 200   # number of generation for GA.
        npop = 100   # number of population of GA model.
    else:
        GAoptions = readGAoptions(filename='./genetic_algorithm/Data/GAoptions.csv')
        nper = 1
        perlen = GAoptions['perlen']
        nstp = GAoptions['nstp']

        CXPB = GAoptions['cxpb']  # probability of cross in genes.
        MUTPB = GAoptions['mutpb']  # probability of mutation
        NGEN = GAoptions['ngen']  # number of generation for GA.
        npop = GAoptions['npop']  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('mf.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load('mf.nam', version='mt3d-usgs',
            exe_name=exe_name_mt3d, model_ws=mt_ws,modflowmodel=copy.deepcopy(mf))


    # update Dis for time domain.
    updateDis(mf, nper=1, perlen=perlen, nstp=nstp, steady=steady)

    # get xy grid.
    xg, yg = flopyGetXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './genetic_algorithm/Data/grid.shp'
    polygons = geopandas.read_file(shpname)

    # load well locations.
    filename = './genetic_algorithm/Data/well_location.xlsx'
    df = pandas.read_excel(filename, sheet_name='source')
    welx = df['X']
    wely = df['Y']
    welz = np.zeros(np.shape(welx))
    welc, welr, well = flopyXyzToIndex(mf,welx,wely,welz)

    # get well index
    print('wel x,y shape = ', np.shape(welx))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_1_low = 1
    welp_1_up = 2000
    welp_2_low = 250
    welp_2_up = 1000
    welp_3_low = 250
    welp_3_up = 1000
    toolbox.register("attr_welindx", np.random.randint, 0, np.shape(welc)[0]-1)
    toolbox.register("attr_welp_1", random.randint, welp_1_low, welp_1_up)
    toolbox.register("attr_welp_2", random.randint, welp_2_low, welp_2_up)
    toolbox.register("attr_welp_3", random.randint, welp_3_low, welp_3_up)

    # Structure initializers: define 'individual' to be an individual
    #                         consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welindx, toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjFuncMoney2)

    # register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [0, welp_1_low, welp_2_low, welp_3_low]
    up = [np.shape(welx)[0]-1, welp_1_up, welp_2_up, welp_3_up]
    toolbox.register("mutate", tools.mutUniformInt, low=low, up=up, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=2)

    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    proc = []
    for i, ind in enumerate(pop):
        p = threading.Thread(target=evalObjFuncMoney2,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3], weight_conc, queue))
        proc.append(p)
        p.start()

    costs = np.zeros((npop,3),dtype=float)
    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses[i], costs[i,:] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses = fitnesses[np.argsort(order)]
    costs     = costs[np.argsort(order),:]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses[i],)

    g = 0
    with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
        fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
        for i in range(npop):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses[i], costs[i,0], costs[i,1], costs[i,2]))

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    best_ind = tools.selBest(pop, 1)[0]

    print("  Evaluated %i individuals" % len(pop))
    avgFit   = [sum(fits)/len(fits)]
    avgStd   = [std]
    minFit   = [min(fits)]
    maxFit   = [max(fits)]
    bestWelx = [welx[best_ind[0]]]
    bestWely = [wely[best_ind[0]]]
    bestWelp_1 = [best_ind[1]]
    bestWelp_2 = [best_ind[2]]
    bestWelp_3 = [best_ind[3]]
    bestCosts  = np.zeros((NGEN+1,3),dtype=float)
    bestCosts[0,:]  = costs[np.argmin(fits),:]

    # Begin the evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=evalObjFuncMoney2, args=(
                    copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welc[ind[0]],welr[ind[0]],ind[1],ind[2],ind[3],weight_conc,queue))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses[i], costs[i,:] = queue.get()

        fitnesses = fitnesses[np.argsort(order)]
        costs     = costs[np.argsort(order),:]

        print('show fitnesses values')
        print(fitnesses)

        print('show values in pop.')
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        print("Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        print(pop)
        for i in range(npop):
            print(pop[i].fitness.values)
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        best_ind = tools.selBest(pop, 1)[0]

        avgFit.append(mean)
        avgStd.append(std)
        minFit.append(min(fits))
        maxFit.append(max(fits))

        bestWelx.append(welx[best_ind[0]])
        bestWely.append(wely[best_ind[0]])
        bestWelp_1.append(best_ind[1])
        bestWelp_2.append(best_ind[2])
        bestWelp_3.append(best_ind[3])
        bestOrder = np.argmin(fits)
        #print('where is argmin of fits?',bestOrder, costs[bestOrder,:])
        bestCosts[g+1,:] = costs[np.argmin(fits),:]

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print(bestWelx, bestWely, bestWelp_1, bestWelp_2, bestWelp_3)

        # save results in text format
        with open('./%s/ga_gen%03d.txt' % (resultsdir, g + 1), 'w') as fid:
            fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_mass,cost1,cost2,cost3\n')
            for i in range(npop):
                fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses[i], costs[i,0], costs[i,1], costs[i,2]))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is R,C,Pumping = %s, total_conc = %s" % (best_ind, best_ind.fitness.values))

    # save results in text format
    with open('./%s/ga_results.txt' % (resultsdir), 'w') as fid:
        fid.write('generation,welx,wely,welp_1,welp_2,welp_3,avgFit,avgStd,minFit,MaxFit,cost1,cost2,cost3\n')
        for i in range(len(avgFit)):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
            i, bestWelx[i], bestWely[i], bestWelp_1[i], bestWelp_2[i], bestWelp_3[i],
            avgFit[i], avgStd[i], minFit[i], maxFit[i],bestCosts[i,0], bestCosts[i,1], bestCosts[i,2]))
    # }}}

    # }}}

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './NSGA_222/mf_MODFLOW_text/'
    mt_ws = './NSGA_222/mf_MT3DUSGS/'

    # set working directory
    workdir = './NSGA_33/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 201
    perlen = [5 for i in range(201)]
    nstp = [1 for j in range(201)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 40   # number of generation for GA.
    npop = 100  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('ND_model_2.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load('ND_model_2.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 174459.425
    fdg.Grid.yoffset = 531413.605

    # Get xy grid
    xg, yg = flopyGetXyzGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2/Data_2/ContaminantArea/Boundary_Boundary_polys.shp'
    polygons = geopandas.read_file(shpname)

    # load well locations
    filename_1 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_2 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_3 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_4 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_5 = './NSGA_222/Data_2/Well/well_location.xlsx'
    df_1 = pandas.read_excel(filename_1, sheet_name='source')
    df_2 = pandas.read_excel(filename_2, sheet_name='source')
    df_3 = pandas.read_excel(filename_3, sheet_name='source')
    df_4 = pandas.read_excel(filename_4, sheet_name='source')
    df_5 = pandas.read_excel(filename_5, sheet_name='source')
    welx_1 = df_1['X']
    wely_1 = df_1['Y']
    welx_2 = df_2['X']
    wely_2 = df_2['Y']
    welx_3 = df_3['X']
    wely_3 = df_3['Y']
    welx_4 = df_4['X']
    wely_4 = df_4['Y']
    welx_5 = df_5['X']
    wely_5 = df_5['Y']
    welz = np.zeros(np.shape(welx_1))
    welc_1, welr_1, well = flopyXyzToIndex(mf, welx_1, wely_1, welz)
    welc_2, welr_2, well = flopyXyzToIndex(mf, welx_2, wely_2, welz)
    welc_3, welr_3, well = flopyXyzToIndex(mf, welx_3, wely_3, welz)
    welc_4, welr_4, well = flopyXyzToIndex(mf, welx_4, wely_4, welz)
    welc_5, welr_5, well = flopyXyzToIndex(mf, welx_5, wely_5, welz)
    # get well index
    print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_1_low = 1
    welp_1_up = 60
    welp_2_low = 1
    welp_2_up = 60
    welp_3_low = 1
    welp_3_up = 60
    welp_4_low = 1
    welp_4_up = 60
    welp_5_low = 1
    welp_5_up = 60
    welp_6_low = 1
    welp_6_up = 60
    welp_7_low = 1
    welp_7_up = 60
    welp_8_low = 1
    welp_8_up = 60
    welp_9_low = 1
    welp_9_up = 60
    welp_10_low = 1
    welp_10_up = 60
    welp_13_low = 1
    welp_13_up = 60
    welp_14_low = 1
    welp_14_up = 60
    welp_15_low = 1
    welp_15_up = 60
    welp_16_low = 1
    welp_16_up = 60
    welp_17_low = 1
    welp_17_up = 60
    welp_18_low = 1
    welp_18_up = 60
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_1_low, welp_1_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_2_low, welp_2_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_3_low, welp_3_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_4_low, welp_4_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_5_low, welp_5_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_6_low, welp_6_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_7_low, welp_7_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_8_low, welp_8_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_9_low, welp_9_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_10_low, welp_10_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_13_low, welp_13_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_14_low, welp_14_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_15_low, welp_15_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_16_low, welp_16_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_17_low, welp_17_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_18_low, welp_18_up)

    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_1_low, welp_2_low, welp_3_low, welp_4_low, welp_5_low, welp_6_low, welp_7_low, welp_8_low, welp_9_low, welp_10_low, welp_13_low, welp_14_low, welp_15_low, welp_16_low, welp_17_low, welp_18_low]
    up = [welp_1_up, welp_2_up, welp_3_up, welp_4_up, welp_5_up, welp_6_up, welp_7_up, welp_8_up, welp_9_up, welp_10_up, welp_13_up, welp_14_up, welp_15_up, welp_16_up, welp_17_up, welp_18_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i]= ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14],ind[15], queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i] = ind[:]
        # welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
        data = np.transpose([welp1_, welp2_, welp3_, welp4_, welp5_, welp6_, welp7_, welp8_, welp9_, welp10_, welp13_,welp14_, welp15_, welp16_, welp17_, welp18_, fitnesses1, fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1', 'welp_2', 'welp_3', 'welp_4', 'welp_5', 'welp_6', 'welp_7', 'welp_8',
                                       'welp_9', 'welp_10', 'welp_13', 'welp_14', 'welp_15',
                                       'welp_16', 'welp_17', 'welp_18', 'cost1(money)', 'cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Plot_NSGA2_pareto_front'):
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #for id in [50, 100, 150, 200]:
    cost1_ = []
    cost2_ = []
    for id in [60, 70, 80, 90, 100]:
        df = pandas.read_csv('./NSGA_2222/Results/ga_gen%03d.csv'%(id))
        #subset = df.loc[:, ['welp_1','welp_2','welp_3','cost1(money)', 'cost2(conc)']]
        #print(subset)
        #df2 = df.loc[subset['cost1(money)'] == '22138208.0']
        #print(df2)
        cost1 = df['cost1(money)'].to_numpy()
        cost2 = df['cost2(conc)'].to_numpy()
        cost1_.append(cost1)
        cost2_.append(cost2)
        ax.scatter(cost1,cost2,label='gen=%d'%(id))
    plt.figure(1, figsize=(8.0, 8.0), dpi=300)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel('Money ($)', fontdict={'family': 'Times New Roman', 'color':'black', 'weight': 'bold', 'size':20})
    ax.set_ylabel('Total mass (mg)', fontdict={'family': 'Times New Roman', 'color':'black', 'weight': 'bold', 'size':20})
    ax.legend()
    plt.show()
    #plt.savefig('Convergence of pareto optimal front')
    '''
    cost1_ = np.array(cost1_)
    cost2_ = np.array(cost2_)
    cost1_ = cost1_.reshape((-1,))
    cost2_ = cost2_.reshape((-1,))
    print(np.shape(cost1_), np.shape(cost2_))
    c1 = []
    c2 = []
    x = np.linspace(0, 3.0e+7, 90)
    for x1,x2 in zip(x[:-2], x[1:]):
        pos = np.where((cost1_ < x2) & (cost1_ >= x1))
        if np.any(pos):
            #print(x1, x2)
            c2.append(np.min(cost2_[pos]))
            #print(np.min(cost2_[pos]))
            pos2 = np.argmin(cost2_[pos])
            #print(pos)
            #print(pos2)
            c1.append(cost1_[pos[0][pos2]])
    c1 = np.array(c1)
    c2 = np.array(c2)
    #cc1 = c1.reshape((75,1))
    #cc2 = c2.reshape((75,1))
    #print(np.shape(c1), np.shape(c2))
    #print(cc1, cc2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.scatter(cost1_,cost2_)
    ax.scatter(c1,c2,label='gen=%d'%(id))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Money ($)', labelpad=10, fontdict={'family': 'Times New Roman', 'color':'black', 'weight': 'bold', 'size':20})
    plt.ylabel('Total Mass (mg)', labelpad=10, fontdict={'family': 'Times New Roman', 'color':'black', 'weight': 'bold', 'size':20})
    #ax.legend()
    plt.show()
    '''
#}}}5
if perform(org, 'Plot GA results.'): # {{{
    data = pandas.read_csv('./genetic_algorithm/GA_result.csv')
    generation = data['generation']
    Total_cost = data['Pumping cost ($)']
    plt.plot(generation, Total_cost, c='blue', linewidth='3')
    plt.title('Pumping cost', fontdict={'family': 'Times New Roman', 'color':'black', 'weight': 'bold', 'size':20})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Generation', labelpad=10, fontdict={'family': 'Times New Roman', 'color':'black', 'weight': 'bold', 'size':15})
    plt.ylabel('Pumping cost ($)', labelpad=10, fontdict={'family': 'Times New Roman', 'color':'black', 'weight': 'bold', 'size':15})
    plt.show()
#}}}
if perform(org, 'NSGA-2 optimization with conceptual model'): # {{{
    namemf = 'conceptual_1'
    namemt3d = 'conceptual_1'
    mf_ws = './conceptual/conceptual_1_MODFLOW_text/'
    mt_ws = './conceptual/conceptual_1_MT3DUSGS'

    # set working directory
    workdir = './conceptual/'
    resultsdir = workdir + '/Results/'
    # depending on machine system, execution file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3d-usgs_1.1.0_64.exe'
    # depending on machine system, execution file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3dms='mt3dms'
    else: # window system.
        exe_name_mt3dms='mt3dms5b.exe'

    # remove existed GA results {{{
    f = workdir  # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)

    for f in glob.glob(workdir + '/GA*'):
        if os.path.isdir(f):
            shutil.rmtree(f)
    for f in glob.glob(workdir + '/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir + '/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 100
    perlen = [1 for i in range(100)]
    nstp = [1 for j in range(100)]
    steady = [False]

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 100   # number of generation for GA.
    npop = 100   # number of population of GA model.

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(7)

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load(namemf + '.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load(namemt3d + '.nam', version='mt3d-usgs',
            exe_name=exe_name_mt3d, model_ws=mt_ws,modflowmodel=copy.deepcopy(mf))

    # load model grid information
    nrow, ncol, nlay, nper = mf.get_nrow_ncol_nlay_nper()

    # update Dis for time domain.
    updateDis(mf, nper=nper, perlen=perlen, nstp=nstp, steady=steady)

    # get xy grid.
    xg, yg = flopyGetXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './conceptual/Data_2/grid.shp'
    polygons = geopandas.read_file(shpname)

    # load well locations.
    filename_1 = './conceptual/Data_2/well_location.xlsx'
    filename_2 = './conceptual/Data_2/well_location_2.xlsx'
    df_1 = pandas.read_excel(filename_1, sheet_name='source')
    df_2 = pandas.read_excel(filename_2, sheet_name='source')
    welx_1 = df_1['X']
    wely_1 = df_1['Y']
    welx_2 = df_2['X']
    wely_2 = df_2['Y']
    welz = np.zeros(np.shape(welx_1))
    welc_1, welr_1, well = flopyXyzToIndex(mf, welx_1, wely_1, welz)
    welc_2, welr_2, well = flopyXyzToIndex(mf, welx_2, wely_2, welz)
    #print(welc)
    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_1_low = 1
    welp_1_up = 2000
    welp_2_low = 1
    welp_2_up = 1000
    welp_3_low = 1
    welp_3_up = 1000
    #def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        #r = random.randint(nrow_low,nrow_up)
        #c = random.randint(ncol_low, ncol_up)
        #return r,c
    toolbox.register("attr_welindx_1", np.random.randint, 0, np.shape(welc_1)[0]-1)
    toolbox.register("attr_welindx_2", np.random.randint, 0, np.shape(welc_2)[0]-1)
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_1_low, welp_1_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_2_low, welp_2_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_3_low, welp_3_up)

    # Structure initializers: define 'individual' to be an individual
    #                         consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welindx_1, toolbox.attr_welindx_2, toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_2)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]

            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)

        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])
        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [0, 0, welp_1_low, welp_2_low, welp_3_low]
    up = [np.shape(welx_1)[0]-1, np.shape(welx_2)[0]-1, welp_1_up, welp_2_up, welp_3_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0.1)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0.1, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welc_1[ind[0]], welr_1[ind[0]], welc_2[ind[1]], welr_2[ind[1]], ind[2], ind[3], ind[4], queue,processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    weli1_  = np.zeros((npop,))
    weli2_  = np.zeros((npop,))
    welx1_  = np.zeros((npop,))
    wely1_  = np.zeros((npop,))
    welx2_  = np.zeros((npop,))
    wely2_ = np.zeros((npop,))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        weli1_[i], weli2_[i], welp1_[i], welp2_[i], welp3_[i] = ind[:]
        welx1_[i], wely1_[i], welx2_[i], wely2_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]]
    data = np.transpose([welx1_,wely1_,welx2_,wely2_,welp1_,welp2_,welp3_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welx_1','wely_1','welx_2','wely_2','welp_1','welp_2','welp_3','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                #del child1.fitness.values
                #del child2.fitness.values

        #for mutant in offspring:
            # mutate an individual with probability MUTPB
            #toolbox.mutate(mutant)
            #del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                            args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welc_1[ind[0]], welr_1[ind[0]], welc_2[ind[1]], welr_2[ind[1]], ind[2], ind[3], ind[4],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)

        print('   save results.')
        weli1_[i], weli2_[i], welp1_[i], welp2_[i], welp3_[i] = ind[:]
        welx1_[i], wely1_[i], welx2_[i], wely2_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]]
        data = np.transpose([welx1_,wely1_,welx2_,wely2_,welp1_,welp2_,welp3_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data,columns=['welx_1','wely_1','welx_2','wely_2','welp_1','welp_2','welp_3','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Gif file for transport model'):
    namemf = 'mf'
    namemt3d = 'mt'
    model_ws = './conceptual/GA18/'
    debug = 0
    #load flow model
    mf = flopy.modflow.Modflow.load(namemf+'.nam', version='mf2005', exe_name='mf2005', model_ws=model_ws)
    #load head file
    data = flopy.utils.binaryfile.UcnFile(model_ws+'/MT3D001.UCN')
    times = data.get_times()
    conc = data.get_data(totim=times[-1])

    #plot result
    plotFlopy3d(mf,conc,layer=0,title='Nitrate')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("x(m)", size=15,)
    plt.ylabel("y(m)", size=15,)
    plt.title("Nitrate",
              fontdict={'family': 'Times New Roman',
                        'color' : 'black',
                        'weight': 'bold',
                        'size': 24})
    plt.show()
    #Gif making
#}}}
if perform(org, 'Plot figure'):
    namemf = 'mf'
    namemt3d = 'mt'
    model_ws_1 = './NSGA_33/GA44'
    model_ws_2 = './NSGA_33/GA41'
    model_ws_3 = './NSGA_33/GA33'
    model_ws_4 = './NSGA_33/GA60'
    # Load model
    mf = flopy.modflow.Modflow.load(namemf + '.mfn', version='mf2005', exe_name='mf2005', model_ws=model_ws_4)
    data_1 = flopy.utils.binaryfile.UcnFile(model_ws_1+'/MT3D001.UCN')
    data_2 = flopy.utils.binaryfile.UcnFile(model_ws_1+'/MT3D002.UCN')
    data_3 = flopy.utils.binaryfile.UcnFile(model_ws_2+'/MT3D001.UCN')
    data_4 = flopy.utils.binaryfile.UcnFile(model_ws_2+'/MT3D002.UCN')
    #data_2 = flopy.utils.binaryfile.UcnFile(model_ws_1+'/MT3D002.UCN')
    #times = data.get_times()
    # Plot multi-component transport model result -> Time step is 200 days
    os.chdir('./NSGA_33/Minimum_TCE')
    #conc_1  = data_1.get_data(totim=1740.0, mflay=0)
    #conc_2  = data_2.get_data(totim=1740.0, mflay=0)
    conc_3  = data_4.get_data(totim=200.0, mflay=1)
    conc_5  = data_4.get_data(totim=400.0, mflay=1)
    conc_6  = data_4.get_data(totim=600.0, mflay=1)
    conc_7  = data_4.get_data(totim=800.0, mflay=1)
    conc_8  = data_4.get_data(totim=1005.0, mflay=1)
    #conc_9  = data_3.get_data(totim=1500.0, mflay=0)
    # Plot and save png file
    #plotFlopy3d(mf, conc_1, title='TCE(test)', fontsize=15, colorbartitle='TCE (mg/L)', caxis=[0, 1])
    #plotFlopy3d(mf, conc_3, title='TCE(minimum)_200 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 3])
    #plotFlopy3d(mf, conc_5, title='TCE(minimum)_400 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 3])
    #plotFlopy3d(mf, conc_6, title='TCE(minimum)_600 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 3])
    #plotFlopy3d(mf, conc_7, title='TCE(minimum)_800 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 3])
    plotFlopy3d(mf, conc_8, title='TCE(minimum)_final', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 3])
    #plotFlopy3d(mf, conc_3, title='TCE(maximum)_scenario_1_final', fontsize=22, colorbartitle='TCE (mg/L)', caxis=[0, 3])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("x(m)", labelpad=12, size=18)
    plt.ylabel("y(m)", labelpad=12, size=18)
    #plt.savefig('TCE(minimum)_200 days', dpi=1000)
    #plt.savefig('TCE(minimum)_400 days', dpi=1000)
    #plt.savefig('TCE(minimum)_600 days', dpi=1000)
    #plt.savefig('TCE(minimum)_800 days', dpi=1000)
    plt.savefig('TCE(minimum)_final', dpi=1000)
    #plt.savefig('TCE(maximum)_scenario_1_final', dpi=1000)
    #plt.show()
#}}}
if perform(org, 'Plot Pareto optimal front'):
    def pareto_frontier(Xs, Ys, maxX=True, maxY=True):
        # Sort the list in either ascending or descending order of X
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]
        # Loop through the sorted list
        for pair in myList[1:]:
            if maxY:
                if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                    p_front.append(pair)  # … and add them to the Pareto frontier
            else:
                if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                    p_front.append(pair)  # … and add them to the Pareto frontier
        # Turn resulting pairs back into a list of Xs and Ys
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return p_frontX, p_frontY

    df = pandas.read_csv('./NSGA_3_Wonju_scenario_1/Results/ga_gen100.csv')

    Xs = df['cost1(money)']
    Ys = df['cost2(conc)']
    # Find lowest values for cost and highest for savings
    p_front = pareto_frontier(Xs, Ys, maxX=False, maxY=False)
    # Plot a scatter graph of all results
    plt.scatter(Xs, Ys)
    plt.xlabel('Cost(KRW)', labelpad=10, fontdict={'family': 'Times New Roman',
                                                   'color': 'black',
                                                   'weight': 'bold',
                                                   'size': 16})
    plt.ylabel('Concentration (mg/L)', labelpad=10, fontdict={'family': 'Times New Roman',
                                                              'color': 'black',
                                                              'weight': 'bold',
                                                              'size': 16})
    plt.xticks(size=16, linespacing=1.0, color='black')
    plt.yticks(size=16, linespacing=1.0, color='black')
    # Then plot the Pareto frontier on top
    #plt.plot(p_front[0], p_front[1])
    #print(p_front[0], p_front[1])
    p1 = np.reshape(p_front[0], (35,1))
    p2 = np.reshape(p_front[1], (35,1))
    plt.figure(2, dpi=300)
    plt.scatter(p1, p2)
    plt.title('Pareto optimal front', pad=10, fontdict={'family': 'Times New Roman',
                        'color': 'black',
                        'weight': 'bold',
                        'size': 20})
    plt.xlabel('Cost(KRW)', labelpad=10, fontdict= {'family' : 'Times New Roman',
                        'color': 'black',
                        'weight': 'bold',
                        'size': 16})
    plt.ylabel('Concentration (mg/L)', labelpad=10, fontdict= {'family' : 'Times New Roman',
                        'color': 'black',
                        'weight': 'bold',
                        'size': 16})
    plt.xticks(size=16, linespacing=1.0, color = 'black')
    plt.yticks(size=16, linespacing=1.0, color = 'black')

    #plt.plot(p1, p2)
    print(p1)
    print(p2)
    #plt.savefig('Pareto optimal front')
    plt.show()
#}}}10
if perform(org, 'Generate hypervolume'):
    # load file
    df = pandas.read_csv('./conceptual/Results/ga_gen100.csv')
    # Define pareto optimal front
    def pareto_frontier(Xs, Ys, maxX=True, maxY=True):
        # Sort the list in either ascending or descending order of X
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]
        # Loop through the sorted list
        for pair in myList[1:]:
            if maxY:
                if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                    p_front.append(pair)  # … and add them to the Pareto frontier
            else:
                if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                    p_front.append(pair)  # … and add them to the Pareto frontier
        # Turn resulting pairs back into a list of Xs and Ys
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return p_frontX, p_frontY

    Xs = df['cost1(money)']
    Ys = df['cost2(conc)']
    # Find lowest values for cost and highest for savings
    p_front = pareto_frontier(Xs, Ys, maxX=False, maxY=False)
    print(p_front[0])
    print(p_front[1])


    # Set reference point in pareto optimal front
    referencePoint = [0, 0]
    hv = hypervolume.HyperVolume(referencePoint=referencePoint)
    df_1 = hypervolume.Hypervolume.compute()
#}}}
if perform(org, 'PNG to GIF'):
    # Create the frames
    frames = []
    imgs = glob.glob('./NSGA_22222_Wonju_scenario_2/gifmaking/*.png')
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    # Change the filepath for GIF
    # Save into a GIF file that loops forever
    frames[0].save('TCE_scenario_2.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=300, loop=0)
#}}}
if perform(org, 'Test to use MODFLOW6 including surface water flow'):
    model_name = "tutorial01_mf1"
    # Create the flopy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name = model_name, exe_name="mf6", version="mf6", sim_ws=".")
    # Create the flopy tdis object (Simulation time definition, perioddata includes perlen, time step, multiplier)
    time = [(10.0, 1, 1.0), (10.0, 1, 1.0), (10.0, 1, 1.0), (10.0, 1, 1.0)]
    tdis = flopy.mf6.ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=4, perioddata=time)
    # Create the iterative model solution (Kind of defining solver and define the model's linearity)
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="MODERATE",
        linear_acceleration="BICGSTAB")
    # Create the flopy groundwater flow (gwf) model object
    model_nam_file = f"{model_name}.nam"
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=model_name,
        model_nam_file=model_nam_file,
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",)
    # Create the DIS package
    #bot = np.linspace(-H / Nlay, -H, Nlay)
    #delrow = delcol = L / (N - 1)
    vertices = []
    '''disu = flopy.mf6.ModflowGwfdisu(
        gwf,
        length_units = "METERS",
        xorigin=0.0,
        yorigin=0.0,
        nodes=300,
        nlay=3,
        ncpl=10,
        nvert=10,
        top=50.0,
        botm=-50.0,)'''
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=3,
        nrow=10,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=50.0,
        botm=[5.0, -10.0, -50.0],
        filename="{}.dis".format(model_name))
    # Create the initial conditions package
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=50.0, filename="{}.ic".format(model_name))
    # Create the constant head package
    chd_rec = []
    layer = 0
    for row_col in range (0, 10):
        chd_rec.append(((layer, row_col, 0), 50))
        chd_rec.append(((layer, row_col, 9), 60))
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_rec,
    )
    # Create the node property flow package
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        icelltype=[1, 0, 0],
        k=[5.5, 1.0, 4.0],
        k33=[0.5, 0.1, 0.4])
    # Storage package
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        save_flows=True,
        iconvert=0,
        ss=[1.0e-4, 1.0e-5, 1.0e-4],
        sy=[0.2, 0.05, 0.1],
        steady_state={0:True},
        transient={1:True}
    )
    # Create the well package (L, R, C, -Q)
    stress_period_data = [(0, 1, 3, -100)]
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=stress_period_data)
    # Evapotranspiration
    # Stress period data includes cellid, elevation surface, ET flux rate, ET depth, proportion of ET depth, pxdp, petm, petm0, aux, boundname
    evt_period = flopy.mf6.ModflowGwfevt.stress_period_data.empty(gwf, 150, nseg=3)
    for col in range (0, 10):
        for row in range (0, 10):
            evt_period[0][col*15+row] = (
                (0, row, col),
                50.0,
                0.04,
                10.0,
                0.2,
                0.5,
                0.3,
                0.1,
                None,
            )
    evt = flopy.mf6.ModflowGwfevt(
        gwf,
        pname = "evt",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=150,
        nseg=3,
        stress_period_data=evt_period,
        filename = "{}.evt".format(model_name)
    )
    # Create River package
    riv_period = {}
    riv_period_array = [
        ((0, 2, 0), 36.9, 1001.0, 35.9, None),
        ((0, 3, 1), 36.8, 1002.0, 35.8, None),
        ((0, 4, 2), 36.7, 1003.0, 35.7, None),
        ((0, 4, 3), 36.6, 1004.0, 35.6, None),
        ((0, 5, 4), 36.5, 1005.0, 35.5, None),
        ((0, 5, 5), 36.4, 1006.0, 35.4, None),
        ((0, 5, 6), 36.3, 1007.0, 35.3, None),
        ((0, 4, 7), 36.2, 1008.0, 35.2, None),
        ((0, 4, 8), 36.1, 1009.0, 35.1, None),
        ((0, 4, 9), 36.0, 1010.0, 35.0, None),
        ((0, 9, 0), 37.9, 1001.0, 36.9, None),
        ((0, 8, 1), 37.8, 1002.0, 36.8, None),
        ((0, 7, 2), 37.7, 1003.0, 36.7, None),
        ((0, 6, 3), 37.6, 1004.0, 36.6, None),
        ((0, 6, 4), 37.5, 1005.0, 36.5, None),
        ((0, 6, 7), 37.2, 1008.0, 36.2, None),
        ((0, 6, 8), 37.1, 1009.0, 36.1, None),
        ((0, 6, 9), 37.0, 1010.0, 36.0, None),
    ]
    riv_period[0] = riv_period_array
    riv = flopy.mf6.ModflowGwfriv(
        gwf,
        pname="riv",
        print_input=True,
        print_flows=True,
        save_flows="{}.cbc".format(model_name),
        boundnames=True,
        maxbound=20,
        stress_period_data=riv_period,
    )
    # Create the recharge package
    rch_sp1 = 0.001
    rch_sp2 = 0.005
    rch_spd = {0: rch_sp1, 1:rch_sp2}
    rch = flopy.mf6.ModflowGwfrcha(gwf, readasarrays=True, pname = "rch", print_input=True, recharge=rch_spd)
    # Create the output control package
    headfile = f"{model_name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{model_name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "LAST")]
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,)
    # Write the datasets
    sim.write_simulation()
    # Run the simulation
    success, buff = sim.run_simulation()
    if not success:
        raise Exception("MODFLOW 6 did not terminate normally.")
    # Post process head results
    h = gwf.output.head().get_data(kstpkper=(0, 0))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    # Create a model map
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    riv_quadmesh = modelmap.plot_bc("riv")
    linecollection = modelmap.plot_grid()
    #pa = modelmap.plot_array(h, vmin=40, vmax=60)
    contours = modelmap.contour_array(h[0])
    ax.clabel(contours, fmt="%2.1f")
    #cb = plt.colorbar(pa, shrink=0.5, ax=ax)
    plt.show()
    # Post process flow
    fname = os.path.join("{}.dis.grb".format(model_name))
    bgf = flopy.mf6.utils.MfGrdFile(fname)
    ia, ja = bgf.ia, bgf.ja
    flowja = gwf.output.budget().get_data(text="FLOW-JA-FACE")[0].squeeze()
    k = 1
    i = 1
    j = 3
    cell_nodes=gwf.modelgrid.get_node([(k, i, j)])
    for celln in cell_nodes:
        print("Printing flows for cell {}".format(celln))
        for ipos in range(ia[celln] + 1, ia[celln + 1]):
            cellm = ja[ipos]
            print(
                "Cell {} flow with cell {} is {}".format(
                    celln, cellm, flowja[ipos]
                )
            )
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_1'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/001/mf_MODFLOW_text/'
    mt_ws = './Stochastic/001/mt_MT3D'

    # set working directory
    workdir = './Stochastic/001/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic001.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}14
if perform(org, 'Optimization with NSGA-2 in Wonju site_2'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/002/mf_MODFLOW_text/'
    mt_ws = './Stochastic/002/mt_MT3D'

    # set working directory
    workdir = './Stochastic/002/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic002.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}15
if perform(org, 'Optimization with NSGA-2 in Wonju site_3'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/003/mf_MODFLOW_text/'
    mt_ws = './Stochastic/003/mt_MT3D'

    # set working directory
    workdir = './Stochastic/003/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic003.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}16
if perform(org, 'Optimization with NSGA-2 in Wonju site_4'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/004/mf_MODFLOW_text/'
    mt_ws = './Stochastic/004/mt_MT3D'

    # set working directory
    workdir = './Stochastic/004/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic004.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}17
if perform(org, 'Optimization with NSGA-2 in Wonju site_5'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/005/mf_MODFLOW_text/'
    mt_ws = './Stochastic/005/mt_MT3D'

    # set working directory
    workdir = './Stochastic/005/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic005.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}18
if perform(org, 'Optimization with NSGA-2 in Wonju site_6'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/006/mf_MODFLOW_text/'
    mt_ws = './Stochastic/006/mt_MT3D'

    # set working directory
    workdir = './Stochastic/006/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic006.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}19
if perform(org, 'Optimization with NSGA-2 in Wonju site_7'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/007/mf_MODFLOW_text/'
    mt_ws = './Stochastic/007/mt_MT3D'

    # set working directory
    workdir = './Stochastic/007/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic007.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}20
if perform(org, 'Optimization with NSGA-2 in Wonju site_8'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/008/mf_MODFLOW_text/'
    mt_ws = './Stochastic/008/mt_MT3D'

    # set working directory
    workdir = './Stochastic/008/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic008.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_9'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/009/mf_MODFLOW_text/'
    mt_ws = './Stochastic/009/mt_MT3D'

    # set working directory
    workdir = './Stochastic/009/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic009.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_10'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/010/mf_MODFLOW_text/'
    mt_ws = './Stochastic/010/mt_MT3D'

    # set working directory
    workdir = './Stochastic/010/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic010.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_11'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/011/mf_MODFLOW_text/'
    mt_ws = './Stochastic/011/mt_MT3D'

    # set working directory
    workdir = './Stochastic/011/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic011.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_12'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/012/mf_MODFLOW_text/'
    mt_ws = './Stochastic/012/mt_MT3D'

    # set working directory
    workdir = './Stochastic/012/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic012.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_13'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/013/mf_MODFLOW_text/'
    mt_ws = './Stochastic/013/mt_MT3D'

    # set working directory
    workdir = './Stochastic/013/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic013.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_14'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/014/mf_MODFLOW_text/'
    mt_ws = './Stochastic/014/mt_MT3D'

    # set working directory
    workdir = './Stochastic/014/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic014.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_15'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/015/mf_MODFLOW_text/'
    mt_ws = './Stochastic/015/mt_MT3D'

    # set working directory
    workdir = './Stochastic/015/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic015.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_16'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/016/mf_MODFLOW_text/'
    mt_ws = './Stochastic/016/mt_MT3D'

    # set working directory
    workdir = './Stochastic/016/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic016.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_17'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/017/mf_MODFLOW_text/'
    mt_ws = './Stochastic/017/mt_MT3D'

    # set working directory
    workdir = './Stochastic/017/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic017.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_18'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/018/mf_MODFLOW_text/'
    mt_ws = './Stochastic/018/mt_MT3D'

    # set working directory
    workdir = './Stochastic/018/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic018.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_19'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/019/mf_MODFLOW_text/'
    mt_ws = './Stochastic/019/mt_MT3D'

    # set working directory
    workdir = './Stochastic/019/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic019.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_20'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/020/mf_MODFLOW_text/'
    mt_ws = './Stochastic/020/mt_MT3D'

    # set working directory
    workdir = './Stochastic/020/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic020.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_21'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/021/mf_MODFLOW_text/'
    mt_ws = './Stochastic/021/mt_MT3D'

    # set working directory
    workdir = './Stochastic/021/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic021.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_22'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/022/mf_MODFLOW_text/'
    mt_ws = './Stochastic/022/mt_MT3D'

    # set working directory
    workdir = './Stochastic/022/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic022.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_23'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/023/mf_MODFLOW_text/'
    mt_ws = './Stochastic/023/mt_MT3D'

    # set working directory
    workdir = './Stochastic/023/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic023.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with NSGA-2 in Wonju site_24'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './Stochastic/024/mf_MODFLOW_text/'
    mt_ws = './Stochastic/024/mt_MT3D'

    # set working directory
    workdir = './Stochastic/024/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3dms5b.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 58
    perlen = [30 for i in range(108)]
    nstp = [1 for j in range(108)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 60   # number of generation for GA.
    npop = 50  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('stochastic024.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.mt.Mt3dms.load('stochastic.nam', version='mt3dms', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 282601.185
    fdg.Grid.yoffset = 530959.515

    # Get xy grid
    xg, yg = flopyUtils.getXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    polygons = geopandas.read_file(shpname)

    # get well index
    #print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 40
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_15", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_16", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_17", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_18", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_19", np.random.randint, welp_low, welp_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14, toolbox.attr_welp_15, toolbox.attr_welp_16, toolbox.attr_welp_17, toolbox.attr_welp_18, toolbox.attr_welp_19), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_3)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    welp15_ = np.zeros((npop,))
    welp16_ = np.zeros((npop,))
    welp17_ = np.zeros((npop,))
    welp18_ = np.zeros((npop,))
    welp19_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], ind[14], ind[15], ind[16], ind[17], ind[18],queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i], welp15_[i], welp16_[i], welp17_[i], welp18_[i], welp19_[i] = ind[:]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,welp15_,welp16_,welp17_,welp18_,welp19_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','welp_15','welp_16','welp_17','welp_18','welp_19','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Optimization with Jeju conceptual model using NSGA-ii'):
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './NSGA_22222/mf_MODFLOW_text/'
    mt_ws = './NSGA_22222/mf_MT3DUSGS/'

    # set working directory
    workdir = './NSGA_22222/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.

    if sys.platform == 'linux':  # linux system
        exe_name_mt3d = 'mt3d-usgs'
    else:  # window system.
        exe_name_mt3d = 'mt3d-usgs_1.1.0_64.exe'

    # remove existed GA results {{{
    f = workdir  # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir + '/GA*'):
        if os.path.isdir(f):
            shutil.rmtree(f)
    for f in glob.glob(workdir + '/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir + '/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 100
    perlen = [1 for i in range(100)]
    nstp = [1 for j in range(100)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 50  # number of generation for GA.
    npop = 100  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('conceptual_15.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load('conceptual_15.nam', version='mt3d-usgs', exe_name=exe_name_mt3d, model_ws=mt_ws,
                                modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    # mf.dis.nper = nper
    # mf.dis.perlen = perlen
    # mf.dis.nstp = nstp
    # mf.dis.steady = steady

    # Get xy grid
    xg, yg = flopyGetXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max
    #shpname = './NSGA_2222/Data/Boundary_new_1_polys.shp'
    #polygons = geopandas.read_file(shpname)

    # get well index
    # print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_1_low = 1
    welp_1_up = 1000
    welp_2_low = 1
    welp_2_up = 1000

    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low, nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r, c
    # toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_1_low, welp_1_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_2_low, welp_2_up)
    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual, (toolbox.attr_welp_1, toolbox.attr_welp_2), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_4)


    def mutBenchmark(individual, low, up, std_welidx, indpb):
        # well location
        if random.random() < indpb:
            r, c = individual[0]
            r = r + int(random.gauss(0, std_welidx))
            c = c + int(random.gauss(0, std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r, c)
        # update each pumping rate
        for i in range(1, len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i], up[i])
        return individual

    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_1_low, welp_2_low]
    up = [welp_1_up, welp_2_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    # low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    # up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                             args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i], fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    # offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    # for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    # with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    # print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i] = ind[:]
        # welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_, welp2_, fitnesses1, fitnesses2])
    df = pandas.DataFrame(data=data,
                          columns=['welp_1', 'welp_2', 'cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        # offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
            # fitness values of the children must be recalculated later
            # del child1.fitness.values
            # del child2.fitness.values

        # for mutant in offspring:
        # mutate an individual with probability MUTPB
        # toolbox.mutate(mutant)
        # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            # print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            # print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], queue, processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1, fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop + offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i] = ind[:]
        data = np.transpose([welp1_, welp2_, fitnesses1, fitnesses2])
        df = pandas.DataFrame(data=data,
                              columns=['welp_1', 'welp_2', 'cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g + 1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Load flow model'):
    namemf = 'surface_water_flow'
    model_ws_1 = './test/'
    model_ws_2 = './NSGA_222/mf_MT3DUSGS'
    # Load flow model
    mf = flopy.modflow.Modflow.load(namemf + '.mfn', version='mfusg', exe_name='mfusg', model_ws=model_ws_1)
    lpf = flopy.modflow.ModflowLpf.load(namemf + '.lpf', mf)
    print(lpf)
    '''data = flopy.utils.binaryfile.Headfile(model_ws_1 + 'surface_water_flow.hed')
    # Plot flow model
    #os.chdir('./NSGA_222')
    head = data.get_data()
    plotFlopy3d(mf, head, title='Flow model', fontsize=15, colorbartitle='Head (m)',caxis=[8, 10])
    #plt.savefig('.png', dpi=1000)
    plt.show()
    # Load PCE model
    data_1 = flopy.utils.binaryfile.UcnFile(model_ws_2 + '/MT3D001.UCN')
    data_2 = flopy.utils.binaryfile.UcnFile(model_ws_2 + '/MT3D002.UCN')
    # Plot concentration model
    conc_1 = data_1.get_data(totim=1000.0, mflay=1)
    conc_2 = data_2.get_data(totim=1000.0, mflay=1)
    #plotFlopy3d(mf, conc_1, title='PCE model', fontsize=15, colorbartitle='TCE (mg/L)', caxis=[0,40])
    #plt.show()'''
#}}}
if perform(org, 'Normal distribution'):
    '''np.random.seed(seed=1000)
    a = np.random.lognormal(2.3, 0.01, size=(204, 110))
    print(a)
    df = pandas.DataFrame(a)
    df.to_csv("normal_distribution_dispersivity.csv", index=False)
    b = np.random.lognormal(-7.6, 0.01, size=(204, 110))
    df_1 = pandas.DataFrame(b)
    df_1.to_csv("normal_distribution_layer2.csv", index=False)
    print(b)
    c = np.random.lognormal(-0.187, 0.7, size=(204,110))
    print(c)
    df_2 = pandas.DataFrame(c)
    df_2.to_csv("lognormal_distribution_K2.csv", index=False)'''
    np.random.seed(seed=0)
    d = np.random.poisson(0.8, size=(204,110))
    print(d)
    df = pandas.DataFrame(d)
    df.to_csv("Poisson_distribution_layer2.csv", index=False)
#}}}
if perform(org, 'GA includes Penalty function'):
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './NSGA_4/mf_MODFLOW_text/'
    mt_ws = './NSGA_4/mf_MT3DUSGS'

    # set working directory
    workdir = './NSGA_5/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0

    # depending on machine system, executio file should be changed.
    if sys.platform == 'linux':  # linux system
        exe_name_mt3d = 'mt3d-usgs'
    else:  # window system.
        exe_name_mt3d = 'mt3d-usgs_1.1.0_64.exe'

    # remove existed GA results {{{
    f = workdir  # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)

    for f in glob.glob(workdir + '/GA*'):
        if os.path.isdir(f):
            shutil.rmtree(f)
    for f in glob.glob(workdir + '/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir + '/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 120
    perlen = [30 for i in range(120)]
    nstp = [1 for j in range(120)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.85  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 30  # number of generation for GA.
    npop = 20  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('ND_model_4.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load('ND_model_4.nam', version='mt3d-usgs',
                                exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # load model grid information
    nrow, ncol, nlay, nper = mf.get_nrow_ncol_nlay_nper()

    # Update Dis for time domain.
    #UpdateDis(mf, nper=201, perlen=perlen, nstp=nstp, steady=steady)

    # Set origin of coordinate
    fdg.Grid.xoffset = 174459.425
    fdg.Grid.yoffset = 531413.605

    # get xy grid.
    xg, yg = flopyGetXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # Contaminant area min max
    #shpname = './NSGA_222/Data_2/ContaminantArea/Boundary_Boundary_polys.shp'
    #polygons = geopandas.read_file(shpname)

    # load well locations.
    filename = './NSGA_4/Data/Well/well_location.xlsx'
    df_1 = pandas.read_excel(filename, sheet_name='source')
    df_2 = pandas.read_excel(filename, sheet_name='source')
    df_3 = pandas.read_excel(filename, sheet_name='source')
    welx_1 = df_1['X']
    wely_1 = df_1['Y']
    welx_2 = df_2['X']
    wely_2 = df_2['Y']
    welx_3 = df_3['X']
    wely_3 = df_3['Y']
    welz_1 = np.zeros(np.shape(welx_1))
    welz_2 = np.zeros(np.shape(welx_2))
    welz_3 = np.zeros(np.shape(welx_3))
    welr_1, welc_1, well_1 = flopyXyzToIndex(mf, welx_1, wely_1, welz_1)
    welr_2, welc_2, well_2 = flopyXyzToIndex(mf, welx_2, wely_2, welz_2)
    welr_3, welc_3, well_3 = flopyXyzToIndex(mf, welx_3, wely_3, welz_3)
    #print(welr)
    #print(welc)
    # get well index
    print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 50
    toolbox.register("attr_welindx_1", np.random.randint, 0, np.shape(welc_1)[0]-1)
    toolbox.register("attr_welindx_2", np.random.randint, 0, np.shape(welc_2)[0]-1)
    toolbox.register("attr_welindx_3", np.random.randint, 0, np.shape(welc_3)[0]-1)
    toolbox.register("attr_welp_1", random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", random.randint, welp_low, welp_up)


    # Structure initializers: define 'individual' to be an individual
    #                         consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welindx_1, toolbox.attr_welindx_2, toolbox.attr_welindx_3, toolbox.attr_welp_1,toolbox.attr_welp_2,toolbox.attr_welp_3), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_5)

    # register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [0,0,0, welp_low,welp_low,welp_low]
    #low = np.matlib.repmat(low, 1, 1)
    up = [np.shape(welx_1)[0]-1,np.shape(welx_2)[0]-1,np.shape(welx_3)[0]-1, welp_up,welp_up,welp_up]
    #up = np.matlib.repmat(up, 1, 1)
    toolbox.register("mutate", tools.mutUniformInt, low=low, up=up, indpb=0.2)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=2)

    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    proc = []
    for i, ind in enumerate(pop):
        p = threading.Thread(target=evalObjNSGA_5,
                             args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welr_1[ind[0]], welc_1[ind[0]], welr_2[ind[1]], welc_2[ind[1]], welr_3[ind[2]], welc_3[ind[2]],ind[3],ind[4],ind[5], queue, processlock, ))
        proc.append(p)
        p.start()

    costs = np.zeros((npop, 3), dtype=float)
    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses[i], costs[i, :] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses = fitnesses[np.argsort(order)]
    costs = costs[np.argsort(order), :]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses[i],)

    g = 0
    with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
        fid.write('index,welx_1,wely_1,welx_2,wely_2,welx_3,wely_3,welp_1,welp_2,welp_3,total_cost,cost1,cost2,cost3\n')
        for i in range(npop):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
            i, welx_1[pop[i][0]], wely_1[pop[i][0]], welx_2[pop[i][1]], wely_2[pop[i][1]],welx_3[pop[i][2]], wely_3[pop[i][2]], pop[i][3], pop[i][4],pop[i][5],fitnesses[i], costs[i, 0], costs[i, 1], costs[i, 2]))

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    best_ind = tools.selBest(pop, 1)[0]

    print("  Evaluated %i individuals" % len(pop))
    avgFit = [sum(fits) / len(fits)]
    avgStd = [std]
    minFit = [min(fits)]
    maxFit = [max(fits)]
    bestWelx_1 = [welx_1[best_ind[0]]]
    bestWely_1 = [wely_1[best_ind[0]]]
    bestWelx_2 = [welx_2[best_ind[1]]]
    bestWely_2 = [wely_2[best_ind[1]]]
    bestWelx_3 = [welx_3[best_ind[2]]]
    bestWely_3 = [wely_3[best_ind[2]]]
    bestWelp_1 = [best_ind[3]]
    bestWelp_2 = [best_ind[4]]
    bestWelp_3 = [best_ind[5]]
    bestCosts = np.zeros((NGEN + 1, 3), dtype=float)
    bestCosts[0, :] = costs[np.argmin(fits), :]

    # Begin the evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1] = %d, %f' % (ind[0], ind[1]))
            #print('welc = %d, welr = %d, welp = %f' % (welr[ind[0]], welc[ind[0]], ind[1]))
            p = threading.Thread(target=evalObjNSGA_5,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, welr_1[ind[0]], welc_1[ind[0]], welr_2[ind[1]], welc_2[ind[1]], welr_3[ind[2]], welc_3[ind[2]],ind[3],ind[4],ind[5], queue, processlock, ))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses[i], costs[i, :] = queue.get()

        fitnesses = fitnesses[np.argsort(order)]
        costs = costs[np.argsort(order), :]

        print('show fitnesses values')
        #print(fitnesses)

        print('show values in pop.')
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        print("Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        print(pop)
        for i in range(npop):
            print(pop[i].fitness.values)
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        best_ind = tools.selBest(pop, 1)[0]

        avgFit.append(mean)
        avgStd.append(std)
        minFit.append(min(fits))
        maxFit.append(max(fits))

        bestWelx_1.append(welx_1[best_ind[0]])
        bestWely_1.append(wely_1[best_ind[0]])
        bestWelx_2.append(welx_2[best_ind[1]])
        bestWely_2.append(wely_2[best_ind[1]])
        bestWelx_3.append(welx_3[best_ind[2]])
        bestWely_3.append(wely_3[best_ind[2]])
        bestWelp_1.append(best_ind[3])
        bestWelp_2.append(best_ind[4])
        bestWelp_3.append(best_ind[5])
        bestOrder = np.argmin(fits)
        # print('where is argmin of fits?',bestOrder, costs[bestOrder,:])
        bestCosts[g + 1, :] = costs[np.argmin(fits), :]

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        # print(bestWelx, bestWely, bestWelp)

        # save results in text format
        with open('./%s/ga_gen%03d.txt' % (resultsdir, g + 1), 'w') as fid:
            fid.write('index,welx_1,wely_1,welx_2,wely_2,welx_3,wely_3,welp_1,welp_2,welp_3,total_cost,cost1,cost2,cost3\n')
            for i in range(npop):
                fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
                i, welx_1[pop[i][0]], wely_1[pop[i][0]], welx_2[pop[i][1]], wely_2[pop[i][1]], welx_3[pop[i][2]], wely_3[pop[i][2]], pop[i][3], pop[i][4], pop[i][5],fitnesses[i], costs[i, 0], costs[i, 1], costs[i, 2]))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is R,C,Pumping = %s, total_conc = %s" % (best_ind, best_ind.fitness.values))

    # save results in text format
    with open('./%s/ga_results.txt' % (resultsdir), 'w') as fid:
        fid.write('generation,welx_1,wely_1,welx_2,wely_2,welx_3,wely_3,welp_1,welp_2,welp_3,avgFit,avgStd,minFit,MaxFit,cost1,cost2,cost3\n')
        for i in range(len(avgFit)):
            fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
                i, bestWelx_1[i], bestWely_1[i], bestWelx_2[i], bestWely_2[i], bestWelx_3[i], bestWely_3[i], bestWelp_1[i], bestWelp_2[i], bestWelp_3[i], avgFit[i], avgStd[i], minFit[i], maxFit[i], bestCosts[i, 0], bestCosts[i, 1], bestCosts[i, 2]))
    # }}}

    # }}}

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}
if perform(org, 'Plot figure'):
    namemf = 'mf'
    namemt3d = 'mt'
    model_ws_1 = './NSGA_2222/mf_MODFLOW_text'
    model_ws_2 = './NSGA_22222_Wonju_scenario_2/GA33'
    model_ws_3 = './NSGA_3_Wonju_scenario_1/GA53'
    model_ws_4 = './NSGA_2222/mf_MT3DUSGS'
    # Load model
    mf = flopy.modflow.Modflow.load(namemf + '.nam', version='mf2005', exe_name='mf2005', model_ws=model_ws_3)
    data_1 = flopy.utils.binaryfile.UcnFile(model_ws_3+'/MT3D001.UCN')
    times = data_1.get_times()
    #data_2 = flopy.utils.binaryfile.HeadFile(model_ws_1+'/MT3D002.UCN')
    #times = data.get_times()
    # Plot multi-component transport model result -> Time step is 200 days
    os.chdir('./NSGA_3_Wonju_scenario_1/gifmaking/')
    #conc_1  = data_1.get_data(totim=1740.0, mflay=0)
    #conc_2  = data_2.get_data(totim=1740.0, mflay=0)
    #conc_0  = data_1.get_data(totim=5.0, mflay=1)
    conc_1  = data_1.get_data(totim=300.0, mflay=0)
    conc_2  = data_1.get_data(totim=600.0, mflay=0)
    conc_3  = data_1.get_data(totim=900.0, mflay=0)
    conc_4  = data_1.get_data(totim=1200.0, mflay=0)
    conc_5  = data_1.get_data(totim=1740.0, mflay=0)
    # Plot and save png file
    #plotFlopy3d(mf, conc_0, title='PCE(minimum)_initial', fontsize=15, colorbartitle='PCE (mg/L)', caxis=[0, 40])
    #plotFlopy3d(mf, conc_1, title='PCE(minimum)_200 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 40])
    #plotFlopy3d(mf, conc_2, title='PCE(minimum)_400 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 40])
    #plotFlopy3d(mf, conc_3, title='PCE(minimum)_600 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 40])
    #plotFlopy3d(mf, conc_4, title='PCE(minimum)_800 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 40])
    plotFlopy3d(mf, conc_5, title='TCE_1740 days', fontsize=20, colorbartitle='TCE (mg/L)', caxis=[0, 3.0])
    #plotFlopy3d(mf, head_2, title='head_initial', fontsize=20, colorbartitle='Head (m)', caxis=[100, 130])
    #plotFlopy3d(mf, head_3, title='head_2nd', fontsize=20, colorbartitle='Head (m)', caxis=[100, 130])
    #plotFlopy3d(mf, head_4, title='head_3rd', fontsize=20, colorbartitle='Head (m)', caxis=[100, 130])
    #plotFlopy3d(mf, head_5, title='head_4th', fontsize=20, colorbartitle='Head (m)', caxis=[100, 130])
    #plotFlopy3d(mf, head_6, title='head_final', fontsize=20, colorbartitle='Head (m)', caxis=[100, 130])
    #plotFlopy3d(mf, conc_3, title='TCE(maximum)_scenario_1_final', fontsize=22, colorbartitle='TCE (mg/L)', caxis=[0, 3])
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.xlabel("x(m)", labelpad=12, size=18)
    plt.ylabel("y(m)", labelpad=12, size=18)
    #plt.savefig('PCE(minimum)_initial', dpi=1000)
    #plt.savefig('PCE(minimum)_200 days', dpi=1000)
    #plt.savefig('PCE(minimum)_400 days', dpi=1000)
    #plt.savefig('PCE(minimum)_600 days', dpi=1000)
    #plt.savefig('PCE(minimum)_800 days', dpi=1000)
    plt.savefig('TCE_1740 days', dpi=600)
    #plt.savefig('head_0', dpi=1000)
    #plt.savefig('head_1', dpi=1000)
    #plt.savefig('head_2', dpi=1000)
    #plt.savefig('head_3', dpi=1000)
    #plt.savefig('head_4', dpi=1000)
    #plt.savefig('head_final', dpi=1000)
    #plt.savefig('TCE(maximum)_scenario_1_final', dpi=1000)
    plt.show()
#}}}
if perform(org, 'Optimization with NSGA-2_NIC'):# {{{
    namemf = 'mf'
    namemt3d = 'mt'
    mf_ws = './NSGA_55/new_NIC_2_MODFLOW_text/'
    mt_ws = './NSGA_55/new_NIC_2_MT3DUSGS/'

    # set working directory
    workdir = './NSGA_5555/'
    resultsdir = workdir + '/Results/'

    # set parameters
    debug = 0
    weight_conc = 10

    # depending on machine system, executio file should be changed.
    if sys.platform == 'linux': # linux system
        exe_name_mt3d='mt3d-usgs'
    else: # window system.
        exe_name_mt3d='mt3d-usgs_1.1.0_64.exe'

    # remove existed GA results {{{
    f= workdir # first make working dir
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Results/'
    if not os.path.isdir(f):
        os.mkdir(f)
    f = workdir + '/Data/'
    if not os.path.isdir(f):
        os.mkdir(f)
    for f in glob.glob(workdir+'/GA*'):
        if os.path.isdir(f):
           shutil.rmtree(f)
    for f in glob.glob(workdir+'/Results/ga_gen*'):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(workdir+'/Results/ga_results.txt'):
        if os.path.isfile(f):
            os.remove(f)
    # }}}

    # set GA parameters
    nper = 201
    perlen = [5 for i in range(201)]
    nstp = [1 for j in range(216)]
    steady = [False]

    # initialize multiprocessing with limited cpu
    processlock = multiprocessing.Semaphore(8)

    CXPB = 0.8  # probability of cross in genes.
    MUTPB = 0.1  # probability of mutation
    NGEN = 30   # number of generation for GA.
    npop = 100  # number of population of GA model.

    # check elapsed time
    time_start = time.time()

    # load all models.
    mf = flopy.modflow.Modflow.load('new_NIC_2.mfn', version='mf2005', exe_name='mf2005', model_ws=mf_ws)
    mt = flopy.mt3d.Mt3dms.load('new_NIC_2.nam', version='mt3d-usgs', exe_name=exe_name_mt3d, model_ws=mt_ws, modflowmodel=copy.deepcopy(mf))

    # update Dis for time domain.
    #mf.dis.nper = nper
    #mf.dis.perlen = perlen
    #mf.dis.nstp = nstp
    #mf.dis.steady = steady

    # Set origin of coordinate
    fdg.Grid.xoffset = 174512.72743837
    fdg.Grid.yoffset = 431411.27826534

    # Get xy grid
    xg, yg = flopyGetXyGrid(mf, center=1)
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    print('show xg shape = ', np.shape(xg))
    print('nrow = %d, ncol = %d' % (nrow, ncol))

    # contaminant area min max

    shpname = './NSGA_55/Data/boundary_polys.shp'
    polygons = geopandas.read_file(shpname)

    # load well locations
    '''
    filename_1 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_2 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_3 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_4 = './NSGA_222/Data_2/Well/well_location.xlsx'
    filename_5 = './NSGA_222/Data_2/Well/well_location.xlsx'
    df_1 = pandas.read_excel(filename_1, sheet_name='source')
    df_2 = pandas.read_excel(filename_2, sheet_name='source')
    df_3 = pandas.read_excel(filename_3, sheet_name='source')
    df_4 = pandas.read_excel(filename_4, sheet_name='source')
    df_5 = pandas.read_excel(filename_5, sheet_name='source')
    welx_1 = df_1['X']
    wely_1 = df_1['Y']
    welx_2 = df_2['X']
    wely_2 = df_2['Y']
    welx_3 = df_3['X']
    wely_3 = df_3['Y']
    welx_4 = df_4['X']
    wely_4 = df_4['Y']
    welx_5 = df_5['X']
    wely_5 = df_5['Y']
    welz = np.zeros(np.shape(welx_1))
    welc_1, welr_1, well = flopyXyzToIndex(mf, welx_1, wely_1, welz)
    welc_2, welr_2, well = flopyXyzToIndex(mf, welx_2, wely_2, welz)
    welc_3, welr_3, well = flopyXyzToIndex(mf, welx_3, wely_3, welz)
    welc_4, welr_4, well = flopyXyzToIndex(mf, welx_4, wely_4, welz)
    welc_5, welr_5, well = flopyXyzToIndex(mf, welx_5, wely_5, welz)
    '''
    # get well index
    # print('wel x,y shape = ', np.shape(welx_1))

    # DEAP individual Data type initialze {{{
    deap.creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMin)

    # Creating toolbox to use DEAP toolbox
    toolbox = deap.base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #    attr_welr : row index of pumping well.
    #    attr_welc : column index of pumping well.
    #    attr_welp : pumping rate for "WEL" package.
    welp_low = 1
    welp_up = 60
    def randwelidx(nrow_low, nrow_up, ncol_low, ncol_up):
        r = random.randint(nrow_low,nrow_up)
        c = random.randint(ncol_low, ncol_up)
        return r,c
    #toolbox.register("attr_welindx", randwelidx, 0, nrow-1, 0, ncol-1)
    toolbox.register("attr_welp_1", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_2", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_3", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_4", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_5", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_6", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_7", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_8", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_9", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_10", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_11", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_12", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_13", np.random.randint, welp_low, welp_up)
    toolbox.register("attr_welp_14", np.random.randint, welp_low, welp_up)

    # Structure initializers: define 'individual' to be an individual
    # Consisting of 50 'attr_bool' elements ('genes')
    toolbox.register("individual", deap.tools.initCycle, deap.creator.Individual,
                     (toolbox.attr_welp_1, toolbox.attr_welp_2, toolbox.attr_welp_3, toolbox.attr_welp_4, toolbox.attr_welp_5, toolbox.attr_welp_6, toolbox.attr_welp_7, toolbox.attr_welp_8, toolbox.attr_welp_9, toolbox.attr_welp_10, toolbox.attr_welp_11, toolbox.attr_welp_12, toolbox.attr_welp_13, toolbox.attr_welp_14), 1)

    # define the population to be a list of 'individual's
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalObjNSGA_6)

    def mutBenchmark(individual,low,up,std_welidx,indpb):
        # well location
        if random.random() < indpb:
            r,c = individual[0]
            r = r + int(random.gauss(0,std_welidx))
            c = c + int(random.gauss(0,std_welidx))
            if r < low[0][0]:
                r = low[0][0]
            elif r > up[0][1]:
                r = up[0][0]
            if c < low[0][0]:
                c = low[0][0]
            elif c > up[0][1]:
                c = up[0][1]
            individual[0] = (r,c)
        # update each pumping rate
        for i in range(1,len(individual)):
            if random.random() < indpb:
                individual[i] = random.uniform(low[i],up[i])

        return individual
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutGaussian, indpb=0.20, mu=10, sigma=10)
    low = [welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low, welp_low]
    up = [welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up, welp_up]
    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=0)
    #low = [(0, 0), welp_1_low, welp_2_low, welp_3_low]
    #up  = [(nrow-1, ncol-1), welp_1_up, welp_2_up, welp_3_up]
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=0, indpb=0.1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # k  : number of selected individual
    # nd : non-dominated algorithm to use.
    toolbox.register("select", tools.selNSGA2, nd='standard')
    # }}}

    random.seed(0)

    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=npop)

    print('check pop')

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses1 = np.zeros((npop,))
    fitnesses2 = np.zeros((npop,))
    order = np.zeros((npop,))

    # multiprocessing
    queue = multiprocessing.Queue()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    proc = []
    for i, ind in enumerate(invalid_ind):
        p = threading.Thread(target=toolbox.evaluate,
                    args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], queue, processlock,))
        proc.append(p)
        p.start()

    for i, p in enumerate(proc):
        p.join()
        order[i], fitnesses1[i], fitnesses2[i] = queue.get()

    # sort results. because the job results are unordered.
    fitnesses1 = fitnesses1[np.argsort(order)]
    fitnesses2 = fitnesses1[np.argsort(order)]

    # conduct main DEAP loop {{{
    for i in range(npop):
        pop[i].fitness.values = (fitnesses1[i],fitnesses2[i])

    # no actual selection
    pop = toolbox.select(pop, len(pop))
    #offspring = tools.selTournamentDCD(pop,int(npop/4)*4)
    #for ind in offspring:
    #    print(ind.fitness.crowding_dist)

    print("  Evaluated %i individuals" % len(pop))
    g = 0
    #with open(resultsdir + 'ga_gen%03d.txt' % (g), 'w') as fid:
    #    fid.write('index,welx,wely,welp_1,welp_2,welp_3,total_conc,cost1,cost2,cost3\n')
    #    for i in range(npop):
    #        fid.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (i, welx[pop[i][0]], wely[pop[i][0]], pop[i][1], pop[i][2], pop[i][3], fitnesses1[i], fitnesses2[i]))
    welp1_ = np.zeros((npop,))
    welp2_ = np.zeros((npop,))
    welp3_ = np.zeros((npop,))
    welp4_ = np.zeros((npop,))
    welp5_ = np.zeros((npop,))
    welp6_ = np.zeros((npop,))
    welp7_ = np.zeros((npop,))
    welp8_ = np.zeros((npop,))
    welp9_ = np.zeros((npop,))
    welp10_ = np.zeros((npop,))
    welp11_ = np.zeros((npop,))
    welp12_ = np.zeros((npop,))
    welp13_ = np.zeros((npop,))
    welp14_ = np.zeros((npop,))
    # check size of xg yg
    #print(np.shape(xg))
    for i, ind in enumerate(pop):
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i] = ind[:]
        #welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
    data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,fitnesses1,fitnesses2])
    df = pandas.DataFrame(data=data,columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','cost1(money)','cost2(conc)'])
    df.to_csv(resultsdir+'ga_gen%03d.csv'%(g))

    # Begin evolution
    for g in range(NGEN):  # {{{
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        #offspring = toolbox.select(pop, len(pop))
        offspring = modules.selTournamentDCD_(pop)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values, child2.fitness.values
                # fitness values of the children must be recalculated later
                # del child1.fitness.values
                # del child2.fitness.values

        # for mutant in offspring:
            # mutate an individual with probability MUTPB
            # toolbox.mutate(mutant)
            # del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        proc = []
        for i, ind in enumerate(invalid_ind):
            #print('ind[0], ind[1], ind[2], ind[3] = %d, %f, %f, %f'%(ind[0], ind[1], ind[2], ind[3]))
            #print('welc = %d, welr = %d, welp_1 = %f, welp_2 = %f, welp_3 = %f'%(welc[ind[0]], welr[ind[0]], ind[1], ind[2], ind[3]))
            p = threading.Thread(target=toolbox.evaluate,
                                 args=(copy.deepcopy(mf), copy.deepcopy(mt), i, workdir, ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6], ind[7], ind[8], ind[9], ind[10], ind[11], ind[12], ind[13], queue,processlock,))
            proc.append(p)
            p.start()

        for i, p in enumerate(proc):
            order[i], fitnesses1[i], fitnesses2[i] = queue.get()

        fitnesses1 = fitnesses1[np.argsort(order)]
        fitnesses2 = fitnesses2[np.argsort(order)]

        print('update offspring fitnessses values')
        for ind, fit1, fit2 in zip(offspring, fitnesses1, fitnesses2):
            ind.fitness.values = (fit1,fit2)

        print("Evaluated %i individuals" % len(invalid_ind))
        # The population is selected by NSGA-2
        pop = toolbox.select(pop+offspring, npop)
        print('   save results.')
        welp1_[i], welp2_[i], welp3_[i], welp4_[i], welp5_[i], welp6_[i], welp7_[i], welp8_[i], welp9_[i], welp10_[i], welp11_[i], welp12_[i], welp13_[i], welp14_[i] = ind[:]
        # welx1_[i], wely1_[i], welx2_[i], wely2_[i], welx3_[i], wely3_[i], welx4_[i], wely4_[i], welx5_[i], wely5_[i] = welx_1[weli1_[i]], wely_1[weli1_[i]], welx_2[weli2_[i]], wely_2[weli2_[i]], welx_3[weli3_[i]], wely_3[weli3_[i]], welx_4[weli4_[i]], wely_4[weli4_[i]], welx_5[weli5_[i]], wely_5[weli5_[i]]
        data = np.transpose([welp1_,welp2_,welp3_,welp4_,welp5_,welp6_,welp7_,welp8_,welp9_,welp10_,welp11_,welp12_,welp13_,welp14_,fitnesses1,fitnesses2])
        df = pandas.DataFrame(data=data, columns=['welp_1','welp_2','welp_3','welp_4','welp_5','welp_6','welp_7','welp_8','welp_9','welp_10','welp_11','welp_12','welp_13','welp_14','cost1(money)','cost2(conc)'])
        df.to_csv(resultsdir + 'ga_gen%03d.csv' % (g+1))
        # }}}

    print("-- End of (successful) evolution --")

    # show elapsed time.
    time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
#}}}20
if perform(org, 'Run Boeun model'):
    model_ws = './Boeun_regional_scale/'
    mf = flopy.modflow.Modflow('Boeun_regional_scale', exe_name='mf2005', model_ws=model_ws)
    success, buff = mf.run_model()
#}}}
if perform(org, 'Load flow model'):
    model_ws = './gw-sw_interaction/'
    namemf = 'Noseong_model_updated_2'
    mf = flopy.modflow.Modflow.load(namemf + '.nam', exe_name='mfnwt', model_ws=model_ws)
    data = flopy.utils.binaryfile.HeadFile(model_ws + 'Noseong_model_updated_2.hed')
    #cbb = flopy.utils.binaryfile.CellBudgetFile(model_ws + 'Boeun_regional_scale.ccf')
    times = data.get_times()
    head = data.get_data(totim=times[-1], mflay=0)
    plotFlopy3d(mf, head, title='head_initial', fontsize=30, colorbartitle='Head (m)', caxis=[70, 180])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("x(m)", labelpad=12, size=20)
    plt.ylabel("y(m)", labelpad=12, size=20)
    plt.show()
    '''
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    modelmap = flopy.plot.PlotMapView(model=mf, layer=0, ax=ax)
    qm = modelmap.plot_ibound()
    lc = modelmap.plot_grid()
    cs = modelmap.contour_array(head, levels=np.linspace(0, 10, 11))
    quiver = modelmap.plot_vector(qx, qy)
    plt.show()'''
#}}}
if perform(org, 'Run GW-SW interaction model'):
    model_ws = './gw-sw_interaction/'
    mf = flopy.modflow.Modflow('Noseong_model_updated_2', exe_name='mfnwt', model_ws=model_ws)
    success, buff = mf.run_model()
#}}}