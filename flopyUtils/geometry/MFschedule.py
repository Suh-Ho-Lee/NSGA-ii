#!/usr/bin/env python

def modflow_schedule(mf, # {{{
        welfname=None, # NOTE: './web/wel_transient.json',
        rchfname=None, # NOTE: './web/rch_transient.json',
        final_time=None,
        oc_keys=['save head','save drawdown'],
        steady=False,
        dt=1,
        nstp_freq=1,nper_freq=1,outputfrequency=1,
        remove_packages = ['wel','rch'],
        verbose=1):
    import flopy
    import os, sys, platform, copy, inspect, json
    import numpy as np
    from .xy2index import xy2index, xyz2index
    from ..utils.print import print_

    '''
    Explain
     load scheduling well and recharge files.

    Usage
     mf = flopyUtils.modflow_schedule(mf,welfname='./web/wel_transient.json',
     rchfname = './web/rch_transient.json'
     )

    rchfname structure
    rchfname = {'time':time,'rch':rch}

    welfname structure
    welfname = {}
    welfname[0] = {'x':x,'y':y,'time':time, 'pumping_rate':pumping_rate}

    Options
     dt           - each time steps...
     verbose      - show process..   (type: bool or int)
     final_time   - check end of simulation. (type: float)
     nstp_freq    - output frequency for each perlen (defaut: nstp_freq=5)
     nper_freq    - output frequency in total nper (default: 1)
     steady       - True - steady state / False - transient.

    See Also
    : ../test/test_model_scheduling.ipynb
    '''
    # get current file name
    func_name = inspect.stack()[0][0].f_code.co_name

    # check modflow class..
    if not isinstance(mf,flopy.modflow.Modflow):
       raise Exception('{}: input argument mf is not class of flopy.modflow.Modflow.'.format(func_name))

    # first remove all rch and wel package
    if ('WEL' in mf.get_package_list()) & ('wel' in remove_packages):
       mf.remove_package('wel')
    if ('RCH' in mf.get_package_list()) & ('rch' in remove_packages):
       mf.remove_package('rch')

    # check input arguments... {{{
    for k in oc_keys:
       if not k in ['save budget', 'save head', 'save drawdown']:
          raise Exception('ERROR: %s is not defined for ModflowOc package.'%(k))
    # }}}

    # check each module exists
    iswel = 1 if welfname else 0
    isrch = 1 if rchfname else 0

    if (not iswel) & (not isrch):
       print('%s: nothing to do...'%(func_name))
       return mf

    # get wel package {{{
    if isinstance(welfname, str):
        if not os.path.isfile(welfname):
            raise Exception('ERROR: we cannot find %s'%(welfname))
        with open(welfname,'r') as fid:
            data_wel = json.load(fid)
    elif isinstance(welfname,dict):
        if verbose:
            print('modflow_schedule: load wel data_wel with dict type...')
        data_wel = copy.deepcopy(welfname)
    # }}}

    # get rch package {{{
    if isinstance(rchfname, str):
        if not os.path.isfile(rchfname):
            raise Exception('ERROR: we cannot find %s'%(rchfname))
        with open(rchfname,'r') as fid:
            data_rch = json.load(fid)
    elif isinstance(rchfname,dict):
        if verbose:
            print('modflow_schedule: load rch data with dict type...')
        data_rch = copy.deepcopy(rchfname)
    # }}}

    # initialize unique time....
    time_unique = []

    if iswel: # {{{
      if verbose:
         print('modflow_schedule: load well data.')
      wel_time_unique = []
      welx = []
      wely = []
      welz = []
      for i in data_wel.keys():
         if steady:
            if len(data_wel[i]['time']) > 1:
               raise Exception('ERROR: for steady state model, length of time schedule should be 1. size of wel time is {}'.format(np.shape(data_wel[i]['time'])))
         wel_time_unique.extend(data_wel[i]['time'])
         welx.append(data_wel[i]['x'])
         wely.append(data_wel[i]['y'])
         if 'z' in data_wel[i].keys():
            if data_wel[i]['z']:
               welz.append(data_wel[i]['z'])

      try:
         wel_time_unique = np.array(np.unique(wel_time_unique),dtype=float)
      except ValueError:
         print(wel_time_unique)
         print('ERROR: we cannot make wel_time_unique matrix...')
      # check wel_time_unique contains zero time for specifying start time.
      if ~np.any(wel_time_unique==0):
         wel_time_unique = np.concatenate(([0.],wel_time_unique))
      welx = np.array(welx)
      wely = np.array(wely)
      welz = np.array(welz,dtype=float)
      nwel   = len(welx) # check number of wells.

      time_unique.append(wel_time_unique)
      # }}}

    if isrch: # {{{
      if verbose:
         print('modflow_schedule: load recharge data.')
      rch_time_unique = np.array(np.unique(data_rch['time']))
      rch_rch         = np.array(data_rch['rch'])*0.001 # mm day-1 > m day-1


      time_unique.append(rch_time_unique)
      # }}}

    # check operating time...
    if final_time:
       # merge final time...
       time_unique.append(np.array([final_time]))
       time_unique = np.unique(np.concatenate(time_unique))
       pos = np.where(time_unique<=final_time)
       time_unique = time_unique[pos]
    else:
       # update time uniqueness
       time_unique = np.concatenate(time_unique)
       time_unique = np.unique(time_unique)

    if verbose:
       print('modflow_schedule: modflow is simulated until {}'.format(np.amax(time_unique)))
       print('modflow_schedule: time_unique =\n{}'.format(time_unique))

    if steady:
       print_('{}: simulate steady state model.'.format(func_name),debug=verbose)
       ntime  = 1
       nper   = 1
       perlen = [.1]
       nstp   = [1]
       MFsteady = [True]
       nper_freq=1
       nstp_freq=1
    else:
       print_('{}: simulate transient model.'.format(func_name),debug=verbose)
       ntime  = len(time_unique) # get number of times.
       perlen = np.diff(time_unique)
       nper   = len(perlen)
       nstp   = np.array(np.ceil(perlen/dt),dtype=int)
       MFsteady = np.zeros((nper,),dtype=int)

    if verbose:
       print('%s: nper    = %d'%(func_name, nper))
       print('{}: perlen = {}'.format(func_name,perlen))
       print('{}: ntsp   = {}'.format(func_name,nstp))

    if iswel: # generate wel schedule matrix  {{{
       print_('{}: generate wel_schedule_matrix ntime x nwel = ({} x {})'.format(func_name,ntime,nwel),debug=verbose)

       wel_schedule_matrix = np.zeros((ntime,nwel))
       if not steady:# transient simulation
          for i, key in enumerate(data_wel.keys()):
              if verbose:
                  print('   assign well = {}'.format(data_wel[key]['name']))
              # check pumping schedule.
              time = data_wel[key]['time']
              pump = data_wel[key]['pumping_rate']
              nt_ = len(time)
              for j in range(nt_):
                  if j < nt_-1:
                      post = np.where( (time[j] <= time_unique) & (time_unique < time[j+1]))
                      wel_schedule_matrix[post,i] = pump[j]
                  elif j == nt_-1:
                      post = np.where( (time[j] <= time_unique) )
                      wel_schedule_matrix[post,i] = pump[j]
       else:
          for i, key in enumerate(data_wel.keys()):
             print_('{}: pumping rate = {}'.format(func_name,data_wel[key]['pumping_rate']),debug=verbose)
             wel_schedule_matrix[0,i] = data_wel[key]['pumping_rate'][0]

       # x, y coordinates to row, cols
       if not np.any(welz):
          welr, welc = xy2index(mf,welx,wely)
          print_('modflow_schedule: welz = {}'.format(welz),debug=verbose)
          well = (mf.nlay-1)*np.ones(np.shape(welr))
       else:
          well, welr, welc = xyz2index(mf,welx,wely,welz)
          #raise Exception('ERROR: z elevation for wel package is not supported yet.')
       
       if verbose:
          print('   wel_schedule_matrix')
          print('   {}'.format(wel_schedule_matrix))

       if verbose:
          print('modflow_schedule: update wel package')
       stress_period_data = {}
       for kper in range(nper):
           data = []
           for i,l,r,c in zip(range(len(welr)),well,welr, welc):
               data.append([l, r, c, wel_schedule_matrix[kper,i]])
           stress_period_data[kper] = data
       mf.wel = flopy.modflow.ModflowWel(mf,stress_period_data=stress_period_data)
    # }}}

    if isrch: # generate rch schedule matrix {{{ 
       if verbose:
           print('generate recharge schedule matrix.')

       # check size of RCH variable.
       rch_ntime = len(data_rch['time'])
       rch_size  = np.array(np.shape(data_rch['rch']))
       if (len(rch_size)==3) & \
          (np.sum(rch_size == np.array((rch_ntime, mf.nrow, mf.ncol)))==3):# time series with 2D array.
          rch_schedule_matrix = np.zeros((ntime,mf.nrow,mf.ncol))
       elif len(rch_size)==1 & (np.sum(rch_size==(rch_ntime,))==1):  # time series data
          rch_schedule_matrix = np.zeros((ntime,))
       else:
          raise Exception('ERROR: current dimension of rch(={}) is not supported.'.format(rch_size))

       time = np.array(data_rch['time'])
       nt_  = len(time)
       for j in range(nt_):
           if j < nt_-1:
               post = np.where( (time[j] <= time_unique) & (time_unique < time[j+1]))
               rch_schedule_matrix[post] = rch_rch[j]
           elif j == nt_-1:
               post = np.where( (time[j] <= time_unique) )
               rch_schedule_matrix[post] = rch_rch[j]
       pos = np.where(np.isnan(rch_schedule_matrix))
       rch_schedule_matrix[pos] = 0

       if verbose:
           print('modflow_schedule: update rch package.')
       stress_period_data = {}
       for kper in range(nper):
           stress_period_data[kper] = rch_schedule_matrix[kper]
       mf.rch = flopy.modflow.ModflowRch(mf,rech=stress_period_data)
       # }}}

    print_('{}: update dis package.'.format(func_name),debug=verbose)
    mf.dis = flopy.modflow.ModflowDis(mf,
            nlay=mf.nlay, nrow=mf.nrow, ncol=mf.ncol,
            delr=mf.dis.delr, delc=mf.dis.delc,
            top=mf.dis.top, botm=mf.dis.botm,
            nper=nper,perlen=perlen,nstp=nstp,steady=MFsteady)

    if verbose:
       print_('modflow_schedule: update oc package.',debug=verbose)
       print('   nper_freq:       %f'%(nper_freq))
       print('   nstp_freq:       %f'%(nstp_freq))
       print('   outputfrequency: %f'%(outputfrequency))
    if 'OC' in mf.get_package_list():
       mf.remove_package('oc')
    stress_period_data = {}
    cn_freq = 0 # count for time frequency..
    for kper in np.linspace(0,nper-1,int(nper/nper_freq),dtype=int):
       _nstp_freq = int(nstp[kper]/nstp_freq)
       if _nstp_freq <= 1:
          _nstp_freq = 2
       print_('   kper: {}, nstp_freq = {}'.format(kper,_nstp_freq),debug=verbose)
       for kstp in np.unique(np.linspace(0,nstp[kper]-1,_nstp_freq,dtype=int)):
          cn_freq += 1
          #print_('   cn_freq%outputfrequency = {}'.format(cn_freq%outputfrequency),debug=verbose)
          if (
                (outputfrequency==1) |                  # freq=1 return all outputs
                (cn_freq%outputfrequency==1) |          # every time frequency..
                ((kper==nper-1) & (kstp==nstp[kper]-1)) # final time step.
               ):
             #print_('   ({},{})'.format(kper,kstp),debug=verbose)
             stress_period_data.update({(kper,kstp):oc_keys})
    if 0:
       print(stress_period_data)
    mf.oc = flopy.modflow.ModflowOc(mf,stress_period_data=stress_period_data)

    # return outputs.
    return mf
    # }}}
