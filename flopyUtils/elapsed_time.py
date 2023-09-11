#!/usr/bin/env python
import numpy as np
import pandas
import matplotlib.pyplot as plt
import time, datetime

def elapsed_time(tstart,tend):# {{{
    '''
    Explain
     show elapsed time

    Usage
     import WNSmodules
     tstarttime = time.time()
     mf.run_model(silent=1)
     etime = WNSmodules.elpased_time(tstart,time.time())
    '''

    # calculated elapsed time with hour:min:sect
    if isinstance(tstart,datetime.datetime) and isinstance(tend,datetime.datetime):
        etime = tend-tstart
    else:
        etime = datetime.timedelta(seconds=tend-tstart)

    return etime
    # }}}
