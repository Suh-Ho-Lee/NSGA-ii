#!/usr/bin/env python
import matplotlib

def axes_default(ax,**kwargs):
   '''
   Explain
    set default axes....
   '''
   # check ax class
   if not (type(ax).__name__=='AxesSubplot'):
      raise Exception('ERROR: current class of ax(=%s) is not ax,matplotlib.axes._subplots.AxesSubplot'%(type(ax)))

   # get options from dicts.....
   options = kwargs.keys()
   fontsize=10
   if 'fontsize' in options:
      fontsize=kwargs['fontsize']

   # apply all options...
   if 'xlabel' in options:
      ax.set_xlabel(kwargs['xlabel'],fontsize=fontsize)
   if 'ylabel' in options:
      ax.set_ylabel(kwargs['ylabel'],fontsize=fontsize)

   ax.tick_params(axis='both', which='major', labelsize=fontsize)
   if 'xticks' in options:
      ax.set_xticks(kwargs['xticks'])
   if 'yticks' in options:
      ax.set_yticks(kwargs['yticks'])

   # set xy limit...
   if 'xlim' in options:
      ax.set_xlim(kwargs['xlim'])
   if 'ylim' in options:
      ax.set_ylim(kwargs['ylim'])

   if 'title' in options:
      ax.set_title(kwargs['title'])

   if 'xlog' in options:
      if kwargs['xlog']:
         ax.set_xscale('log')

   if 'ylog' in options:
      if kwargs['ylog']:
         ax.set_yscale('log')

   # get parent of axis...
   fig = ax.figure
   if 'dpi' in options:
      fig.set_dpi(kwargs['dpi'])

   if 'facecolor' in options:
      fig.set_facecolor(kwargs['facecolor'])
   else:
      fig.set_facecolor('w')
