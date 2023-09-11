#!/usr/bin/env python
import os, sys, platform

def makedir(dirname):
   '''
   Explain
    make directory
   '''
   if not os.path.exists(dirname):
      os.makedirs(dirname)

if __name__ == '__main__':
   makedir('./temp1/temp2')
