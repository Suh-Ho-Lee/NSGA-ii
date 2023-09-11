#!/usr/bin/env python

def islinux():
   '''
   Explain
    check system is linux or window system...
   '''
   import sys
   checkSys = {'win32':False,'linux':True}
   return checkSys[sys.platform]

if __name__ == '__main__':
   print(islinux())

