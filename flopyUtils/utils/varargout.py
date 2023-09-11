#!/usr/bin/env python

import  traceback

def varargout(*args): # {{{
    '''
    Explain
     this function is enable to variable outputs.

    Usage
        def func():\n
             a = 1\n
             b = 2\n
             return varargout(a,b)\n
        \n
        a = func() \n
        a,b = func()\n

    Reference
     https://stackoverflow.com/questions/14147675/nargout-in-python
    '''
    callInfo = traceback.extract_stack()
    callLine = str(callInfo[-3].line)
    split_equal = callLine.split('=')
    split_comma = split_equal[0].split(',')
    num = len(split_comma)
    return args[0:num] if num > 1 else args[0]
# }}}
