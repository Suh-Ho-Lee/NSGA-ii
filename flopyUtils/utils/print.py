#!/usr/bin/env python

# utils
def mfPrint(string,debug=False):# {{{
    '''

    Usage:
        >>> mfPrint('write something',debug=debug)

    Variables\n
        string : show variables \n
        debug  : show or not.\n
    '''
    if debug:
        print(string)
# }}}
def print_(string,debug=False):# {{{
    '''

    Usage:
        >>> mfPrint('write something',debug=debug)

    Variables\n
        string : show variables \n
        debug  : show or not.\n
    '''
    if debug:
        print(string)
# }}}
def mf96Print(string,debug=False): # {{{
    '''
    Explain
     Print debuging. This function same as "mfPrint".

    Usage
        >>> mf96Print("write something",debug=debug)

    See also.
     mfPrint,
    '''
    mfPrint(string,debug=debug)
# }}}
