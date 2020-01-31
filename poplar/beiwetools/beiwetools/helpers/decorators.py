''' Some decorators to use when defining functions.

These decorators add some extra attributes and methods to a function.
These are intended to simplify defining functions that may be used in 
sequences of procedures, such as post-processing tasks.  

The main decorator is @easy.  This decorator adds some extra features,
and provides a method that takes packed kwargs while returning labelled output:
            
    Example:
        >>> f_returns = ('x', 'y')
        >>> @easy(f_returns)
        >>> def f(a, b):
        >>>     x = a+1
        >>>     y = 2*b
        >>>     return(x, y)

    What the function does:        
        >>> f(a = 1, b = 2)
        (2, 4)

    Here are two dictionaries with keyword arguments:
        >>> d = {'a':1, 'b':2}
        >>> dd = {'a':1, 'b':2, 'c':3}

    And f's output:
        >>> f(**d)
        (2, 4)
        >>> f(**dd)
        f() got an unexpected keyword argument 'c'

    However:        
        >>> f.easy(d)
        OrderedDict([('x', 2), ('y', 4)])
        >>> f.easy(dd)
        OrderedDict([('x', 2), ('y', 4)])        
'''
import logging
from inspect import getfullargspec
from collections import OrderedDict


logger = logging.getLogger(__name__)

def returns_to_dict(f, returns, ordered = True):
    '''
    Get a function that bundles f's output as a dictionary.

    Args:
        f (func): A function.
        returns (list): List of names for what f returns.
        ordered (bool): If True, ff returns an OrderedDict.
        
    Returns:
        ff (func): 
            Takes f's input as a dictionary.
            Returns f's output as a dictionary.
        
    Example:
        >>> def f(a, b): return(a+1, 2*b)
        >>> f(a = 1, b = 2)
        (2, 4)

        >>> ff = returns_to_dict(f, ['x', 'y'], ordered = False)
        >>> ff({'a':1, 'b':2})
        {'x': 2, 'y': 4}
    '''
    if ordered:
        ff = lambda **kwargs : OrderedDict(zip(returns, f(**kwargs)))
    else:
        ff = lambda **kwargs : dict(zip(returns, f(**kwargs)))
    return(ff)    


def returns(r):
    '''
    Attach r, a list of names for f's output.
    '''
    def wrapper(f):
        f.returns = r
        return(f)
    return(wrapper)        


def to_dictionary():
    '''
    Add a method that attaches labels to output.
    f must have a returns attribute that is a list of output labels.
    '''
    def wrapper(f):
        f.to_dict = returns_to_dict(f, f.returns)
        return(f)
    return(wrapper)

    
def get_kwargs(g = None):
    '''
    Add a method that drops foreign keyword arguments.
    Use g = None unless it's an odd situation like the definition of easy().
    '''
    def wrapper(f):
        if g is None:
            args = getfullargspec(f).args
        else: 
            args = getfullargspec(g).args
        def gk(d):
            kwargs = {}
            keys = list(d.keys())
            for k in keys:
                if k in args: kwargs[k] = d[k]                    
            return(kwargs)
        f.get_kwargs = gk
        return(f)
    return(wrapper)


def easy(r):
    '''
    Adds an "easy" method.
    Combine @returns, @to_dictionary, and @get_kwargs.
    '''
    def wrapper(f):
        @get_kwargs(f)
        @to_dictionary()
        @returns(r)        
        def ff(**kwargs): return(f(**kwargs))
        def eff(kwargs): 
            d = ff.get_kwargs(kwargs)
            return(ff.to_dict(**d))
        ff.easy = eff
        return(ff)
    return(wrapper)