'''Demo of logging package
'''
import logging
import time


logger = logging.getLogger(__name__)


def try_inverse(x):
    '''
    Try to get the multiplicative inverse of a number.
    
    Args:
        x (int or float):  Input number.
        
    Returns:
        y (float or NoneType):  Inverse of x if it exists; None otherwise.
    '''
    try:    
        y = 1/x
    except:
        y = None
        logger.debug("Might be a problem when dividing by %s." % str(x))
        logger.warning("Unable to divide.")
        logger.error("Serious problem at time %s." % str(time.time()))
        logger.critical("Critical problem!")
    return(y)
