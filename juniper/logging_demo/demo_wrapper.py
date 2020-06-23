'''Demo of logging package
'''
import logging
from demo_function import try_inverse


logger = logging.getLogger(__name__)


def wrapper(numbers):
    '''
    Args:
        numbers (list):  List of floats or ints.
        
    Returns:
        results (list):  List of floats.
    '''
    logger.info("Begin")

    results = []
    for x in numbers:
        y = try_inverse(x)
        results.append(y)

    logger.info("End")
    return(results)    
    