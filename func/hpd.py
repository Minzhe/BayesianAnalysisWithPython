#################################################
###                  hpd.py                   ###
#################################################

import numpy as np
from scipy.stats import kde

def hpd_grid(sample, alpha=0.05, roundto=2):
    '''
    Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode.

    Parameters
    ----------
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
        
    Returns
    ----------
    hpd: array with the lower 
    '''