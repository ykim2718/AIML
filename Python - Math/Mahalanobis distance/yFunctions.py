# -*- coding: utf-8 -*-
"""
Created on Wed Nov 5 17:24:06 2014
@author: yRocket
"""

import numpy as np
import scipy as sp
from scipy import stats
import scipy.spatial


def MahalanobisDistance(x, y, xavg, yavg, xsdv, ysdv, xycov):
    """ 
    Mahalanobis distance in between two 1-d arrays
    (copyLeft) yRocket, 11/10/2014
    Parameters: float x,y,xavg,yavg,xsdv,ysdv,xycov
    Returns: float md
    """
    xy = [x, y]
    avg = [xavg, yavg]
    cm = [[xsdv ** 2, xycov], [xycov, ysdv ** 2]]
    cm_inverse = np.linalg.inv(cm)
    md = sp.spatial.distance.mahalanobis(xy, avg, cm_inverse)
    return md


def MahalanobisDistanceArray(x, y):
    """
    Mahalanobis distance array in between two 1-d arrays
    (copyLeft) yRocket, 11/10/2014
    Parameters: list x,y
    Returns: list md
    """
    a = np.vstack((x, y))
    a_brushed = np.ma.compress_cols(np.ma.fix_invalid(a))
    a_brushed_total = len(a_brushed[0])
    m = np.cov(a_brushed)
    m = m / a_brushed_total * (a_brushed_total - 1)  # sample variance to population variance
    avg = [np.mean(a_brushed[0]), np.mean(a_brushed[1])]
    m_inverse = np.linalg.inv(m)
    md = []
    for i in range(len(a[0])):
        # print(i,a[0,i],a[1,i])
        xy = [a[0, i], a[1, i]]
        d = sp.spatial.distance.mahalanobis(xy, avg, m_inverse)
        md.append(d)
    return md


def yLSRwithMDCut(x, y, md, mdcut):
    """ 
    Least Sqaures Regression with Mahalanobis distance cut
    (copyLeft) yRocket, 11/11/2014 - 11/12
    Parameters: list x,y,md, Float mdcut
    Returns: list slope, yintercept, r_value
    """
    md = np.array(md)
    md[md > mdcut] = np.nan
    a = np.vstack((x, y, md))
    b = np.ma.compress_cols(np.ma.fix_invalid(a))
    if (b.shape[1] < 2): return (np.nan, np.nan, np.nan)
    slope, intercept, r_value, p_value, std_err = stats.linregress(b[0], b[1])
    return (slope, intercept, r_value)
