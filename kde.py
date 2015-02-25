"""
An implementation of the kde bandwidth selection method outlined in:

Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

Based on the implementation in Matlab by Zdravko Botev.

Daniel B. Smith, PhD
Updated 1-23-2013
"""

from __future__ import division

import numpy as np
import scipy.optimize
import scipy.fftpack

pi = np.pi
sqrtpi = np.sqrt(pi)
sqrt2pi = np.sqrt(2*pi)
pisq = pi**2

def kde(data, N=None, MIN=None, MAX=None, overfit_factor=1.0):
    
    # Parameters to set up the mesh on which to calculate
    N = 2**14 if N is None else int(2**np.ceil(np.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range/10 if MIN is None else MIN
        MAX = maximum + Range/10 if MAX is None else MAX
    
    # Range of the data
    R = MAX-MIN
    
    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = np.histogram(data, bins=N, range=(MIN,MAX))
    DataHist = DataHist/M
    DCTData = scipy.fftpack.dct(DataHist, norm=None)
    
    M = M
    I = np.arange(1,N, dtype=np.float64)**2
    SqDCTData = np.float64((DCTData[1:]/2.0)**2)
    
    # The fixed point calculation finds the bandwidth = t_star
    failure = True
    for guess in np.logspace(-1,2,20):
        #wstderr(str(guess) + ' ')
        try:
            t_star = scipy.optimize.brentq(fixed_point, 0, guess, args=(np.float64(M), I, SqDCTData))
            failure = False
            break
        except ValueError:
            failure = True

    if failure:
        raise ValueError('Initial root-finding failed.')

    # Smooth the DCTransformed data using t_star divided by an overfitting
    # param that allows sub-optimal but allows for "sharper" features
    SmDCTData = DCTData*np.exp(-np.arange(N)**2*pisq*t_star/(2*overfit_factor))
    
    # Inverse DCT to get density
    density = scipy.fftpack.idct(SmDCTData, norm=None)*N/R
    
    mesh = (bins[0:-1]+bins[1:])/2.
    
    bandwidth = np.sqrt(t_star)*R
    
    density = density/np.trapz(density, mesh)
    return bandwidth, mesh, density

def fixed_point(t, M, I, a2):
    l=7
    x0 = numexpr.evaluate('I**l')
    x1 = np.exp(-I*pisq*t)
    x2 = x0 * a2 * x1
    x3 = np.sum(x2)
    f = 2*pisq**l * x3
    for s in xrange(l, 1, -1):
        K0 = np.prod(np.arange(1, 2.*s, 2))/sqrt2pi
        const = (1 + (0.5)**(s + 0.5))/3.
        time = (2*const*K0/M/f)**(2./(3.+2.*s))
        x0 = numexpr.evaluate('I**s')
        x10 = -I * pisq * time
        x1 = np.exp(x10)
        x2 = x0 * a2 * x1
        x3 = np.sum(x2)
        f = 2*pisq**s * x3
    return t-(2*M*sqrtpi*f)**(-0.4)
