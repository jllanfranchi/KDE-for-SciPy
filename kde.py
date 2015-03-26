#!/usr/bin/env python

"""
An implementation of the kde bandwidth selection method outlined in:

Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

Based on the implementation in Matlab by Zdravko Botev.

Daniel B. Smith, PhD
Updated 1-23-2013

Further modified by J. L. Lanfranchi
2015-02-24: Faster via quad -> double precision, more numpy vectorized
  functions, numexpr for a couple of the slower evaluations. Note that the
  double precision may make this fail in some circumstances, but I haven't seen
  it do so yet. Regardless, modifying the calls to float64 -> float128 and
  eliminating the numexpr calls (only supports doubles) should make it
  equivalent to the original implementation.

2015-03-09: Add variable-bandwidth implementation that does the following:
  1) compute optimal bandwidth using the improved-Sheather-Jones (ISJ)
     algorithm described in the Botev paper cited above
  2) Use a modified version of the variable-bandwidth algorithm described in:
       I.S. Abramson, On bandwidth variation in kernel estimates - A square
       root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982
     The modification I made to this Ambramson paper is to force the
     peak-density point to use the ISJ BW found in step (1). This is done by
     dividing the inverse-square-root bandwidths by the bandwidth at the
     peak-density point and multiplying by the ISJ BW. (This appears to do
     quite well at both capturing the peak's characteristics and smoothing out
     bumps in the tails, but we should be cautious if false structures near the
     peak may arise due to densities similar to that of the peak.)
"""

from __future__ import division

import numexpr
import numpy as np
import mkl
import math
import scipy.optimize
import scipy.fftpack
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d
from numba import vectorize, double, f4, f8
from numba import jit, autojit, njit
import gaussians as GAUS
import multiprocessing

#openmp_num_threads = min(multiprocessing.cpu_count(), 8)
openmp_num_threads = multiprocessing.cpu_count()

#print 'fast numpy?', np.use_fastnumpy
#mkl.set_num_threads(mkl.get_max_threads())
mkl.set_num_threads(1)

numexpr.set_num_threads(1)
numexpr.set_vml_accuracy_mode('fast')
#numexpr.set_vml_num_threads(mkl.get_max_threads())
numexpr.set_vml_num_threads(1)
#print numexpr.get_vml_version()

pi = np.pi
sqrtpi = np.sqrt(pi)
sqrt2pi = np.sqrt(2*pi)
pisq = pi**2


#def gaussian(x, mu, sigma):
#    return 1./(sqrt2pi*sigma) * np.exp(-(x-mu)**2/2./sigma**2)

#@vectorize(['f8(f8,f8,f8)', 'f4(f4,f4,f4)'], nopython=True)
#def ufunc_gaussian(x, mu, sigma):
#    twosigma2 = 2*sigma**2
#    invsqrt2pisigma = 1./(math.sqrt(math.pi*2)*sigma)
#    return invsqrt2pisigma * math.exp(-((x-mu)**2)/twosigma2)

#@jit('void(f8[:],f8[:],f8[:],f8[:])')
#def gaussians(f, x, mu, sigma):
#    N = len(mu)
#    J = len(x)
#    for j in range(J):
#        for n in range(N):
#            f[j] += 1./((2*math.pi)*sigma[n])**(0.5) * math.exp(-((x[j]-mu[n])**2)/(2*sigma[n]**2)) #ufunc_gaussian(x, mu[n], sigma[n])
#    return f

#@profile
def gaussian(x, mu, sigma):
    twosigma2 = 2*sigma**2 #numexpr.evaluate('2*sigma**2')
    #invsqrt2pisigma = 
    sqrt2pisigma = sqrt2pi * sigma
    x1 = -(x - mu)**2 / twosigma2
    return numexpr.evaluate('exp(x1)') / sqrt2pisigma

#@profile
#@jit(f8[:](f8[:], f8[:], f8[:]))
#def gaussians(x, mu, sigma):
#    minustwosigma2 = -2*sigma**2
#    invsqrt2pisigma = 1./sqrt2pi*sigma
#    f = np.zeros_like(x)
#    #[f.__iadd__(numexpr.evaluate('invsqrt2pisigma[n] * exp(-((x-mu[n])**2)/twosigma2[n])')) for n in xrange(len(mu))]
#    for n in xrange(len(mu)):
#        y1 = (x-mu[n])**2 / minustwosigma2[n]
#        y2 = invsqrt2pisigma[n] * np.exp(y1)
#        #f += invsqrt2pisigma[n] * np.exp(-((x-mu[n])**2)/twosigma2[n])
#        f += y2
#    return f

#@profile
def gaussians(outbuf, x, mu, sigma, **kwargs):
    [outbuf.__iadd__(gaussian(x, mu[n], sigma[n])) for n in xrange(len(mu))]


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
    
    # TRY: Gaussian KDE initial estimate
    #kde = gaussian_kde(data, bw_method='silverman')
    #bins = np.linspace(MIN, MAX, N+1)
    #bin_centers = (bins[0:-1]+bins[1:])/2.
    #DataHist = kde.evaluate(bin_centers)
    
    #print len(DataHist)
    
    DCTData = scipy.fftpack.dct(DataHist, norm=None)
    
    M = M
    I = np.arange(1,N, dtype=np.float64)**2
    SqDCTData = np.float64((DCTData[1:]/2.0)**2)
    
    # The fixed point calculation finds the bandwidth = t_star
    failure = True
    for guess in np.logspace(-1,2,20):
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


#@profile
def vbw_kde(data, N=None, MIN=None, MAX=None, evaluate_dens=True, evaluate_at=None, overfit_factor=1.0):
    '''
    Parameters
    ----------
    data            The data points for which the density estimate is sought
    
    N               Number of points with which to form regular mesh, from MIN
                      to MAX; this gets DCT'd, so N should be a power of two.
                      -> Default: 2**14 (16384)
    
    MIN             Minimum of range over which to compute density.
                      -> Default: min(data) - range(data)/10
    
    MAX             Maximum of range over which to compute density>
                      -> Default: max(data) + range(data)/10
    
    evaluate_dens   Whether to evaluate the density either at the mesh points
                      defined by N, MIN, and MAX, or at the points specified by
                      the argument evaluate_at. If False, only the gaussians'
                      bandwidths and the mesh locations (no density) are
                      returned. Evaluating the density is a large fraction of
                      total execution time, so setting this to False saves time
                      if only the bandwidths are desired.
                      -> Default: True
    
    evaluate_at     Points at which to evaluate the density. If None is
                      specified, evaluates at points on the mesh defined by
                      MIN, MAX, and N.
                      -> Default: None
    
    overfit_factor  EXPERIMENTAL: For the first part of the algorithm, the
                      improved-Sheather-Jones fixed-bandwidth (ISJ-FBW) bit,
                      the density can be overfit by specifying overfit_factor >
                      1.0 and underfit using a value < 1.0.
                      -> Default: 1.0
    
    Returns
    -------
    kernel_bandwidths The gaussian bandwidths, one for each data point 
    
    evaluate_at       Locations at which the density is evaluated
    
    vbw_dens_est      Density estimates at the mesh points, or None if
                        evaluate_dens is False
    
    Notes
    -----
    Specifying the range:
    
        The specification of MIN and MAX are critical for obtaining a
    reasonable density estimate. If the true underlying density slowly decays
    to zero on one side or the other, like a gaussian, specifying too-small a
    range will distort the edge the VBW-KDE finds. On the other hand, an abrupt
    cut-off in the distribution should be accompanied by a similar cutoff in
    the computational range (MIN and/or MAX). The algorithm here will
    approximate such a sharp cut-off with roughly the same performance to the
    reflection method for standard KDE's (as the fixed-BW portion uses a DCT of
    the data), but note that this will not perform as well as polynomial-edges
    or other modifications that have been proposed in the literature.
    
    overfit_factor, other tweaks:
    
        I've seen no improvement by changing this parameter, but it remains for
    experimental purposes. Other avenues to explore include changing the
    "normalization" of the variable-bandwidth bit that I use which forces it to
    have a bandwidth at the peak matching that found by the ISJ-FBW part 
    '''
    # Parameters to set up the mesh on which to calculate
    if N is None:
        N = 2**14 #if N is None else int(2**np.ceil(np.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        if Range == 0:
            warnings.warn('Range of data is 0; there are ' + str(len(data)) +
                          ' data points.')
        MIN = minimum - Range/10 if MIN is None else MIN
        MAX = maximum + Range/10 if MAX is None else MAX
    
    # Range for computation
    R = MAX-MIN
    
    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = np.histogram(data, bins=N, range=(MIN,MAX))
    DataHist = DataHist/M
    
    # TRY: Gaussian KDE initial estimate
    #kde = gaussian_kde(data, bw_method='silverman')
    #bins = np.linspace(MIN, MAX, N+1)
    #bin_centers = (bins[0:-1]+bins[1:])/2.
    #DataHist = kde.evaluate(bin_centers)
    
    DCTData = scipy.fftpack.dct(DataHist, norm=None)
    
    M = M
    I = np.arange(1,N, dtype=np.float64)**2
    SqDCTData = np.float64((DCTData[1:]/2.0)**2)
    
    # The fixed point calculation finds the bandwidth = t_star
    failure = True
    for guess in np.logspace(-1,2,20):
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
    fbw_dens_on_mesh = scipy.fftpack.idct(SmDCTData, norm=None)*N/R
    
    # Start by defining the mesh as the bins' centers
    mesh = (bins[0:-1]+bins[1:])/2.
    # But add the lower and upper edges in case data points live there
    fbw_dens_on_mesh = fbw_dens_on_mesh/np.trapz(fbw_dens_on_mesh, mesh)
    isj_bandwidth = np.sqrt(t_star)*R
    
    # Create linear interpolator for this new density then find density est. at
    # the original data points' locations; call this fbw_dens_at_datapoints
    interp = interp1d(x=mesh, y=fbw_dens_on_mesh,
                      kind='linear',
                      copy=False,
                      bounds_error=True,
                      fill_value=np.nan,
                      assume_sorted=True) 
    fbw_dens_at_datapoints = interp(data)
    
    # Note below diverges from the published Ambramson method, by forcing the
    # bandwidth at the max of the density distribution to be exactly the
    # bandwidth found above with the improved Sheather-Jones BW selection
    # technique. Refs:
    #   I.S. Abramson, On bandwidth variation in kernel estimates - A square
    #       root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982
    #   P. Hall, T. C. Hu, J. S. Marron, Improved Variable Window Kernel
    #       Estimates of Probability Densities, Annals of Statistics Vol. 23,
    #       No. 1, 1-10, 1995
    root_pknorm_fbw_dens_est = np.sqrt(fbw_dens_at_datapoints/np.max(fbw_dens_at_datapoints))
    kernel_bandwidths = isj_bandwidth/root_pknorm_fbw_dens_est
    
    if evaluate_at is None:
        evaluate_at = mesh
    
    if not evaluate_dens:
        return kernel_bandwidths, evaluate_at, None
    vbw_dens_est = np.zeros_like(evaluate_at)
    GAUS.gaussians(
        outbuf  = vbw_dens_est,
        x       = evaluate_at,
        mu      = data,
        sigma   = kernel_bandwidths,
        threads = openmp_num_threads
    )
    vbw_dens_est /= len(data)
    
    vbw_dens_est = vbw_dens_est/np.trapz(y=vbw_dens_est, x=evaluate_at)
    
    return kernel_bandwidths, evaluate_at, vbw_dens_est


#@profile
def fixed_point(t, M, I, a2):
    l=7
    x0 = numexpr.evaluate('I**l')
    x1 = numexpr.evaluate('exp(-I*pisq*t)')
    x2 = x0 * a2 * x1
    x3 = np.sum(x2)
    f = 2*pisq**l * x3
    for s in xrange(l, 1, -1):
        K0 = np.prod(np.arange(1, 2.*s, 2))/sqrt2pi
        const = (1 + (0.5)**(s + 0.5))/3.
        time = (2*const*K0/M/f)**(2./(3.+2.*s))
        x0 = numexpr.evaluate('I**s')
        x10 = -I * pisq * time
        x1 = numexpr.evaluate('exp(x10)') #np.exp(x10)
        #x1 = numexpr.evaluate('exp(-I*pisq*time)') #np.exp(x10)
        x2 = x0 * a2 * x1
        x3 = np.sum(x2)
        f = 2*pisq**s * x3
    return t-(2*M*sqrtpi*f)**(-0.4)


def speedTest():
    import os
    import cPickle
    with file('test_cython/data.pkl', 'rb') as F:
        enuerr = cPickle.load(F)
    min_e = min(enuerr)
    max_e = max(enuerr)
    ran_e = max_e - min_e
    for n in xrange(100):
        vbw_kde(data=enuerr, N=2**12, MIN=min_e-ran_e/2., MAX=max_e+ran_e/2, overfit_factor=1.00)


if __name__ == "__main__":
    speedTest()
