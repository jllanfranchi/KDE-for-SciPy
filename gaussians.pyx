# -*- coding: utf-8 -*-

cimport cython
from cython.parallel import prange
from libc.math cimport exp, sqrt, M_PI

cdef double sqrtpi = sqrt(M_PI)
cdef double sqrt2pi = sqrt(2*M_PI)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian(double[::1] outbuf,
                 double[::1] x,
                 double mu,
                 double sigma,
                 int threads=4):
    cdef double twosigma2 = 2*(sigma*sigma)
    cdef double sqrt2pisigma = sqrt2pi * sigma
    cdef double xlessmu
    cdef double x1
    cdef Py_ssize_t i
    for i in prange(outbuf.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        xlessmu = x[i]-mu
        x1 = -xlessmu*xlessmu / twosigma2
        outbuf[i] += exp(x1) / sqrt2pisigma


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussians(double[::1] outbuf,
                  double[::1] x,
                  double[::1] mu,
                  double[::1] sigma,
                  int threads=4):
    cdef double twosigma2
    cdef double sqrt2pisigma
    cdef double xlessmu
    cdef double x1
    cdef Py_ssize_t i, gaus_n
    # NOTE that the order of the loops is important, as
    # updating the outbuf is NOT thread safe!
    for i in prange(x.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        for gaus_n in xrange(mu.shape[0]):
            twosigma2 = 2*(sigma[gaus_n]*sigma[gaus_n])
            sqrt2pisigma = sqrt2pi * sigma[gaus_n]
            xlessmu = x[i]-mu[gaus_n]
            x1 = -(xlessmu*xlessmu) / twosigma2
            outbuf[i] += exp(x1) / sqrt2pisigma

# TODO: For another day...
#def fixed_point(t, M, I, a2):
#    cdef int l = 7
#    cdef Py_ssize_t i
#    for i in prange(I.shape[0],
#                         nogil=True,
#                         num_threads=threads,
#                         schedule='static'):
#    x0 = numexpr.evaluate('I**l')
#    x1 = numexpr.evaluate('exp(-I*pisq*t)')
#    x2 = x0 * a2 * x1
#    x3 = np.sum(x2)
#    f = 2*pisq**l * x3
#    for s in xrange(l, 1, -1):
#        K0 = np.prod(np.arange(1, 2.*s, 2))/sqrt2pi
#        const = (1 + (0.5)**(s + 0.5))/3.
#        time = (2*const*K0/M/f)**(2./(3.+2.*s))
#        x0 = numexpr.evaluate('I**s')
#        x10 = -I * pisq * time
#        x1 = numexpr.evaluate('exp(x10)') #np.exp(x10)
#        #x1 = numexpr.evaluate('exp(-I*pisq*time)') #np.exp(x10)
#        x2 = x0 * a2 * x1
#        x3 = np.sum(x2)
#        f = 2*pisq**s * x3
#    return t-(2*M*sqrtpi*f)**(-0.4)
