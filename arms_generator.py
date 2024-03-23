import numpy as np


def sample_spherical(npoints, ndim):
    vec = np.random.randn(ndim, npoints)
    vec /= (np.linalg.norm(vec, axis=0))
    #print((np.linalg.norm(vec, axis=0)))
    return vec.T

def sample_random(n,d):
    return np.random.rand(n, d)


