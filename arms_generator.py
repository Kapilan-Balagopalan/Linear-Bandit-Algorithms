import numpy as np


def sample_spherical(npoints, ndim):
    vec = np.random.randn(ndim, npoints)
    vec /= (np.linalg.norm(vec, axis=0))
    #print((np.linalg.norm(vec, axis=0)))
    return vec.T

def sample_random(n,d):
    return np.random.rand(n, d)

def sample_end_of_optimism(eps):
    A = np.zeros((3,2))
    A[0, 0] = 1
    A[0, 1] = 0
    A[1, 0] = 0
    A[1, 1] = 1
    A[2, 0] = 1 - eps
    A[2, 1] = 2*eps
    return A
def sample_problem_dependent_experiment():
    A = np.zeros((2,2))
    A[0, 0] = 1
    A[0, 1] = 0
    A[1, 0] = 0
    A[1, 1] = 1
    return A

def sample_offline_eval_experiment():
    A = np.zeros((2,2))
    A[0, 0] = 0.8
    A[0, 1] = 0.6
    A[1, 0] = 0.75
    A[1, 1] = 0.65
    return A

def worse_case_scenario_experiment(K):
    A = np.zeros((K,2))
    A[0,0] = 1
    A[0,1] = 0
    for i in range(K-1):
        A[i+1,0] = 0
        A[i+1,1] = 1
    return A