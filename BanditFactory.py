from Lin_SGMED_ver1 import *
from Lin_SGMED_ver2 import *
from Lin_IMED_ver1 import *
from Lin_ZHU import *
from OFUL import *
from Bandit_Env import *

def bandit_factory(name, X, R, S,n):
    K, d = X.shape
    if (name == "OFUL"):
        opt = {
            'X': X,
            'lam': R**2/S**2,
            'R' : R,
            'S': S,
            'flags': {},
            'subsample_func' :None,
            'multiplier': 1.0,  # the multiplier to the radius_sq,
            'subsample_rate' : 1.0
        };
        algo = Oful(**opt)
        return algo
    elif (name == "Lin-SGMED-1"):
        opt = {
            'X': X,
            'lam': ((R**2)*d)/S**2,
            'R' : R,
            'S': S,
            'flags': {"version":1}
        };
        algo = Lin_SGMED(**opt)
        return algo
    elif (name == "Lin-SGMED-2"):
        opt = {
            'X': X,
            'lam': ((R**2)*d)/S**2,
            'R' : R,
            'S': S,
            'flags': {"version":2}
        };
        algo = Lin_SGMED(**opt)
        return algo
    elif (name == "Lin-IMED-1"):
        opt = {
            'X': X,
            'lam': ((R**2))/S**2,
            'R' : R,
            'S': S,
            'flags': {},
            'subsample_func' :None,
            'multiplier': 1.0,  # the multiplier to the radius_sq,
            'subsample_rate' : 1.0
        };
        algo = Lin_IMED(**opt)
        return algo
    elif (name == "LinZHU"):
        opt = {
            'X': X,
            'lam': ((R**2))/S**2,
            'R' : R,
            'S': S,
            'flags': {},
            'N':n
        };
        algo = Lin_ZHU(**opt)
        return algo
    else:
        raise NotImplementedError()
