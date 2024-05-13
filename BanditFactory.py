from Lin_SGMED import *
from Lin_IMED import *
from Lin_ZHU import *
from OFUL import *
from Bandit_Env import *
from Lin_EXP2 import *
from Lin_TS_Freq import *

def bandit_factory(test_type,name, X, R, S,n,opt_coeff,emp_coeff):
    K, d = X.shape
    if (name == "OFUL"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'flags': {"version": None,"type":test_type},
            'subsample_func' :None,
            'multiplier': 1.0,  # the multiplier to the radius_sq,
            'subsample_rate' : 1.0
        };
        algo = Oful(**opt)
        return algo
    if (name == "Lin-TS-Freq"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'N':n,
            'flags': {"version": None,"type":test_type},
        };
        algo = Lin_TS_FREQ(**opt)
        return algo
    if (name == "EXP2"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'N':n,
            'flags': {"version": None,"type":test_type},
        };
        algo = Lin_EXP2(**opt)
        return algo
    elif (name == "Lin-SGMED-1"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'opt_coeff' : opt_coeff,
            'emp_coeff' : emp_coeff,
            'flags': {"version":1,"type":test_type}
        };
        algo = Lin_SGMED(**opt)
        return algo
    elif (name == "Lin-SGMED-2"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'opt_coeff' : opt_coeff,
            'emp_coeff' : emp_coeff,
            'flags': {"version":2, "type":test_type}
        };
        algo = Lin_SGMED(**opt)
        return algo
    elif (name == "Lin-IMED-1"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'flags': {"version": 1, "type":test_type}
        };
        algo = Lin_IMED(**opt)
        return algo
    elif (name == "Lin-IMED-3"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'flags': {"version": 3, "type":test_type}
        };
        algo = Lin_IMED(**opt)
        return algo
    elif (name == "LinZHU"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'flags': {"version":"fixed", "type":test_type},
            'N':n
        };
        algo = Lin_ZHU(**opt)
        return algo
    elif (name == "LinZHU-AT"):
        opt = {
            'X': X,
            'R' : R,
            'S': S,
            'flags': {"version":"anytime", "type":test_type},
            'N':n
        };
        algo = Lin_ZHU(**opt)
        return algo
    else:
        raise NotImplementedError()
