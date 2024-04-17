from Lin_SGMED import *
from Lin_IMED import *
from Lin_ZHU import *
from OFUL import *
from Bandit_Env import *

def bandit_factory(test_type,name, X, R, S,n,opt_coeff):
    K, d = X.shape
    if(test_type == "EOPT"):
        if (name == "OFUL"):
            opt = {
                'X': X,
                'lam': 1/S**2,
                'R' : R,
                'S': S,
                'flags': {"type":"EOPT"},
                'subsample_func' :None,
                'multiplier': 1.0,  # the multiplier to the radius_sq,
                'subsample_rate' : 1.0
            };
            algo = Oful(**opt)
            return algo
        elif (name == "Lin-SGMED-1"):
            opt = {
                'X': X,
                'lam': (d)/S**2,
                'R' : R,
                'S': S,
                'opt_coeff' : opt_coeff,
                'flags': {"version":1,"type":"EOPT"}
            };
            algo = Lin_SGMED(**opt)
            return algo
        elif (name == "Lin-SGMED-2"):
            opt = {
                'X': X,
                'lam': (d)/S**2,
                'R' : R,
                'S': S,
                'opt_coeff' : opt_coeff,
                'flags': {"version":2, "type":"EOPT"}
            };
            algo = Lin_SGMED(**opt)
            return algo
        elif (name == "Lin-IMED-1"):
            opt = {
                'X': X,
                'lam': (1)/S**2,
                'R' : R,
                'S': S,
                'flags': {"version": 1, "type":"EOPT"}
            };
            algo = Lin_IMED(**opt)
            return algo
        elif (name == "LinZHU"):
            opt = {
                'X': X,
                'lam': (1)/S**2,
                'R' : R,
                'S': S,
                'flags': {"version":"fixed", "type":"EOPT"},
                'N':n
            };
            algo = Lin_ZHU(**opt)
            return algo
        elif (name == "LinZHU-AT"):
            opt = {
                'X': X,
                'lam': (1)/S**2,
                'R' : R,
                'S': S,
                'flags': {"version":"anytime", "type":"EOPT"},
                'N':n
            };
            algo = Lin_ZHU(**opt)
            return algo
        else:
            raise NotImplementedError()
    elif(test_type == "Sphere"):
        if (name == "OFUL"):
            opt = {
                'X': X,
                'lam': R**2/S**2,
                'R' : R,
                'S': S,
                'flags': {"type":"Sphere"},
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
                'opt_coeff' : opt_coeff,
                'flags': {"version":1,"type":"Sphere"}
            };
            algo = Lin_SGMED(**opt)
            return algo
        elif (name == "Lin-SGMED-2"):
            opt = {
                'X': X,
                'lam': ((R**2)*d)/S**2,
                'R' : R,
                'S': S,
                'opt_coeff' : opt_coeff,
                'flags': {"version":2, "type":"Sphere"}
            };
            algo = Lin_SGMED(**opt)
            return algo
        elif (name == "Lin-IMED-1"):
            opt = {
                'X': X,
                'lam': ((R**2))/S**2,
                'R' : R,
                'S': S,
                'flags': {"version":1, "type":"Sphere"}
            };
            algo = Lin_IMED(**opt)
            return algo
        elif (name == "LinZHU"):
            opt = {
                'X': X,
                'lam': ((R**2))/S**2,
                'R' : R,
                'S': S,
                'flags': {"version":"fixed", "type":"Sphere"},
                'N':n
            };
            algo = Lin_ZHU(**opt)
            return algo
        elif (name == "LinZHU-AT"):
            opt = {
                'X': X,
                'lam': ((R**2))/S**2,
                'R' : R,
                'S': S,
                'flags': {"version":"anytime", "type":"Sphere"},
                'N':n
            };
            algo = Lin_ZHU(**opt)
            return algo
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
