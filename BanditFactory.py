#from Lin_SGMED import *
from Lin_IMED import *
from Lin_ZHU import *
from OFUL import *
from Bandit_Env import *
from Lin_EXP2 import *
from Lin_TS_Freq import *
from Lin_SGMED_NOPT import *
from LinMED import *

def bandit_factory(test_type,name, X, R, S,n,opt_coeff,emp_coeff,n_mc_samples):
    K, d = X.shape
    opt_gen = {
        'X': X,
        'R': R,
        'S': S,
        'N': n,
        'flags': {"version": 1, "type": test_type}
    };
    if (name == "OFUL"):
        opt_oful = {}
        opt_oful.update(opt_gen)
        opt_oful['flags'] = {"version": None,"type":test_type}
        algo = Oful(**opt_oful)
        return algo
    if (name == "Lin-TS-Freq"):
        opt_ts = {'n_mc_samples':n_mc_samples}
        opt_ts.update(opt_gen)
        opt_ts['flags'] = {"version": None, "type": test_type}
        algo = Lin_TS_FREQ(**opt_ts)
        return algo
    if (name == "EXP2"):
        opt_exp = {}
        opt_exp.update(opt_gen)
        opt_exp['flags'] = {"version": None, "type": test_type}
        algo = Lin_EXP2(**opt_exp)
        return algo
    elif (name == "Lin-SGMED-1"):
        opt_sgmed1 = { 'opt_coeff': opt_coeff,
        'emp_coeff': emp_coeff}
        opt_sgmed1.update(opt_gen)
        algo = Lin_SGMED(**opt_sgmed1)
        return algo
    elif (name == "Lin-SGMED-NOPT"):
        opt_sgmednopt = {'opt_coeff': opt_coeff,
                      'emp_coeff': emp_coeff}
        opt_sgmednopt .update(opt_gen)
        opt_sgmednopt['flags'] = {"version": 2, "type": test_type}
        algo = Lin_SGMED_NOPT(**opt_sgmednopt)
        return algo
    elif (name == "Lin-SGMED-2"):
        opt_sgmed2 = {'opt_coeff': opt_coeff,
                         'emp_coeff': emp_coeff}
        opt_sgmed2.update(opt_gen)
        opt_sgmed2['flags'] = {"version": 2, "type": test_type}
        algo = Lin_SGMED(**opt_sgmed2)
        return algo
    elif (name == "LinMED"):
        opt_med = {'opt_coeff': opt_coeff,
                      'emp_coeff': emp_coeff}
        opt_med.update(opt_gen)
        opt_med['flags'] = {"version": 2, "type": test_type}
        algo = Lin_SGMED(**opt_med)
        return algo
    elif (name == "Lin-IMED-1"):
        opt_imed1 = {}
        opt_imed1.update(opt_gen)
        opt_imed1['flags'] = {"version": 1, "type": test_type}
        algo = Lin_IMED(**opt_imed1)
        return algo
    elif (name == "Lin-IMED-3"):
        opt_imed1 = {}
        opt_imed1.update(opt_gen)
        opt_imed1['flags'] = {"version": 3, "type": test_type}
        algo = Lin_IMED(**opt_imed1)
        return algo
    elif (name == "LinZHU"):
        opt_zhu = {}
        opt_zhu .update(opt_gen)
        opt_zhu['flags'] = {"version": "fixed", "type": test_type}
        algo = Lin_ZHU(**opt_zhu)
        return algo
    elif (name == "LinZHU-AT"):
        opt_zhuat = {}
        opt_zhuat.update(opt_gen)
        opt_zhuat['flags'] = {"version": "anytime", "type": test_type}
        algo = Lin_ZHU(**opt_zhuat)
        return algo
    else:
        raise NotImplementedError()
