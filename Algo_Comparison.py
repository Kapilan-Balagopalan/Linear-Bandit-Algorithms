from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt


import numpy.random as ra
import numpy.linalg as la

from Lin_SGMED_ver1 import *

from OFUL import *

def init_end_of_optimism(eps):
    #np.random.seed(seed)
    noise_sigma = 0.1
    delta = 0.01
    S = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 3
    sVal_horizon = n = 1000000
    sVal_lambda = d
    mVal_I = np.eye(sVal_dimension)
    mVal_lvrg_scr_orgn = sVal_lambda*mVal_I
    sVal_arm_set = A = sample_end_of_optimism(eps)
    theta_true = A[0,:]
    #print(theta_true.shape)
    #print(A.shape)
    #theta_true = S*(theta_true/ (np.linalg.norm(theta_true, axis=0)))
    best_arm = A[0,:]
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, sVal_arm_set, theta_true,\
           noise_sigma, delta, S, best_arm


eps = 0.005
d, K ,n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, X , theta_true,noise_sigma,delta,S,best_arm = init_end_of_optimism(eps)
R = noise_sigma
linSGMED_inst = Lin_SGMED(X, sVal_lambda , R , S , flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)
OFUL_inst = Oful(X, sVal_lambda , R , S , flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)



acc_regret_linSGMED = 0
acc_regret_OFUL  = 0
acc_regret_arr_linSGMED = np.zeros(n)
acc_regret_arr_OFUL = np.zeros(n)

for t in range(n):
    x_t_linSGMED, radius_sq_linSGMED = linSGMED_inst.next_arm()
    x_t_OFUL, radius_sq_OFUL = OFUL_inst.next_arm()

    inst_regret_linSGMED = calc_regret(x_t_linSGMED , theta_true , X)
    inst_regret_OFUL = calc_regret(x_t_OFUL , theta_true , X)

    acc_regret_linSGMED = acc_regret_linSGMED  + inst_regret_linSGMED
    acc_regret_OFUL = acc_regret_OFUL + inst_regret_OFUL

    acc_regret_arr_linSGMED[t] = acc_regret_linSGMED
    acc_regret_arr_OFUL[t] = acc_regret_OFUL
    # print(Arm_t)
    reward_t_linSGMED  = receive_reward(x_t_linSGMED , theta_true, noise_sigma,X)
    reward_t_OFUL = receive_reward(x_t_OFUL, theta_true, noise_sigma, X)

    linSGMED_inst.update(x_t_linSGMED, reward_t_linSGMED)
    OFUL_inst.update(x_t_OFUL, reward_t_OFUL)



plt.plot(np.arange(n), acc_regret_arr_linSGMED , label="Lin-SGMED")
plt.plot(np.arange(n), acc_regret_arr_OFUL , label="OFUL")
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time")
plt.legend()
plt.show()

