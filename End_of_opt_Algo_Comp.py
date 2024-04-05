from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt


import numpy.random as ra
import numpy.linalg as la

from Lin_SGMED_ver1 import *
from Lin_SGMED_ver2 import *
from Lin_IMED_ver1 import *
from Lin_ZHU import *
from OFUL import *


def init_end_of_optimism(eps):
    #np.random.seed(seed)
    noise_sigma = 0.1
    delta = 0.01
    S = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 3
    sVal_horizon = n = 100000
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


eps = 0.01

d, K, n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, X, theta_true, noise_sigma, delta, S, best_arm = init_end_of_optimism(eps)
R = noise_sigma
linSGMED_inst = Lin_SGMED(X, sVal_lambda, R, S, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)
linSGMED2_inst = Lin_SGMED2(X, sVal_lambda, R, S, flags=None, subsample_func=None, subsample_rate=1.0,
                                multiplier=1.0)
lam = 1 / (S ** 2)
linIMED_inst = Lin_IMED(X, sVal_lambda, R, S, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)
linZHU_inst = Lin_ZHU(X, sVal_lambda, R, S,n, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)
OFUL_inst = Oful(X, lam, R, S, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)





acc_regret_linSGMED = 0
acc_regret_linIMED = 0
acc_regret_linZHU = 0
acc_regret_linSGMED2 = 0
acc_regret_OFUL  = 0


n_trials = 10

acc_regret_arr_linSGMED = np.zeros((n_trials,n))
acc_regret_arr_linIMED = np.zeros((n_trials,n))
acc_regret_arr_linZHU = np.zeros((n_trials,n))
acc_regret_arr_linSGMED2 = np.zeros((n_trials,n))
acc_regret_arr_OFUL = np.zeros((n_trials,n))

for j in range(n_trials):
    eps = 0.01

    d, K, n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, X, theta_true, noise_sigma, delta, S, best_arm = init_end_of_optimism(
        eps)
    R = noise_sigma
    linSGMED_inst = Lin_SGMED(X, sVal_lambda, R, S, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)
    linSGMED2_inst = Lin_SGMED2(X, sVal_lambda, R, S, flags=None, subsample_func=None, subsample_rate=1.0,
                                multiplier=1.0)
    lam = 1 / (S ** 2)
    linIMED_inst = Lin_IMED(X, sVal_lambda, R, S, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)
    linZHU_inst = Lin_ZHU(X, sVal_lambda, R, S, n, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)
    OFUL_inst = Oful(X, lam, R, S, flags=None, subsample_func=None, subsample_rate=1.0, multiplier=1.0)


    acc_regret_linSGMED = 0
    acc_regret_linIMED = 0
    acc_regret_linZHU = 0
    acc_regret_linSGMED2 = 0
    acc_regret_OFUL = 0

    for t in range(n):
        x_t_linSGMED, radius_sq_linSGMED = linSGMED_inst.next_arm()
        x_t_linIMED, radius_sq_linIMED = linIMED_inst.next_arm()
        x_t_linZHU, radius_sq_linZHU = linZHU_inst.next_arm()
        x_t_linSGMED2, radius_sq_linSGMED2 = linSGMED2_inst.next_arm()
        x_t_OFUL, radius_sq_OFUL = OFUL_inst.next_arm()

        inst_regret_linSGMED = calc_regret(x_t_linSGMED, theta_true, X)
        inst_regret_linIMED = calc_regret(x_t_linIMED, theta_true, X)
        inst_regret_linZHU = calc_regret(x_t_linZHU, theta_true, X)
        inst_regret_linSGMED2 = calc_regret(x_t_linSGMED2, theta_true, X)
        inst_regret_OFUL = calc_regret(x_t_OFUL, theta_true, X)

        acc_regret_linSGMED = acc_regret_linSGMED + inst_regret_linSGMED
        acc_regret_linIMED = acc_regret_linIMED + inst_regret_linIMED
        acc_regret_linZHU = acc_regret_linZHU + inst_regret_linZHU
        acc_regret_linSGMED2 = acc_regret_linSGMED2 + inst_regret_linSGMED2
        acc_regret_OFUL = acc_regret_OFUL + inst_regret_OFUL

        acc_regret_arr_linSGMED[j][t] = acc_regret_linSGMED
        acc_regret_arr_linIMED[j][t] = acc_regret_linIMED
        acc_regret_arr_linZHU[j][t] = acc_regret_linZHU
        acc_regret_arr_linSGMED2[j][t] = acc_regret_linSGMED2
        acc_regret_arr_OFUL[j][t] = acc_regret_OFUL
        # print(Arm_t)
        reward_t_linSGMED = receive_reward(x_t_linSGMED, theta_true, noise_sigma, X)
        reward_t_linIMED = receive_reward(x_t_linIMED, theta_true, noise_sigma, X)
        reward_t_linZHU = receive_reward(x_t_linZHU, theta_true, noise_sigma, X)
        reward_t_linSGMED2 = receive_reward(x_t_linSGMED2, theta_true, noise_sigma, X)
        reward_t_OFUL = receive_reward(x_t_OFUL, theta_true, noise_sigma, X)

        linSGMED_inst.update(x_t_linSGMED, reward_t_linSGMED)
        linIMED_inst.update(x_t_linIMED, reward_t_linIMED)
        linZHU_inst.update(x_t_linZHU, reward_t_linZHU)
        linSGMED2_inst.update(x_t_linSGMED2, reward_t_linSGMED2)
        OFUL_inst.update(x_t_OFUL, reward_t_OFUL)

t_alpha = 0.5

acc_regret_arr_linSGMED_mean = np.sum(acc_regret_arr_linSGMED, axis=0) / n_trials
acc_regret_arr_linSGMED_std = np.std(acc_regret_arr_linSGMED, axis=0, ddof=1)
acc_regret_arr_linSGMED2_mean = np.sum(acc_regret_arr_linSGMED2, axis=0) / n_trials
acc_regret_arr_linSGMED2_std = np.std(acc_regret_arr_linSGMED2, axis=0, ddof=1)
acc_regret_arr_linIMED_mean = np.sum(acc_regret_arr_linIMED, axis=0) / n_trials
acc_regret_arr_linIMED_std = np.std(acc_regret_arr_linIMED, axis=0, ddof=1)
acc_regret_arr_linZHU_mean = np.sum(acc_regret_arr_linZHU, axis=0) / n_trials
acc_regret_arr_linZHU_std = np.std(acc_regret_arr_linZHU, axis=0, ddof=1)

acc_regret_arr_linSGMED_confidence_up = acc_regret_arr_linSGMED_mean + (
            t_alpha * acc_regret_arr_linSGMED_std) / np.sqrt(n_trials)
acc_regret_arr_linSGMED_confidence_down = acc_regret_arr_linSGMED_mean - (
            t_alpha * acc_regret_arr_linSGMED_std) / np.sqrt(n_trials)

acc_regret_arr_linIMED_confidence_up = acc_regret_arr_linIMED_mean + (t_alpha * acc_regret_arr_linIMED_std) / np.sqrt(
    n_trials)
acc_regret_arr_linIMED_confidence_down = acc_regret_arr_linIMED_mean - (t_alpha * acc_regret_arr_linIMED_std) / np.sqrt(
    n_trials)

acc_regret_arr_linZHU_confidence_up = acc_regret_arr_linZHU_mean + (t_alpha * acc_regret_arr_linZHU_std) / np.sqrt(
    n_trials)
acc_regret_arr_linZHU_confidence_down = acc_regret_arr_linZHU_mean - (t_alpha * acc_regret_arr_linZHU_std) / np.sqrt(
    n_trials)

acc_regret_arr_linSGMED2_confidence_up = acc_regret_arr_linSGMED2_mean + (
            t_alpha * acc_regret_arr_linSGMED2_std) / np.sqrt(n_trials)
acc_regret_arr_linSGMED2_confidence_down = acc_regret_arr_linSGMED2_mean - (
            t_alpha * acc_regret_arr_linSGMED2_std) / np.sqrt(n_trials)

acc_regret_arr_OFUL_mean = np.sum(acc_regret_arr_OFUL, axis=0) / n_trials
acc_regret_arr_OFUL_std = np.std(acc_regret_arr_OFUL, axis=0, ddof=1)
acc_regret_arr_OFUL_confidence_up = acc_regret_arr_OFUL_mean + (t_alpha * acc_regret_arr_OFUL_std) / np.sqrt(n_trials)
acc_regret_arr_OFUL_confidence_down = acc_regret_arr_OFUL_mean - (t_alpha * acc_regret_arr_OFUL_std) / np.sqrt(n_trials)

plt.plot(np.arange(n), acc_regret_arr_linSGMED_mean, label="Lin-SGMED")
plt.fill_between(np.arange(n), acc_regret_arr_linSGMED_confidence_down, acc_regret_arr_linSGMED_confidence_up)
plt.plot(np.arange(n), acc_regret_arr_linIMED_mean, label="Lin-IMED")
plt.fill_between(np.arange(n), acc_regret_arr_linIMED_confidence_down, acc_regret_arr_linIMED_confidence_up)
plt.plot(np.arange(n), acc_regret_arr_linZHU_mean, label="Lin-ZHU")
plt.fill_between(np.arange(n), acc_regret_arr_linZHU_confidence_down, acc_regret_arr_linZHU_confidence_up)
plt.plot(np.arange(n), acc_regret_arr_linSGMED2_mean, label="Lin-SGMED2")
plt.fill_between(np.arange(n), acc_regret_arr_linSGMED2_confidence_down, acc_regret_arr_linSGMED2_confidence_up)
plt.plot(np.arange(n), acc_regret_arr_OFUL_mean, label="OFUL")
plt.fill_between(np.arange(n), acc_regret_arr_OFUL_confidence_down, acc_regret_arr_OFUL_confidence_up)
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time for End of Optimism")

plt.legend()
plt.show()

