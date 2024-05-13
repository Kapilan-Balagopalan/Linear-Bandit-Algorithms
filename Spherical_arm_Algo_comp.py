from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt


import numpy.random as ra
import numpy.linalg as la


from BanditFactory import *

import ipdb

from datetime import datetime

import os 


from tqdm import tqdm
 
 


def init(seed,K,n,d):
    np.random.seed(seed)
    noise_sigma = 0.1
    delta = 0.01
    S = 1
    sVal_dimension = d
    sVal_arm_size = K
    sVal_horizon = n
    sVal_lambda = d
    mVal_I = np.eye(sVal_dimension)
    mVal_lvrg_scr_orgn = sVal_lambda*mVal_I
    sVal_arm_set = A = sample_spherical(sVal_arm_size,sVal_dimension)
    theta_true = np.random.randn(d, 1)
    #print(theta_true.shape)
    #print(A.shape)
    theta_true = S*(theta_true/ (np.linalg.norm(theta_true, axis=0)))
    best_arm = np.argmax(np.matmul(A, theta_true))
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, sVal_arm_set, theta_true,\
           noise_sigma, delta, S, best_arm






K = 100
n = 10000
d = 10



n_algo = 9

algo_list = [None]*n_algo
algo_names = ["EXP2","OFUL","Lin-SGMED-1","Lin-SGMED-2","Lin-IMED-1","LinZHU","LinZHU-AT","Lin-IMED-3", "Lin-TS-Freq"]
#algo_names = ["OFUL", "Lin-TS-Freq"]
n_trials = 10

cum_regret_arr=  np.zeros((n_trials,n,n_algo))

test_type = "Sphere"
opt_coeff = 1/16
emp_coeff = 15/32

for j in tqdm(range(n_trials)):
    #seed = np.random.randint(1, 15751)
    seed = 15751 + j
    d, K, n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, X, theta_true, noise_sigma, delta, S, best_arm = init(seed, K, n,
                                                                                                            d)
    R= noise_sigma
    i = 0
    for name in algo_names:
        algo_list[i] = bandit_factory(test_type,name,X,R,S,n,opt_coeff,emp_coeff)
        i = i+1

    cum_regret = 0
    for i in range(n_algo):
        cum_regret = 0
        for t in range(n):
            arm  = algo_list[i].next_arm()
            inst_regret = calc_regret(arm, theta_true, X)
            cum_regret = cum_regret + inst_regret
            cum_regret_arr[j][t][i] =  cum_regret
            reward = receive_reward(arm, theta_true, noise_sigma, X)
            algo_list[i].update(arm,reward)
        

t_alpha = 1


cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()
print(cum_regret_mean.shape)

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'

with open(file_name, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)

i=0

for name in algo_list:
    plt.plot(np.arange(n), cum_regret_mean[:,i] , label=algo_names[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i], alpha=.3)
    i = i + 1

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time")
plt.legend()
plt.show()