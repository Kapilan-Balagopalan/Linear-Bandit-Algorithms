from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt


import numpy.random as ra
import numpy.linalg as la

from datetime import datetime

import os 

from BanditFactory import *

import ipdb



from tqdm import tqdm
 
 


def init(seed,K,n,d):
    np.random.seed(seed)
    noise_sigma = 1
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






K = 50
n = 5000
d = 10



n_algo = 2

algo_list = [None]*n_algo
algo_names = ["Lin-SGMED-1","Lin-SGMED-2" ]
#algo_names = ["LinZHU" ]
n_trials = 15

test_type = "Sphere"
n_exp_ind = 16
exp_ind  = np.arange(n_exp_ind)

cum_regret_arr=  np.zeros((n_trials,n,n_algo,n_exp_ind))



opt_coeff_arr = np.power(0.5,np.arange(n_exp_ind))
emp_coeff = 0

final_regret_arr = np.zeros((n_exp_ind,n_algo))
#print(opt_coeff)

for k in range(len(opt_coeff_arr)):
    opt_coeff = opt_coeff_arr[k]
    emp_coeff = (1 - opt_coeff)/2
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
                cum_regret_arr[j][t][i][k] =  cum_regret
                reward = receive_reward(arm, theta_true, noise_sigma, X)
                algo_list[i].update(arm,reward)

t_alpha = 1


final_regret_arr = cum_regret_arr[:,n-1,:,:] 


final_regret_arr_mean = np.sum(final_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(final_regret_arr, axis=0, ddof=1)

final_regret_confidence_up = final_regret_arr_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
final_regret_confidence_down = final_regret_arr_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
#cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
#cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()
#print(cum_regret_mean.shape)

#cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
#cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'

with open(file_name, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)

with open(file_name, 'rb') as f:

    a = np.load(f)
    b = np.load(f)

i=0


for name in algo_list:
    plt.plot(np.arange(n_exp_ind), final_regret_arr_mean[i,:] , label=algo_names[i])
    plt.fill_between(np.arange(n_exp_ind),final_regret_confidence_up[i,:] , final_regret_confidence_down[i,:] , alpha=.3)
    i = i + 1



# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Optimal design coefficient")
plt.ylabel("Regret")
plt.title("Regret with optimal design coefficient")
plt.legend()
plt.show()