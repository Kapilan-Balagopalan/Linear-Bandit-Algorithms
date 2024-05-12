from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt


from BanditFactory import *
import ipdb

from datetime import datetime

import os 

from tqdm import tqdm


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


eps = 0.007

d, K, n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, X, theta_true, noise_sigma, delta, S, best_arm = init_end_of_optimism(eps)

n_algo = 2

algo_list = [None]*n_algo
algo_names = ["OFUL", "Lin-SGMED-2" ]
n_trials = 1

cum_regret_arr=  np.zeros((n_trials,n,n_algo))
pulled_arm_index = np.zeros((n,n_algo))
test_type = "EOPT"
opt_coeff = 0.5
emp_coeff = 0.5

for j in tqdm(range(n_trials)):
    seed = 15751 + j
    np.random.seed(seed)
    R= noise_sigma
    i = 0
    for name in algo_names:
        algo_list[i] = bandit_factory(test_type,name,X,R,S,n,opt_coeff,emp_coeff)
        i = i+1

    cum_regret = 0
    for i in range(n_algo):
        cum_regret = 0
        for t in range(n):
            arm = algo_list[i].next_arm()
            pulled_arm_index[t][i] = arm
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



i=0

for name in algo_list:
    plt.plot(np.arange(n), cum_regret_mean[:,i] , label=algo_names[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i], alpha=.3)
    i = i + 1


arm_frq = np.zeros((K, n,n_algo))




now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'

with open(file_name, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)
    np.save(f,pulled_arm_index)

        
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time")
plt.legend()
plt.show()

for t in range(n-1):
    for i in range(n_algo):
        if (pulled_arm_index[t][i] == 0):
            arm_frq[0][t+1][i] = arm_frq[0][t][i] + 1
            arm_frq[1][t+1][i] = arm_frq[1][t][i] 
            arm_frq[2][t+1][i] = arm_frq[2][t][i] 
        if (pulled_arm_index[t][i] == 1):
            arm_frq[0][t+1][i] = arm_frq[0][t][i] 
            arm_frq[1][t+1][i] = arm_frq[1][t][i] + 1
            arm_frq[2][t+1][i] = arm_frq[2][t][i] 
        if (pulled_arm_index[t][i] == 2):
            arm_frq[0][t+1][i] = arm_frq[0][t][i] 
            arm_frq[1][t+1][i] = arm_frq[1][t][i] 
            arm_frq[2][t+1][i] = arm_frq[2][t][i] + 1


j=0
arm_names = ["best", "worst", "second best" ]

for name in algo_list:
    for i in range(K):
        labels = algo_names[j] + arm_names[i]
        plt.plot(np.arange(n), arm_frq[i,:,j] , label=labels)
    j = j + 1


print(arm_frq[1][n-1][0])
print(arm_frq[1][n-1][1])
print(arm_frq[0][n-1][0])
print(arm_frq[0][n-1][1])
print(arm_frq[2][n-1][0])
print(arm_frq[2][n-1][1])


plt.xlabel("Time")
plt.ylabel("freq")
plt.title("Freq of arm pull with time")
plt.legend()
plt.show()


