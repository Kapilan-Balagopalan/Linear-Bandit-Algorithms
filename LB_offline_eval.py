from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import numpy.random as ra
import numpy.linalg as la

from BanditFactory import *

import ipdb

from datetime import datetime

import os

from tqdm import tqdm

def init_offline_eval_exp():
    #np.random.seed(seed)
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 2
    sVal_horizon = n = 10000
    n_trials = 2000
    sVal_arm_set = A = sample_offline_eval_experiment(K)
    theta_true = np.zeros((d,1))
    theta_true[0][0] = 1
    best_arm = A[K-1,:]

    return sVal_dimension, sVal_arm_size,sVal_horizon , sVal_arm_set, theta_true,noise_sigma, delta, S_true, best_arm,n_trials


d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm,n_trials = init_offline_eval_exp()

n_algo = 2

algo_list = [None] * n_algo

algo_names = ["LinMED", "Lin-TS-Freq"]

Noise_Mismatch = 1
Norm_Mismatch = 1
R_true = noise_sigma

cum_regret_arr = np.zeros((n_trials, n, n_algo))

offline_logged_data = np.zeros((n_trials, n, n_algo, 3))

test_type = "Sphere"

emp_coeff = [0.99, 0.9, 0.5]
opt_coeff = [0.005, 0.1, 0.5]

n_mc_samples = 1000
prob_min_thresh = 0.0005


mu_hat = np.zeros((n_trials, n_algo, 1))
for j in tqdm(range(n_trials)):
    i = 0
    for name in algo_names:
        algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n,d,
                                          opt_coeff[2], emp_coeff[2], n_mc_samples,False, 5)
        i = i + 1

    i = 0
    for i in range(n_algo):
        cum_mu_hat = 0
        for t in range(n):
            arm = algo_list[i].next_arm(X)
            offline_logged_data[j][t][i][0] = arm
            prob_chosen = algo_list[i].get_probability_arm(X)
            offline_logged_data[j][t][i][1] = prob_chosen
            inst_regret = calc_regret(arm, theta_true, X)
            reward = receive_reward(arm, theta_true, noise_sigma, X)
            offline_logged_data[j][t][i][2] = reward
            if (i==1):
                cum_mu_hat = cum_mu_hat + reward / np.maximum(prob_min_thresh, prob_chosen)
            else:
                cum_mu_hat = cum_mu_hat + reward / prob_chosen

            algo_list[i].update_delayed(X[arm,:], reward)

        mu_hat[j][i][0] = cum_mu_hat / (K * n)
        cum_mu_hat = 0

current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/'

script_name = os.path.basename(__file__)
file_name = 'offline_eval_data.npy'

completeName = os.path.join(prefix, file_name)

with open(completeName, 'wb') as f:

    np.save(f, mu_hat)


uniform_average_regret = np.matmul(np.mean(X,axis = 0), theta_true)


plt.hist(mu_hat[:, 0, 0], bins=100, color='skyblue', alpha=0.5, edgecolor='black', label='LinMED')
plt.axvline(mu_hat[:, 0, 0].mean(), color='skyblue', linestyle='dashed', linewidth=1)
plt.hist(mu_hat[:, 1, 0], bins=100, color='green', alpha=0.5, edgecolor='black', label='LinTS')
plt.axvline(mu_hat[:, 1, 0].mean(), color='green', linestyle='dashed', linewidth=1)
plt.axvline(x=uniform_average_regret, color='black', label='axvline - full height')


print(mu_hat[:, 0, 0].mean())
print(mu_hat[:, 1, 0].mean())


plt.xlabel('Expected Reward')
plt.ylabel('Trials')

plt.title('Histogram of Expected rewards vs trials')
plt.legend()
plt.savefig(prefix + 'Img-2.png', format='png')
plt.show()






