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

def init_offline_eval_exp():
    #np.random.seed(seed)
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 2
    sVal_horizon = n = 100

    sVal_arm_set = A = sample_offline_eval_experiment()
    theta_true = np.zeros((d,1))
    theta_true[0][0] = 1
    best_arm = A[0,:]

    return sVal_dimension, sVal_arm_size,sVal_horizon , sVal_arm_set, theta_true,\
           noise_sigma, delta, S_true, best_arm


d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm = init_offline_eval_exp()

n_algo = 2

algo_list = [None]*n_algo

algo_names = ["LinMED","Lin-TS-Freq"]


n_trials = 1000

Noise_Mismatch = 1
Norm_Mismatch = 1
R_true = noise_sigma

cum_regret_arr= np.zeros((n_trials,n,n_algo))

offline_logged_data = np.zeros((n_trials,n,n_algo,3))

test_type = "Sphere"

emp_coeff = [0.99,0.9,0.5]
opt_coeff = [0.005,0.05,0.25]

n_mc_samples = 1000
prob_min_thresh = 0.0001

mu_hat = np.zeros((n_trials,n_algo,1))

for j in tqdm(range(n_trials)):
    seed = 15751 + j
    i = 0
    for name in algo_names:
        algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n,
                                          opt_coeff[0], emp_coeff[0],n_mc_samples)
        i = i + 1

    i = 0
    for i in range(n_algo):
        cum_mu_hat = 0
        for t in range(n):
            arm = algo_list[i].next_arm()
            offline_logged_data[j][t][i][0] = arm
            prob_chosen = algo_list[i].get_probability_arm()
            offline_logged_data[j][t][i][1] = prob_chosen
            inst_regret = calc_regret(arm, theta_true, X)

            reward = receive_reward(arm, theta_true, noise_sigma, X)
            offline_logged_data[j][t][i][2] = reward
            cum_mu_hat = cum_mu_hat +  reward/np.maximum(prob_min_thresh,prob_chosen)
            algo_list[i].update(arm, reward)

        mu_hat[j][i][0] = cum_mu_hat/(K*n)
        cum_mu_hat = 0


plt.hist(mu_hat[:,0,0], bins=30, color='skyblue', edgecolor='black',label = 'LinMED' )
plt.hist(mu_hat[:,1,0], bins=30, color='green', edgecolor='black', label='LinTS')

print(mu_hat[:,0,0])
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
plt.legend()
plt.show()

