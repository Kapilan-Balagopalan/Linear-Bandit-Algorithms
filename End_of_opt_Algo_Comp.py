from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

from BanditFactory import *
import ipdb

from datetime import datetime

import os

from tqdm import tqdm

n_algo = 9
algo_list = [None] * n_algo
algo_names = ["OFUL","Lin-TS-Freq","LinZHU","LinZHU-AT","Lin-IMED-1","Lin-IMED-3","LinMED","LinMED","LinMED"]
algo_names_plot = ["OFUL","Lin-TS-Freq","LinZHU","LinZHU-AT","Lin-IMED-1","Lin-IMED-3","LinMED-99","LinMED-90","LinMED-50"]
test_type = "Sphere"
emp_coeff = [0.99, 0.9, 0.5]
opt_coeff = [0.005, 0.05, 0.25]

n_cpu = 10
n_trials = 10

algo_color = ["r","b","g", "c","m","k", "gray", "black", "orange"]

def init_end_of_optimism(eps):
    #np.random.seed(seed)
    noise_sigma = 0.1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 3
    sVal_horizon = n = 10000
    sVal_arm_set = A = sample_end_of_optimism(eps)
    theta_true = A[0,:]
    best_arm = A[0,:]

    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_arm_set, theta_true,noise_sigma, delta, S_true, best_arm


eps = 0.005



d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm = init_end_of_optimism(eps)
cum_regret_arr = np.zeros((n_trials, n, n_algo))

Noise_Mismatch = 1
Norm_Mismatch = 1
delay_switch = False
reward_delay = 20
R_true = noise_sigma

for j in tqdm(range(n_trials)):
    i = 0
    for name in algo_names:
        if (i >= 6):
            algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                          opt_coeff[i - 6], emp_coeff[i - 6], 1000, delay_switch, reward_delay)
        else:
            algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                          opt_coeff[0], emp_coeff[0], 1000, delay_switch, reward_delay)
        i = i + 1
    for i in range(n_algo):
        cum_regret = 0
        for t in range(n):
            arm = algo_list[i].next_arm(X)
            inst_regret = calc_regret(arm, theta_true, X)
            cum_regret = cum_regret + inst_regret
            cum_regret_arr[j][t][i] = cum_regret
            reward = receive_reward(arm, theta_true, noise_sigma, X)
            algo_list[i].update_delayed(X[arm, :], reward)
        

t_alpha = 0.5


cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()
print(cum_regret_mean.shape)

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


i=0

for name in algo_names:
    plt.plot(np.arange(n), cum_regret_mean[:,i],color = algo_color[i] , label=algo_names_plot[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i],color = algo_color[i], alpha=.3)
    i = i + 1



now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] + '0005'+ date_time + '.npy'
current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/'
completeName = os.path.join(prefix , file_name)

with open(completeName, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)

plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time 0.005")
plt.legend()
plt.savefig(prefix + 'EOPT1.eps',format = 'eps',dpi=300)
plt.savefig(prefix + 'EOPT1.png',format = 'png')
plt.show() 

