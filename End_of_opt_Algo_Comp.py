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


n_algo = 6

algo_list = [None]*n_algo
algo_names = ["LinMED","LinMED", "LinMED","LinMED","LinMED", "LinMED"]
n_trials = 10


cum_regret_arr=  np.zeros((n_trials,n,n_algo))
pulled_arm_index = np.zeros((n,n_algo))
test_type = "EOPT"
emp_coeff = [0.99,0.9,0.5]
opt_coeff = [0.005,0.05,0.25]

Noise_Mismatch = 1
Norm_Mismatch = 1
n_mc_samples = 0

for j in tqdm(range(n_trials)):
    seed = 15751 + j
    np.random.seed(seed)
    R_true= noise_sigma
    i = 0
    for name in algo_names:
        if(i < 3):
            algo_list[i] = bandit_factory(test_type,name,X, R_true*Noise_Mismatch , S_true*Norm_Mismatch, n,opt_coeff[i],emp_coeff[i],n_mc_samples)
        else:
            algo_list[i] = bandit_factory(test_type,name,X, R_true*Noise_Mismatch, S_true*Norm_Mismatch, n,opt_coeff[i-3],emp_coeff[i-3],n_mc_samples)
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
        

t_alpha = 0.5


cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()
print(cum_regret_mean.shape)

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)



n_algo = 6

algo_list = [None]*n_algo
algo_names = ["LinMED","LinMED", "LinMED","LinMED","LinMED", "LinMED"]
n_trials = 10
algo_color = ['y',"r","b","g", "c","gray"]


i=0

for name in algo_names:
    plt.plot(np.arange(n), cum_regret_mean[:,i],color = algo_color[i] , label=algo_names[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i],color = algo_color[i], alpha=.3)
    i = i + 1



now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] + '0005'+ date_time + '.npy'

with open(file_name, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)
    np.save(f,pulled_arm_index)

plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time 0.005")
plt.legend()
plt.savefig('EOPT1.eps',format = 'eps',dpi=300)
plt.savefig('EOPT1.png',format = 'png')
plt.show() 

# ####################################################################This section is temporary one used only when we plot from saved data ########################################################################################### 

""" stored_data_1 = "End_of_opt_Algo_Comp000505212024051459.npy"

with open(stored_data_1, 'rb') as f:

    a = np.load(f)




n = 1000000
K = 3
d = 2
t_alpha = 0.25
#print(a.shape)
#print(c.shape)
cum_regret_arr = a
#algo_names = ["SpannerIGW","OFUL","Lin-SGMED","Lin-IMED-1", "Lin-IMED-3","SpannerIGW-Anytime","EXP2", "Lin-TS-Freq"]
#algo_color = ['y',"r","b","g", "c","m","k", "gray"]
algo_names = ["OFUL","LinMED","Lin-IMED-1", "Lin-IMED-3","Lin-TS-Freq"]
algo_color = ['r',"b","g","c", "gray"]
n_algo = 5

algo_list = [None]*n_algo

n_trials = 10

cum_regret_arr=  np.zeros((n_trials,n,n_algo))

cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()

X = np.arange(n)

sample_size = 1000


print(cum_regret_mean.shape)

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)



i=0
for name in algo_names:
    plt.plot(np.arange(n), cum_regret_mean[:,i],color = algo_color[i] , label=algo_names[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i],color = algo_color[i], alpha=.3)
    i = i + 1



plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time")
plt.legend()
plt.savefig('EOPT3.eps',format = 'eps',dpi=300)
plt.savefig('EOPT3.png',format = 'png')
plt.show() 

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] + '002' +  date_time + '.npy'
with open(file_name, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)  """
 


# ####################################################################This section is temporary one used only when we plot from saved data ########################################################################################### 

#
        
# Naming the x-axis, y-axis and the whole graph


