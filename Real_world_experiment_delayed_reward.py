from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

from BanditFactory import *
import ipdb

from datetime import datetime

import os 

from tqdm import tqdm

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def init_real_world_exp():
    #np.random.seed(seed)
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 4
    sVal_arm_size = K = 10
    sVal_horizon = n = 1000
    arm_set, contexts, theta_true = generate_real_world_armset(d,n_users_aug = 10, n_movies_aug = 10)
    #best_arm = np.argmax(np.matmul(A,theta_true))
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, theta_true,\
           noise_sigma, delta, S_true, arm_set,contexts


d, K, n, theta_true, noise_sigma, delta, S_true, arm_set, contexts = init_real_world_exp()

n_algo = 9

algo_list = [None]*n_algo
algo_names =algo_names = ["OFUL","Lin-TS-Freq","LinZHU","LinZHU-AT","Lin-IMED-1","Lin-IMED-3","LinMED","LinMED","LinMED"]
n_trials = 1000
Noise_Mismatch = 1
Norm_Mismatch = 1
n_mc_samples = 0

cum_regret_arr=  np.zeros((n_trials,n,n_algo))

test_type = "Sphere"

emp_coeff = [0.99,0.9,0.5]
opt_coeff = [0.005,0.05,0.25]


X = np.zeros((arm_set.shape[0], d))
for j in tqdm(range(n_trials)):
    seed = 15751
    np.random.seed(seed)
    R_true= noise_sigma
    i = 0
    n_contexts = contexts.shape[0]
    #print(n_contexts)

    #print(X.shape)


    for name in algo_names:

        if (i >= 7):
            algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                          opt_coeff[i - 7], emp_coeff[i - 7], 1000, False, 0)
        else:
            algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                          opt_coeff[0], emp_coeff[0], 1000, False, 0)
        i = i+1

    cum_regret = 0


    for i in range(n_algo):
        cum_regret = 0
        for t in range(n):
            current_context = contexts[np.random.choice(n_contexts, size=1, replace=False), :]

            for k in range(arm_set.shape[0]):
                # print(arm_set.shape[0])
                X[k, :] = np.outer(arm_set[k, :], current_context).ravel()

            for l in range(d):
                temp_min = np.min(X[:, l])
                temp_max = np.max(X[:, l])
                if (temp_max - temp_min == 0):
                    continue
                else:
                    X[:, l] = (X[:, l] - temp_min) / (temp_max - temp_min)

            #print(X)

            arm = algo_list[i].next_arm(X)
            inst_regret = calc_regret(arm, theta_true, X)
            cum_regret = cum_regret + inst_regret
            cum_regret_arr[j][t][i] =  cum_regret
            reward = receive_reward(arm, theta_true, noise_sigma, X)
            algo_list[i].update_delayed(X[arm,:],reward)


t_alpha = 1


cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()
#print(cum_regret_mean.shape)

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)

file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'

current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/'
completeName = os.path.join(prefix , file_name)

with open(completeName, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)
    np.save(f,emp_coeff)

t_alpha = 1


cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


i=0

algo_names = ["OFUL","LinMED"]
algo_color = ['y',"r","b","g", "c","m","k", "gray"]
for name in algo_names:
    plt.plot(np.arange(n), cum_regret_mean[:,i] ,color = algo_color[i], label=algo_names[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i],color = algo_color[i], alpha=.3)
    i = i + 1

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time gamma= 1")
plt.legend()
plt.savefig(prefix + 'PDE11.eps',format = 'eps')
plt.savefig(prefix + 'PDE11.png',format = 'png')
plt.show()


i=0
for name in algo_names:
    if( algo_names[i]== "SpannerIGW" or  algo_names[i]== "Lin-TS-Freq" or algo_names[i]== "SpannerIGW-Anytime"):
        i = i+1
        continue
    plt.plot(np.arange(n), cum_regret_mean[:,i] , label=algo_names[i], color = algo_color[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i], color = algo_color[i],alpha=.3)
    i = i + 1

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time gamma= 1")
plt.legend()
plt.savefig(prefix + 'PDE21.eps',format = 'eps')
plt.savefig(prefix + 'PDE21.png',format = 'png')
plt.show()
# ####################################################################This section is temporary one used only when we plot from saved data ###########################################################################################

#
        


# n = 1000
#
# t_alpha = 0.5
#
# n_algo = 2
#
# algo_names = ["Lin-TS-Freq","LinMED"]
#
#
# algo_list = [None]*n_algo
#
# n_trials = 1000
#
# cum_regret_arr=  np.zeros((n_trials,n,n_algo))
#
# stored_data = "Real_world_experiment09182024080111.npy"
#
# with open(stored_data, 'rb') as f:
#
#     a = np.load(f)
#     b = np.load(f)
#
#
#
# cum_regret_arr = a
#
#
# cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
# cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
# #ipdb.set_trace()
# print(cum_regret_mean.shape)
#
# cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
# cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
#
#
# i=0
# algo_color = ['y',"r","b","g", "c","m","k", "gray"]
# for name in algo_names:
#     plt.plot(np.arange(n), cum_regret_mean[:,i] ,color = algo_color[i], label=algo_names[i])
#     plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i],color = algo_color[i], alpha=.3)
#     i = i + 1
#
# # Naming the x-axis, y-axis and the whole graph
# plt.xlabel("Time")
# plt.ylabel("Regret")
# plt.title("Regret with time")
# plt.legend()
# plt.savefig('PDE1.eps',format = 'eps',dpi=300)
# plt.savefig('PDE1.png',format = 'png')
# plt.show()
#
#
# now = datetime.now() # current date and time
# date_time = now.strftime("%m%d%Y%H%M%S")
#
# script_name = os.path.basename(__file__)
# file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'
# with open(file_name, 'wb') as f:
#
#     np.save(f, cum_regret_arr)
#     np.save(f,algo_names)

