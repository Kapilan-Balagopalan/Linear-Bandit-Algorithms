from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

from BanditFactory import *
import ipdb

from datetime import datetime

import os 

from tqdm import tqdm


def init_prob_dep_exp():
    #np.random.seed(seed)
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 2
    sVal_horizon = n = 10000
    sVal_arm_set = A = sample_problem_dependent_experiment()
    theta_true = A[0,:]

    best_arm = A[0,:]
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_arm_set, theta_true,\
           noise_sigma, delta, S_true, best_arm


d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm = init_prob_dep_exp()

n_algo = 3

algo_list = [None]*n_algo
algo_names = ["LinMED","LinMED", "LinMED"]
n_trials = 10
Noise_Mismatch = 1
Norm_Mismatch = 1
n_mc_samples = 0

cum_regret_arr=  np.zeros((n_trials,n,n_algo))

test_type = "Sphere"

emp_coeff = [0.99,0.9,0.5]
opt_coeff = [0.005,0.05,0.25]




for j in tqdm(range(n_trials)):
    seed = 15751 + j
    np.random.seed(seed)
    R_true= noise_sigma
    i = 0
    for name in algo_names:

        algo_list[i] = bandit_factory(test_type,name,X,R_true*Norm_Mismatch,S_true*Noise_Mismatch , n,opt_coeff[i],emp_coeff[i],n_mc_samples)
        
        i = i+1

    cum_regret = 0
    for i in range(n_algo):
        cum_regret = 0
        for t in range(n):
            arm = algo_list[i].next_arm()
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
    np.save(f,emp_coeff) 

t_alpha = 1


cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


i=0

algo_names = ["LinMED-99","LinMED-90", "LinMED-50"]
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
plt.savefig('PDE11.eps',format = 'eps')
plt.savefig('PDE11.png',format = 'png')
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
plt.savefig('PDE21.eps',format = 'eps')
plt.savefig('PDE21.png',format = 'png')
plt.show() 
# ####################################################################This section is temporary one used only when we plot from saved data ########################################################################################### 

#
        


""" n = 1000000
K = 2
d = 2
t_alpha = 0.5


algo_names = ["SpannerIGW","OFUL","Lin-SGMED","Lin-IMED-1", "Lin-IMED-3","SpannerIGW-Anytime","EXP2", "Lin-TS-Freq"]
algo_color = ['y',"r","b","g", "c","m","k", "gray"]
n_algo = 8

algo_list = [None]*n_algo

n_trials = 10

cum_regret_arr=  np.zeros((n_trials,n,n_algo))

stored_data = "Problem_Dep_Exp05182024132226.npy"

with open(stored_data, 'rb') as f:

    a = np.load(f)
    b = np.load(f)




cum_regret_arr= a

print(b)
stored_data_2 = "Problem_Dep_Exp05182024143729.npy"

with open(stored_data_2, 'rb') as g:

    cum_regret_arr_2 = np.load(g)
    print(np.load(g))

print(cum_regret_arr_2.shape)
print(cum_regret_arr.shape)


stored_data_3 = "Problem_Dep_Exp05182024175558.npy"

with open(stored_data_3, 'rb') as h:

    cum_regret_arr_3 = np.load(h)
    print(np.load(h))

stored_data_4 = "Problem_Dep_Exp05202024213058.npy"

with open(stored_data_4, 'rb') as l:

    cum_regret_arr_4 = np.load(l)
    print(np.load(l))





for i in range(n_trials):
    for j in range(n):
        cum_regret_arr[i][j][2] = cum_regret_arr_3[i][j][0]
        cum_regret_arr[i][j][6] = cum_regret_arr_2[i][j][1]
        cum_regret_arr[i][j][4] = cum_regret_arr_4[i][j][0]

cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()
print(cum_regret_mean.shape)

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


i=0
algo_names = ["SpannerIGW","OFUL","Lin-SGMED","Lin-IMED-1", "Lin-IMED-3","SpannerIGW-Anytime","EXP2", "Lin-TS-Freq"]
algo_color = ['y',"r","b","g", "c","m","k", "gray"]
for name in algo_names:
    plt.plot(np.arange(n), cum_regret_mean[:,i] ,color = algo_color[i], label=algo_names[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i],color = algo_color[i], alpha=.3)
    i = i + 1

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time")
plt.legend()
plt.savefig('PDE1.eps',format = 'eps',dpi=300)
plt.savefig('PDE1.png',format = 'png')
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
plt.title("Regret with time")
plt.legend()
plt.savefig('PDE2.eps',format = 'eps',dpi=300)
plt.savefig('PDE2.png',format = 'png')
plt.show() 

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'
with open(file_name, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)  """


# ####################################################################This section is temporary one used only when we plot from saved data ########################################################################################### 

#
        
# Naming the x-axis, y-axis and the whole graph