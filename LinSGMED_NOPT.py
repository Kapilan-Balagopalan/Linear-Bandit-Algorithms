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
    noise_sigma = 3
    delta = 0.01
    S_true = 1
    sVal_dimension = d
    sVal_arm_size = K
    sVal_horizon = n
    sVal_lambda = d
    mVal_I = np.eye(sVal_dimension)
    mVal_lvrg_scr_orgn = sVal_lambda*mVal_I
    sVal_arm_set = A = worse_case_scenario_experiment(K)
    theta_true = theta_true = A[0,:]
    #print(theta_true.shape)
    #print(A.shape)
    #theta_true = S*(theta_true/ (np.linalg.norm(theta_true, axis=0)))
    best_arm = A[0,:]
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, sVal_arm_set, theta_true,\
           noise_sigma, delta, S_true, best_arm






#K = 100
n = 10000
d = 2

size_k_set = 5

K_set = [8,16,32,64,128]

n_algo = 6

algo_list = [None]*n_algo
algo_names = ["LinMED","LinMED","LinMED","LinMED", "LinMED","Lin-SGMED-NOPT"]
n_trials = 10


cum_regret_arr=  np.zeros((n_trials,n,n_algo))

test_type = "Sphere"
emp_coeff = [0.99,0,0.99,0.9,0.5]
opt_coeff = [0,0,0.005,0.05,0.25]
c_gamma = 0.5
cum_final_point_regret = np.zeros((n_algo,size_k_set))
k_set_counter = 0
Noise_Mismatch = 1
Norm_Mismatch = 1


for K_in in K_set:
    for j in tqdm(range(n_trials)):
    #seed = np.random.randint(1, 15751)
        seed = 15751 + j
        d, K, n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, X, theta_true, noise_sigma, delta, S_true, best_arm = init(seed, K_in , n,
                                                                                                                d)
        R_true = noise_sigma
        i = 0
        for name in algo_names:
            if(i < 5):
                algo_list[i] = bandit_factory(test_type,name,X,R_true*Noise_Mismatch,S_true*Norm_Mismatch,n,opt_coeff[i],emp_coeff[i])
            else:
                algo_list[i] = bandit_factory(test_type,name,X,R_true*Noise_Mismatch,S_true*Norm_Mismatch,n,0,0)
            i = i+1
        i=0
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
            
            cum_final_point_regret[i][k_set_counter] = cum_final_point_regret[i][k_set_counter] + cum_regret
    k_set_counter = k_set_counter + 1    



t_alpha = 1


now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'
with open(file_name, 'wb') as f:

    np.save(f, cum_final_point_regret)

i=0
algo_names = ["LinMED-9900","LinMED-0000","LinMED-99","LinMED-90", "LinMED-50","Lin-SGMED-NOPT"]
algo_color = ['y',"r","b","g", "c","m","k", "gray"]
for name in algo_list:
    plt.plot(np.log(K_set), cum_final_point_regret[i,:]/n_trials , color = algo_color[i],label=algo_names[i])
    #plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i], alpha=.3)
    i = i + 1

# Naming the x-axis, y-axis and the whole graph
plt.grid()
plt.xlabel("Number of Arms")
plt.ylabel("Cumulate regret")
plt.title("Regret with number of arms c_gamma = 1")
plt.legend()
plt.savefig('gerfge1.png',format = 'png')
plt.show()
 
# ####################################################################This section is temporary one used only when we plot from saved data ########################################################################################### 

#

""" stored_data_1 = "LinSGMED_NOPT05192024180339.npy"
stored_data_2 = "LinSGMED_NOPT05192024224733.npy"
stored_data_3 = "LinSGMED_NOPT05192024232559.npy"

with open(stored_data_1, 'rb') as f:

    a = np.load(f)

with open(stored_data_2, 'rb') as g:

    b = np.load(g)

with open(stored_data_3, 'rb') as h:

    c = np.load(h)
#cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
#cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


i=0
K_set = [16,32,64]
n_trials = 10
n_algo = 2

n = 100000
d = 2
size_k_set = 7

K_set = [16,32,64,128,256,512,1024]

n_algo = 2

algo_list = [None]*n_algo
algo_names = ["Lin-SGMED-1","Lin-SGMED-NOPT"]
#algo_names = ["OFUL", "Lin-TS-Freq"]
n_trials = 10

cum_regret_arr=  np.zeros((n_trials,n,n_algo))
cum_final_point_regret = np.zeros((n_algo,size_k_set))

cum_final_point_regret[:,0:3] = a
cum_final_point_regret[:,3:6] = b
cum_final_point_regret[:,6] = c[:,0]

#size_k_set = 7

#K_set = [16,32,64,128,256,512,1024]

n_algo = 2

algo_list = [None]*n_algo
algo_names = ["Lin-SGMED","Lin-SGMED-NOPT"]

for name in algo_list:
    plt.plot(np.log(K_set), np.log((cum_final_point_regret[i,:]/n_trials)) , label=algo_names[i])
    #plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i], alpha=.3)
    i = i + 1



# Naming the x-axis, y-axis and the whole graph
plt.grid()
plt.xlabel("Number of Arms (log (K))")
plt.ylabel("Cumulative regret(log ($Reg_n$))")
plt.title("Regret with $K$")
plt.legend()
plt.savefig('LAS1.eps',format = 'eps',dpi=300)
plt.savefig('LAS1.png',format = 'png')
plt.show() 

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] +  date_time + '.npy'
with open(file_name, 'wb') as f:

    np.save(f, cum_final_point_regret) """


# ####################################################################This section is temporary one used only when we plot from saved data ########################################################################################### 

#