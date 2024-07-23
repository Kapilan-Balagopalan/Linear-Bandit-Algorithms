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
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d
    sVal_arm_size = K
    sVal_horizon = n
    sVal_arm_set = A = sample_spherical(sVal_arm_size,sVal_dimension)
    theta_true = np.random.randn(d, 1)

    theta_true = S_true*(theta_true/ (np.linalg.norm(theta_true, axis=0)))
    best_arm = np.argmax(np.matmul(A, theta_true))
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_arm_set, theta_true,noise_sigma, delta, S_true, best_arm






K = 20
n = 100
d = 2



n_algo = 7

algo_list = [None]*n_algo
algo_names = ["EXP2","Lin-IMED-3","Lin-TS-Freq","LinMED", "LinZHU","OFUL","Lin-SGMED-NOPT"]
#algo_names = ["OFUL", "Lin-TS-Freq"]
n_trials = 5



emp_coeff = [0.99,0.9,0.5]
opt_coeff = [0.005,0.05,0.25]


n_mc_samples = 0
cum_regret_arr=  np.zeros((n_trials,n,n_algo))

test_type = "Sphere"
Noise_Mismatch = 1
Norm_Mismatch = 1

for j in tqdm(range(n_trials)):
    seed = 15751 + j
    d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm = init(seed, K, n, d)
    R_true = noise_sigma
    i = 0
    for name in algo_names:
        if(i < 7):
            algo_list[i] = bandit_factory(test_type,name,X,R_true*Noise_Mismatch,S_true*Norm_Mismatch ,n,opt_coeff[0],emp_coeff[0],n_mc_samples)
        else :
            algo_list[i] = bandit_factory(test_type,name,X,R_true*Noise_Mismatch,S_true*Norm_Mismatch,n,opt_coeff[i-7],emp_coeff[i-7],n_mc_samples)
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
        

t_alpha = 0.6


cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
#ipdb.set_trace()
print(cum_regret_mean.shape)

cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


name_common = "d="+ str(d) + "K=" + str(K)
now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")


prefix = 'C:/Users/Kapilan/OneDrive - University of Arizona/Academia_Kapilan/Research/Source_code/Lin-SGMED/Lin-SGMED/logs/'

script_name = os.path.basename(__file__)
file_name = os.path.splitext(script_name)[0] +name_common +  date_time + '.npy'

completeName = os.path.join(prefix , file_name)

with open(completeName, 'wb') as f:

    np.save(f, cum_regret_arr)
    np.save(f,algo_names)




#n_algo = 7

#algo_list = [None]*n_algo
#algo_names = ['SpannerIGW', 'OFUL' ,'Lin-IMED-1' ,'Lin-IMED-3',
# 'SpannerIGW-Anytime', 'EXP2' ,'Lin-TS-Freq',r'$\text{LinMED}(\alpha_{\text{emp}} = 0.99)$',
# r'$\text{LinMED}(\alpha_{\text{emp}} = 0.90)$', r'$\text{LinMED}(\alpha_{\text{emp}} = 0.50)$']

algo_color = ['black',"red","lime", "darkgreen","gray","yellow", "saddlebrown", "magenta","darkmagenta","darkviolet","dodgerblue","blue","midnightblue"]

n_trials = 30



i=0
for name in algo_names:
    plt.plot(np.arange(n), cum_regret_mean[:,i] , label=algo_names[i],color = algo_color[i])
    plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i], alpha=.3,color = algo_color[i])
    i = i + 1


# Naming the x-axis, y-axis and the whole graph

name_eps = "IMG" +  "-d="+ str(d) + "-K=" + str(K) + "HT"+".eps"
name_png = "IMG" +  "-d="+ str(d) + "-K=" + str(K)  + "HT" + ".png"

completeName_eps = os.path.join(prefix , name_eps)
completeName_png = os.path.join(prefix , name_png)



plt.xlabel("Time")
plt.ylabel("Regret")
plt.title("Regret with time")
plt.legend()
plt.savefig(completeName_eps,format = 'eps',dpi=300)
plt.savefig(completeName_png, format = 'png')
plt.show() 