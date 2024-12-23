from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

from BanditFactory import *
import ipdb

from datetime import datetime

import os

from tqdm import tqdm

n_algo = 8
algo_list = [None] * n_algo
algo_names = ["OFUL","Lin-TS-Freq","Lin-TS-Bayes","Lin-IMED-1","Lin-IMED-3","LinMED","LinMED","LinMED"]
algo_names_plot = ["OFUL","Lin-TS-Freq","Lin-TS-Bayes","Lin-IMED-1","Lin-IMED-3","LinMED-99","LinMED-90","LinMED-50"]
test_type = "Sphere"
emp_coeff = [0.99, 0.9, 0.5]
opt_coeff = [0.005, 0.05, 0.25]

n_cpu = 10
n_trials = 20

current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/EOPT_0005/'

algo_color = ["r","b","g", "c","m","orange", "gray", "black", "orange"]
eps = 0.005

def init_end_of_optimism(eps):
    #np.random.seed(seed)
    noise_sigma = 10
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 3
    sVal_horizon = n = 1000000
    sVal_arm_set = A = sample_end_of_optimism(eps)
    theta_true = A[0,:]
    best_arm = A[0,:]

    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_arm_set, theta_true,noise_sigma, delta, S_true, best_arm






def thread_process(n_gap,seed_in,X,noise_sigma,theta_true,S_true,n,d,K, delay_switch, reward_delay):
    np.random.seed(seed_in)
    Noise_Mismatch = 1
    Norm_Mismatch = 1

    #return

    cum_regret_arr = np.zeros((n_gap, n, n_algo))
    R_true = noise_sigma


    for j in tqdm(range(n_gap)):
        i = 0
        for name in algo_names:
            if (i >= 5 ):
                algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n,d,
                                          opt_coeff[i-5], emp_coeff[i-5],1000, delay_switch ,reward_delay)
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
                algo_list[i].update_delayed(X[arm,:], reward)




    script_name = os.path.basename(__file__)
    file_name = str(seed_in) + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:

        np.save(f, cum_regret_arr)


def t():
    np.random.seed(15751)
    d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm = init_end_of_optimism(eps)

    n_gap = int(n_trials/n_cpu)
    pool = Pool(processes=n_cpu)

    delay_switch = False
    reward_delay = 20


    cum_regret_arr = np.zeros((n_trials, n, n_algo))

    for i in range(n_cpu):
        pool.apply_async(thread_process, args=(n_gap,i,X,noise_sigma,theta_true, S_true,n,d,K, delay_switch,reward_delay))

    pool.close()
    pool.join()



    script_name = os.path.basename(__file__)
    for i in range(n_cpu):
        start_ind = int(i * n_gap)
        end_ind = int((i + 1) * n_gap)
        file_name = str(i) + '.npy'

        completeName = os.path.join(prefix, file_name)

        with open(completeName, 'rb') as f:
            cum_regret_arr[start_ind:end_ind,:,:] = np.load(f)



    t_alpha = 0.1


    cum_regret_mean = np.sum(cum_regret_arr, axis=0)/n_trials
    cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
    #ipdb.set_trace()
    print(cum_regret_mean.shape)

    cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)
    cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std)/np.sqrt(n_trials)


    name_common = "d="+ str(d) + "K=" + str(K)
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y%H%M%S")





    script_name = os.path.basename(__file__)
    file_name = os.path.splitext(script_name)[0] +name_common +  date_time + '.npy'

    completeName = os.path.join(prefix , file_name)

    with open(completeName, 'wb') as f:

        np.save(f, cum_regret_arr)
        np.save(f,algo_names)


    i=0
    for name in algo_names_plot:
        X = np.arange(n)
        Y = cum_regret_mean[:,i]
        P = cum_regret_confidence_up[:,i]
        Q = cum_confidence_down[:,i]
        plt.plot(X[::1000], Y[::1000], label=algo_names_plot[i],color = algo_color[i],linewidth= 3)
        plt.fill_between(X[::1000],P[::1000],Q[::1000] , alpha=.4,color = algo_color[i])
        i = i + 1


    # Naming the x-axis, y-axis and the whole graph

    #name_eps = "EOPT_002_W"+".eps"
    #name_png = "EOPT_002_W" + ".png"
    name_pdf = "EOPT_0005" + ".pdf"

    #completeName_eps = os.path.join(prefix , name_eps)
    #completeName_png = os.path.join(prefix , name_png)
    completeName_pdf = os.path.join(prefix, name_pdf)



    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    #plt.title("Regret with time")
    plt.ticklabel_format(style='sci', axis='y',scilimits=(3, 3))
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    #plt.savefig(completeName_eps,format = 'eps',dpi=300)
    #plt.savefig(completeName_png, format = 'png')
    plt.savefig(completeName_pdf, format='pdf')
    plt.show()


if __name__ == '__main__':
    t()
