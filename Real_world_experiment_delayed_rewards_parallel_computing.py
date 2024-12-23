from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

from BanditFactory import *
import ipdb

from datetime import datetime

import os 

from tqdm import tqdm

n_algo = 5
algo_list = [None] * n_algo
algo_names = ["OFUL","Lin-TS-Bayes","LinMED","LinMED","LinMED"]
algo_names_plot = ["OFUL","Lin-TS-Bayes","LinMED-99","LinMED-90","LinMED-50"]
test_type = "Sphere"
emp_coeff = [0.99, 0.9, 0.5]
opt_coeff = [0.005, 0.05, 0.25]

n_cpu = 10
n_trials = 100

algo_color = ["r","b","g", "c","m","k", "gray", "black", "orange"]

current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/Delayed_Rewards_Experiment_20_Delay/'

def init_real_world_exp():
    #np.random.seed(seed)
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 4
    sVal_arm_size = K = 10
    sVal_horizon = n = 5000
    arm_set, contexts, theta_true  = generate_real_world_armset(d,n_users_aug = 10, n_movies_aug = 10)


    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, theta_true,\
           noise_sigma, delta, S_true, contexts,arm_set



def thread_process(n_gap,seed_in, contexts ,arm_set, noise_sigma, theta_true, S_true,n,d, delay_switch, reward_delay):
    print(seed_in)
    np.random.seed(seed_in)
    #d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm = init_real_world_exp()

    #return

    Noise_Mismatch = 1
    Norm_Mismatch = 1
    R_true = noise_sigma

    cum_regret_arr = np.zeros((n_gap, n, n_algo))

    #return


    n_contexts = contexts.shape[0]
    # print(n_contexts)

    # print(X.shape)
    X = np.zeros((arm_set.shape[0], d))
    #current_context = contexts[np.random.choice(n_contexts, size=1, replace=False), :]
    #X = np.zeros((arm_set.shape[0],d))
    for j in tqdm(range(n_gap)):
        n_contexts = contexts.shape[0]



        i = 0
        for name in algo_names:
            if (i >= 2):
                algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n,d,
                                          opt_coeff[i-2], emp_coeff[i-2],1000, delay_switch ,reward_delay)
            else:
                algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                              opt_coeff[0], emp_coeff[0], 1000, delay_switch, reward_delay)
            i = i + 1

        i = 0
        cum_regret = 0
        for i in range(n_algo):
            cum_regret = 0
            for t in range(n):
                current_context = contexts[np.random.choice(n_contexts, size=1, replace=False), :]
                norm_list = np.zeros(arm_set.shape[0])
                for k in range(arm_set.shape[0]):
                    X[k, :] = np.outer(arm_set[k, :], current_context).ravel()
                    norm_list[k] = np.linalg.norm(X[k, :])

                X = X / np.min(norm_list)

                arm = algo_list[i].next_arm(X)
                inst_regret = calc_regret(arm, theta_true, X)
                cum_regret = cum_regret + inst_regret
                cum_regret_arr[j][t][i] = cum_regret
                reward = receive_reward(arm, theta_true, noise_sigma, X)
                algo_list[i].update_delayed(X[arm,:], reward)
                #print(X)




    file_name = str(seed_in) + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:

        np.save(f, cum_regret_arr)



def t():
    np.random.seed(15751)
    d, K, n, theta_true, noise_sigma, delta, S_true,contexts, arm_set = init_real_world_exp()

    n_gap = int(n_trials/n_cpu)
    pool = Pool(processes=n_cpu)

    delay_switch = True
    reward_delay = 20

    cum_regret_arr = np.zeros((n_trials, n, n_algo))

    for i in range(n_cpu):
        pool.apply_async(thread_process, args=(n_gap,i, contexts ,arm_set ,noise_sigma, theta_true, S_true,n,d, delay_switch, reward_delay))

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

    #uniform_average_regret = (np.matmul(X[0, :], theta_true) + np.matmul(X[1, :], theta_true)) / 2

    name_common = "d=" + str(d) + "K=" + str(K)
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y%H%M%S")



    script_name = os.path.basename(__file__)
    file_name = os.path.splitext(script_name)[0] + name_common + date_time + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:
        np.save(f, cum_regret_arr)

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

    completeName = os.path.join(prefix , file_name)

    with open(completeName, 'wb') as f:

        np.save(f, cum_regret_arr)

    t_alpha = 1


    i=0

    for name in algo_names_plot:
        plt.plot(np.arange(n), cum_regret_mean[:,i],color = algo_color[i], label=algo_names_plot[i],lw= 3)
        plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i],cum_confidence_down[:,i],color = algo_color[i], alpha=.3,lw= 3)
        i = i + 1

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Regret", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)



    plt.legend(fontsize = 15)
    #plt.legend(handles=legend_handles, fontsize=15)
    #plt.savefig(prefix + 'PDE11.eps', format='eps')
    #plt.savefig(prefix + 'PDE11.png', format='png')
    plt.savefig(prefix + 'DRE_20.pdf', format='pdf')
    plt.show()
    i=0
    for name in algo_names_plot:
        plt.plot(np.arange(n), np.log(cum_regret_mean[:,i]) ,color = algo_color[i], label=algo_names_plot[i],lw= 3)
        plt.fill_between(np.arange(n),np.log(cum_regret_confidence_up[:,i]),np.log( cum_confidence_down[:,i]),color = algo_color[i], alpha=.3,lw= 3)
        i = i + 1

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("log-Regret", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)



    plt.legend(fontsize = 15)
    #plt.legend(handles=legend_handles, fontsize=15)
    #plt.savefig(prefix + 'PDE11.eps', format='eps')
    #plt.savefig(prefix + 'PDE11.png', format='png')
    plt.savefig(prefix + 'DRE_LOG_20.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    t()
