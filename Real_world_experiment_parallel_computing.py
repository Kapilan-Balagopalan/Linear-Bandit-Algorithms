from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

from BanditFactory import *
import ipdb

from datetime import datetime

import os 

from tqdm import tqdm


def init_real_world_exp():
    #np.random.seed(seed)
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 20
    sVal_arm_size = K = 20
    sVal_horizon = n = 10000
    arm_set, contexts, theta_true  = generate_real_world_armset(n_users_aug = 20, n_movies_aug = 20)


    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, theta_true,\
           noise_sigma, delta, S_true, contexts,arm_set



def thread_process(n_gap,seed_in, contexts ,arm_set, noise_sigma, theta_true, S_true,n,d):
    print(seed_in)
    #d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm = init_real_world_exp()

    n_algo = 5
    #n_trials = 1000


    algo_list = [None] * n_algo

    algo_names = ["EXP2", "Lin-TS-Freq", "LinMED", "LinZHU-AT", "OFUL"]

    Noise_Mismatch = 1
    Norm_Mismatch = 1
    R_true = noise_sigma

    cum_regret_arr = np.zeros((n_gap, n, n_algo))


    test_type = "Sphere"

    emp_coeff = [0.99, 0.9, 0.5]
    opt_coeff = [0.005, 0.1, 0.5]

    n_contexts = contexts.shape[0]
    # print(n_contexts)

    # print(X.shape)
    X = np.zeros((arm_set.shape[0], 20))
    #current_context = contexts[np.random.choice(n_contexts, size=1, replace=False), :]
    #X = np.zeros((arm_set.shape[0],d))
    for j in tqdm(range(n_gap)):
        n_contexts = contexts.shape[0]
        # print(n_contexts)
        #X = np.zeros((arm_set.shape[0], 20))

        current_context = contexts[np.random.choice(n_contexts, size=1, replace=False),:]
        for k in range(arm_set.shape[0]):
            X[k, :] = np.outer(arm_set[k, :], current_context).ravel()



        # Initialize PCA and reduce to 2 components
        # n_components_movies = 5

        X_final = X

        best_arm = np.argmax(np.matmul(X_final, theta_true))
        i = 0
        for name in algo_names:
            algo_list[i] = bandit_factory(test_type, name, X_final, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n,
                                          opt_coeff[0], emp_coeff[0],1000)
            i = i + 1

        i = 0
        cum_regret = 0
        for i in range(n_algo):
            cum_regret = 0
            for t in range(n):
                arm = algo_list[i].next_arm()
                inst_regret = calc_regret(arm, theta_true, X_final)
                cum_regret = cum_regret + inst_regret
                cum_regret_arr[j][t][i] = cum_regret
                reward = receive_reward(arm, theta_true, noise_sigma, X_final)
                algo_list[i].update_delayed(arm, reward)



    current_dir = os.path.dirname(__file__)
    prefix = current_dir + '/logs/'

    script_name = os.path.basename(__file__)
    file_name = str(seed_in) + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:

        np.save(f, cum_regret_arr)



def t():
    d, K, n, theta_true, noise_sigma, delta, S_true,contexts, arm_set = init_real_world_exp()
    n_algo = 5

    algo_names = ["EXP2", "Lin-TS-Freq", "LinMED", "LinZHU", "OFUL"]

    n_cpu = 10
    n_trials = 1000
    n_gap = int(n_trials/n_cpu)
    pool = Pool(processes=n_cpu)


    cum_regret_arr = np.zeros((n_trials, n, n_algo))

    for i in range(n_cpu):
        pool.apply_async(thread_process, args=(n_gap,i, contexts ,arm_set ,noise_sigma, theta_true, S_true,n,d))

    pool.close()
    pool.join()

    current_dir = os.path.dirname(__file__)
    prefix = current_dir + '/logs/'

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

    current_dir = os.path.dirname(__file__)
    prefix = current_dir + '/logs/'

    script_name = os.path.basename(__file__)
    file_name = os.path.splitext(script_name)[0] + name_common + date_time + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:
        np.save(f, cum_regret_arr)
        np.save(f, algo_names)

    t_alpha = 3


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

    current_dir = os.path.dirname(__file__)
    prefix = current_dir + '/logs/'
    completeName = os.path.join(prefix , file_name)

    with open(completeName, 'wb') as f:

        np.save(f, cum_regret_arr)
        np.save(f,algo_names)

    t_alpha = 1


    i=0
    algo_names = ["EXP2", "Lin-TS-Freq", "LinMED", "LinZHU", "OFUL"]
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


if __name__ == '__main__':
    t()
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

