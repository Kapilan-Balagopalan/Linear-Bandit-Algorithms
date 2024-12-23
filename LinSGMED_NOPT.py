from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy.random as ra
import numpy.linalg as la


from BanditFactory import *

import ipdb

from datetime import datetime

import os 


from tqdm import tqdm
 
 


def init(K,n,d):
    noise_sigma = 1
    delta = 0.01
    S_true = 1
    sVal_dimension = d
    sVal_arm_size = K
    sVal_horizon = n
    sVal_arm_set = A = worse_case_scenario_experiment(K)
    theta_true = theta_true = A[0,:]

    best_arm = A[0,:]
    # print(best_arm)
    return sVal_arm_set, theta_true,noise_sigma, delta, S_true, best_arm






#K = 100
n = 20000
d = 2

size_k_set = 5

K_set = [4,8,16,32,64]

n_algo = 4

algo_list = [None]*n_algo
algo_names = ["Lin-SGMED-NOPT","LinMED","LinMED","LinMED"]

n_cpu = 10
n_trials = 10

n_mc_samples = 90
test_type = "Sphere"
emp_coeff = [0.99,0.9,0.5]
opt_coeff = [0.005,0.05,0.25]

algo_names_plot = ["LinMEDNOPT", "LinMED(" + r'$\alpha_{emp}$' + " = 0.99)", "LinMED(" + r'$\alpha_{emp}$' + "= 0.90)",
                  "LinMED(" + r'$\alpha_{emp}$' + " = 0.50)"]
algo_color = ['y', "r", "b", "g", "c", "m", "k", "gray"]


def thread_process(n_gap,seed_in):
    np.random.seed(seed_in)
    Noise_Mismatch = 3
    Norm_Mismatch = 1
    return
    cum_final_point_regret = np.zeros((n_algo, size_k_set))
    k_set_counter = 0
    for K_in in K_set:
        X, theta_true, noise_sigma, delta, S_true, best_arm = init(K_in, n, d)
        for j in tqdm(range(n_gap)):
            R_true = noise_sigma
            i = 0

            for name in algo_names:
                if (i < 1):
                    algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                                  opt_coeff[i], emp_coeff[i], n_mc_samples, False, 5)

                else:
                    algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                                  opt_coeff[i - 1], emp_coeff[i - 1], n_mc_samples, False, 5)
                i = i + 1
                #print("comes here")
            for i in range(n_algo):
                cum_regret = 0
                for t in range(n):
                    arm = algo_list[i].next_arm(X)
                    inst_regret = calc_regret(arm, theta_true, X)
                    cum_regret = cum_regret + inst_regret
                    #cum_regret_arr[j][t][i] = cum_regret
                    reward = receive_reward(arm, theta_true, noise_sigma, X)
                    algo_list[i].update_delayed(X[arm, :], reward)

                cum_final_point_regret[i][k_set_counter] = cum_final_point_regret[i][k_set_counter] + cum_regret
        k_set_counter = k_set_counter + 1
        
    current_dir = os.path.dirname(__file__)
    prefix = current_dir + '/logs/'

    script_name = os.path.basename(__file__)
    file_name = str(seed_in) + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:

        np.save(f, cum_final_point_regret)








def t():
    np.random.seed(15751)
    n_gap = int(n_trials/n_cpu)
    pool = Pool(processes=n_cpu)


    cum_final_point_regret = np.zeros((n_algo, size_k_set))

    for i in range(n_cpu):
        pool.apply_async(thread_process, args=(n_gap,i))

    pool.close()
    pool.join()

    current_dir = os.path.dirname(__file__)
    prefix = current_dir + '/logs/Final_Paper_K_Dependency/'

    script_name = os.path.basename(__file__)
    for i in range(n_cpu):
        start_ind = int(i * n_gap)
        end_ind = int((i + 1) * n_gap)
        file_name = str(i) + '.npy'

        completeName = os.path.join(prefix, file_name)

        with open(completeName, 'rb') as f:
            cum_final_point_regret[:,:] = cum_final_point_regret[:,:]  + np.load(f)

    t_alpha = 1

    i = 0
    for name in algo_list:
        plt.plot(np.log(K_set), np.log(cum_final_point_regret[i, :] / n_trials), color=algo_color[i],
                 label=algo_names_plot[i],linewidth=3)
        # plt.fill_between(np.arange(n),cum_regret_confidence_up[:,i], cum_confidence_down[:,i], alpha=.3)
        i = i + 1

    # Naming the x-axis, y-axis and the whole graph
    plt.grid()
    plt.xlabel("Number of Arms",fontsize=15)
    plt.ylabel("Cumulate regret" + r'$\log(Reg_n)$',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title("Regret with number of arms")
    plt.legend(fontsize=15)
    plt.savefig(prefix + 'KDEP1.png', format='png')
    #plt.set_rasterized(True)
    plt.savefig(prefix + 'KDEP1.eps', format='eps', dpi=300)
    plt.savefig(prefix + 'KDEP1.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    t()