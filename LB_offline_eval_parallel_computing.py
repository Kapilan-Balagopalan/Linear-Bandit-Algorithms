from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import numpy.random as ra
import numpy.linalg as la
from scipy.stats import gaussian_kde
from BanditFactory import *

import ipdb

from datetime import datetime

import os

from tqdm import tqdm

current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/OLE_5000/'

def init_offline_eval_exp():
    #np.random.seed(seed)
    noise_sigma = 0.1
    delta = 0.01
    S_true = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 2
    sVal_horizon = n = 1000
    n_trials = 5000
    sVal_arm_set = A = sample_offline_eval_experiment(K)
    theta_true = np.zeros((d,1))
    theta_true[0][0] = 1
    best_arm = A[K-1,:]

    return sVal_dimension, sVal_arm_size,sVal_horizon , sVal_arm_set, theta_true,noise_sigma, delta, S_true, best_arm,n_trials



def thread_process(n_gap,seed_in):
    print(seed_in)
    #return
    d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm,n_trials = init_offline_eval_exp()
    np.random.seed(seed_in)
    n_algo = 2
    algo_list = [None] * n_algo

    algo_names = ["LinMED", "Lin-TS-Freq"]
    #return
    Noise_Mismatch = 1
    Norm_Mismatch = 1
    R_true = noise_sigma

    #return
    test_type = "Sphere"

    emp_coeff = [0.99, 0.9, 0.5]
    opt_coeff = [0.005, 0.1, 0.5]

    n_mc_samples = 100000
    prob_min_thresh = (1/n_mc_samples)*0.5

    mu_hat = np.zeros((n_gap, n_algo, 1))
    count = 0
    frq_count = np.zeros((2,n_algo, K))
    #return
    for j in tqdm(range(n_gap)):
        i = 0
        #if(j <= n_gap):
            #continue
        #continue
        for name in algo_names:
            algo_list[i] = bandit_factory(test_type, name, X, R_true * Noise_Mismatch, S_true * Norm_Mismatch, n, d,
                                          opt_coeff[2], emp_coeff[2], n_mc_samples, False, 5)
            i = i + 1

        for i in range(n_algo):
            cum_mu_hat = 0
            for t in range(n):

                arm = algo_list[i].next_arm(X)
                frq_count[0][i][arm] = frq_count[0][i][arm] + 1
                prob_chosen = algo_list[i].get_probability_arm(X)

                reward = receive_reward(arm, theta_true, noise_sigma, X)
                if (i==1):
                    if(prob_chosen < prob_min_thresh):
                        count = count + 1
                    frq_count[1][i][arm] = frq_count[1][i][arm] + 1 / np.maximum(prob_min_thresh, prob_chosen)
                    cum_mu_hat = cum_mu_hat + reward / np.maximum(prob_min_thresh, prob_chosen)
                else:
                    frq_count[1][i][arm] = frq_count[1][i][arm] + 1 / prob_chosen
                    cum_mu_hat = cum_mu_hat + reward / prob_chosen

                algo_list[i].update_delayed(X[arm,:], reward)

            mu_hat[j][i][0] = cum_mu_hat / (K * n)


    script_name = os.path.basename(__file__)
    file_name = str(seed_in) + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:

        np.save(f, mu_hat)

    frq_count = frq_count/(n_gap*n)
    print("LinTS: count ", frq_count[0][1][:] )
    print("LinTS: prob ", frq_count[1][1][:])
    print("LinMED count: ", frq_count[0][0][:])
    print("LinMED prob: ", frq_count[1][0][:])
    print(count/n_gap)

def t():
    d, K, n, X, theta_true, noise_sigma, delta, S_true, best_arm,n_trials = init_offline_eval_exp()
    algo_names = ["LinMED-50", "Lin-TS-Freq"]

    n_cpu = 10
    n_gap = int(n_trials/n_cpu)
    pool = Pool(processes=n_cpu)
    n_algo = 2
    mu_hat = np.zeros((n_trials, n_algo, 1))

    for i in range(n_cpu):
        pool.apply_async(thread_process, args=( n_gap,i))

    pool.close()
    pool.join()


    script_name = os.path.basename(__file__)
    for i in range(n_cpu):
        start_ind = int(i * n_gap)
        end_ind = int((i + 1) * n_gap)
        file_name = str(i) + '.npy'

        completeName = os.path.join(prefix, file_name)

        with open(completeName, 'rb') as f:
            mu_hat[start_ind:end_ind,:,:] = np.load(f)

    uniform_average_regret = np.matmul(np.mean(X,axis = 0), theta_true)

    name_common = "d=" + str(d) + "K=" + str(K)
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y%H%M%S")



    script_name = os.path.basename(__file__)
    file_name = os.path.splitext(script_name)[0] + name_common + date_time + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:
        np.save(f, mu_hat)
        np.save(f, algo_names)

    #print(mu_hat)

    LinMED_estimated_vals = mu_hat[:, 0, 0]
    LinMED_estimated_vals = LinMED_estimated_vals[LinMED_estimated_vals < 2]
    LinMED_estimated_vals = LinMED_estimated_vals[LinMED_estimated_vals > -2]

    LinTS_estimated_vals = mu_hat[:, 1, 0]
    LinTS_estimated_vals = LinTS_estimated_vals[LinTS_estimated_vals < 2]
    LinTS_estimated_vals = LinTS_estimated_vals[LinTS_estimated_vals > -2]
    #print(LinTS_estimated_vals)
    kde = gaussian_kde(LinMED_estimated_vals)
    x_vals = np.linspace(min(LinMED_estimated_vals), max(LinMED_estimated_vals), 1000)
    y_vals = kde(x_vals)
    plt.plot(x_vals, y_vals, color='skyblue', lw=3)

    kde = gaussian_kde(LinTS_estimated_vals)
    x_vals = np.linspace(min(LinTS_estimated_vals), max(LinTS_estimated_vals), 1000)
    y_vals = kde(x_vals)
    plt.plot(x_vals, y_vals, color='green', lw=3)

    plt.hist(LinMED_estimated_vals, density=True, bins=100, color='skyblue', alpha=0.4, edgecolor='skyblue', label='LinMED-50')
    plt.axvline(mu_hat[:, 0, 0].mean(), color='skyblue', linestyle='dashed', linewidth=3,label = 'Mean IPW- LinMED-50')
    plt.hist(LinTS_estimated_vals, density=True, bins=100, color='green', alpha=0.4, edgecolor='green', label='Lin-TS-Freq')
    plt.axvline(mu_hat[:, 1, 0].mean(), color='green', linestyle='dashed', linewidth=3,label = 'Mean IPW - Lin-TS-Freq')
    plt.axvline(x=uniform_average_regret, color='black',label = 'Oracle',linewidth=3)

   # print(mu_hat[:, 0, 0])
    print("The mean of LinMED : ",mu_hat[:, 0, 0].mean())
    print("The mean of LinTS : " , mu_hat[:, 1, 0].mean())
    print("The variance of LinMED : ", mu_hat[:, 0, 0].var())
    print("The variance of LinTS : ", mu_hat[:, 1, 0].var())
    print("Online Uniform results : ", uniform_average_regret)
    # Adding labels and title
    plt.xlabel('IPW scores',fontsize = 15)
    plt.ylabel('Trials',fontsize = 15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title('Histogram of IPW scores vs trials')
    plt.legend(fontsize = 15)
    plt.savefig(prefix + 'OLE.eps', format='eps')
    plt.savefig(prefix + 'OLE.png', format='png')
    plt.savefig(prefix + 'OLE.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    t()






