import numpy as np
from MAB_BenoulliTS import *
from MAB_KL_MS import *
from multiprocessing import Pool, cpu_count
from datetime import datetime
import sys
import os
import random


from tqdm import tqdm
mu_1_true = 0.80
mu_2_true = 0.90
n = 10000
n_trials = 2000
n_mc_samples = 1000
prob_min_thresh = 1/n_trials * 0.5
K = 2
n_algo = 2

def thread_process(n_gap,seed_in):
    #print(seed_in)
    #random.seed(seed_in)
    algo_list = [None] * n_algo

    cum_regret_arr = np.zeros((n_gap, n, n_algo))
    mu_hat = np.zeros((n_gap, n_algo, 1))
    offline_logged_data = np.zeros((n_gap, n, n_algo, 3))


    X = np.zeros((K, 1))

    X[0][0] = mu_1_true
    X[1][0] = mu_2_true



    for j in tqdm(range(n_gap)):
        algo_list[0] = MAB_KL_MS(K, X, n_mc_samples)
        algo_list[1] = MAB_TS_Bernoulli(K, X, n_mc_samples)
        for i in range(n_algo):
            cum_mu_hat = 0
            cum_regret = 0
            for t in range(n):
                arm = algo_list[i].next_arm()

                offline_logged_data[j][t][i][0] = arm
                prob_chosen = algo_list[i].get_probability_arm()
                offline_logged_data[j][t][i][1] = prob_chosen
                reward = np.random.binomial(1, X[arm][0], size=None)
                cum_regret = cum_regret + np.max(X) - X[arm][0]
                cum_regret_arr[j][t][i] = cum_regret
                offline_logged_data[j][t][i][2] = reward
                if (i==0):
                    cum_mu_hat = cum_mu_hat + reward / prob_chosen
                else:
                    cum_mu_hat = cum_mu_hat + reward / np.maximum(prob_min_thresh, prob_chosen)
                # print(arm)
                # print(prob_chosen)
                algo_list[i].update(arm, reward)

            mu_hat[j][i][0] = cum_mu_hat / (K * n)

    current_dir = os.path.dirname(__file__)
    prefix = current_dir + '/logs/'

    script_name = os.path.basename(__file__)
    file_name = str(seed_in) + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:

        np.save(f, mu_hat)



def t():

    X = np.zeros((K, 1))
    X[0][0] = mu_1_true
    X[1][0] = mu_2_true

    n_cpu = 10
    n_gap =int( n_trials/n_cpu)
    pool = Pool(processes=n_cpu)

    mu_hat = np.zeros((n_trials, n_algo, 1))

    #random.seed(15751)

    for i in range(n_cpu):
        pool.apply_async(thread_process, args=(n_gap,i))

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
            mu_hat[start_ind:end_ind,:,:] = np.load(f)

    uniform_average_regret = np.sum(X) / K

    #print(mu_hat[:, 0, 0])
    ms_h = np.max(mu_hat[:, 0, 0])
    ms_l = np.min(mu_hat[:, 0, 0])

    ts_h = np.max(mu_hat[:, 1, 0])
    ts_l = np.min(mu_hat[:, 1, 0])

    bin_ms = int((ms_h - ms_l)/0.05)
    bin_ts = int((ts_h - ts_l) / 0.05)

    plt.hist(mu_hat[:, 0, 0], bins=100, color='skyblue', alpha=0.5, edgecolor='black', label='KL-MS')
    plt.axvline(mu_hat[:, 0, 0].mean(), color='skyblue', linestyle='dashed', linewidth=1)
    plt.hist(mu_hat[:, 1, 0], bins=100, color='green', alpha=0.5, edgecolor='black', label='BernoulliTS')
    plt.axvline(mu_hat[:, 1, 0].mean(), color='green', linestyle='dashed', linewidth=1)
    plt.axvline(x=uniform_average_regret, color='black', label='axvline - full height')

    # print(mu_hat[:, 0, 0])
    print(mu_hat[:, 0, 0].mean())
    print(mu_hat[:, 1, 0].mean())
    # Adding labels and title
    plt.xlabel('Expected Reward')
    plt.ylabel('Trials')
    plt.title('Histogram of Expected rewards vs trials')
    plt.legend()
    plt.savefig('Img-2.png', format='png')
    plt.show()

    # t_alpha = 0.1
    #
    # cum_regret_mean = np.sum(cum_regret_arr, axis=0) / n_trials
    # cum_regret_mean_std = np.std(cum_regret_arr, axis=0, ddof=1)
    # # ipdb.set_trace()
    #
    # cum_regret_confidence_up = cum_regret_mean + (t_alpha * cum_regret_mean_std) / np.sqrt(n_trials)
    # cum_confidence_down = cum_regret_mean - (t_alpha * cum_regret_mean_std) / np.sqrt(n_trials)
    #
    # i = 0
    # algo_names = ["KL-MS", "BernoulliTS"]
    # algo_color = ['y', "r", "b", "g", "c", "m", "k", "gray"]
    #
    # for name in algo_names:
    #     plt.plot(np.arange(n), cum_regret_mean[:, i], color=algo_color[i], label=algo_names[i])
    #     plt.fill_between(np.arange(n), cum_regret_confidence_up[:, i], cum_confidence_down[:, i], color=algo_color[i],
    #                      alpha=.3)
    #     i = i + 1
    #
    # # Naming the x-axis, y-axis and the whole graph
    # plt.xlabel("Time")
    # plt.ylabel("Regret")
    # plt.title("Regret with time gamma= 1")
    # plt.legend()
    # plt.savefig('PDE11.eps', format='eps')
    # plt.savefig('PDE11.png', format='png')
    # plt.show()


if __name__ == '__main__':
    t()
