from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt


def init(seed,K,n,d):
    np.random.seed(seed)
    noise_sigma = 1
    delta = 0.01
    S = 1
    sVal_dimension = d
    sVal_arm_size = K
    sVal_horizon = n
    sVal_lambda = d
    mVal_I = np.eye(sVal_dimension)
    mVal_lvrg_scr_orgn = sVal_lambda*mVal_I
    sVal_arm_set = A = sample_random(sVal_arm_size,sVal_dimension)
    theta_true = np.random.randn(d, 1)
    #print(theta_true.shape)
    #print(A.shape)
    theta_true = S*(theta_true/ (np.linalg.norm(theta_true, axis=0)))
    best_arm = np.argmax(np.matmul(A, theta_true))
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, sVal_arm_set, theta_true,\
           noise_sigma, delta, S, best_arm

def init_end_of_optimism(eps):
    #np.random.seed(seed)
    noise_sigma = 0.1
    delta = 0.01
    S = 1
    sVal_dimension = d = 2
    sVal_arm_size = K = 3
    sVal_horizon = n = 2000
    sVal_lambda = d
    mVal_I = np.eye(sVal_dimension)
    mVal_lvrg_scr_orgn = sVal_lambda*mVal_I
    sVal_arm_set = A = sample_end_of_optimism(eps)
    theta_true = A[0,:]
    #print(theta_true.shape)
    #print(A.shape)
    #theta_true = S*(theta_true/ (np.linalg.norm(theta_true, axis=0)))
    best_arm = A[0,:]
    # print(best_arm)
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, sVal_arm_set, theta_true,\
           noise_sigma, delta, S, best_arm

def calc_q_opt_design(A):
    sVal_opt_design_arms, sampling_time = optimal_design_algo(A)
    prob_dist = optimal_probability(A, sVal_opt_design_arms)
    return prob_dist

def sample_action(A,MED_prob_dist):
    K,d = A.shape
    ind = np.random.choice(K, 1, p= MED_prob_dist)
    return A[ind,:]

def receive_reward(Arm_t,theta_true, noise_sigma):
    #print(Arm_t.shape)
    #print(theta_true.shape)
    noise = np.random.normal(0,noise_sigma, 1)
    #print(noise)
    reward = np.dot(Arm_t,theta_true)
    #print(reward)
    final_reward = reward + noise
    #print(final_reward)
    return final_reward

def estimate_empirical_reward_gap(theta_t, A):
    #print(theta_t.shape)
    #print(A.shape)
    reward_A = np.matmul(A,theta_t)
    Delta_A = np.max(reward_A) - reward_A
    return Delta_A


def calc_gamma_t(t,d,sVal_lambda,delta,S,noise_sigma):
    gamma_t = (noise_sigma*np.sqrt(d*np.log(1 + t/(sVal_lambda*d)) + 2*np.log(2/delta)) + np.sqrt(sVal_lambda)*S)**2
    return gamma_t


def calc_MED_ver1_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t):
    K,d = A.shape
    MED_quo = np.zeros(K)
    a = A[empirical_best_arm, :]
    vVal_lev_score_emp_best = np.matmul(np.matmul(a.T, mVal_lvrg_scr_orgn_inv_t), a)
    #print(vVal_lev_score_emp_best)
    #print(a.shape)
    for i in range(K):
        a = A[i,:]
        vVal_lev_score_a = np.matmul(np.matmul(a.T,mVal_lvrg_scr_orgn_inv_t),a)
        MED_quo[i] = np.exp(-(Delta_empirical_gap[i])**2/(gamma_t*(vVal_lev_score_a + vVal_lev_score_emp_best)))
    #print(MED_quo)
    return MED_quo

def calc_MED_ver2_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t):
    K,d = A.shape
    MED_quo = np.zeros(K)
    a_best = A[empirical_best_arm, :]
    #vVal_lev_score_emp_best = np.matmul(np.matmul(a.T, mVal_lvrg_scr_orgn_inv_t), a)
    #print(vVal_lev_score_emp_best)
    #print(a.shape)
    for i in range(K):
        if i != empirical_best_arm :
            a = A[i, :]
            vVal_lev_score_a_and_best = np.matmul(np.matmul((a - a_best).T, mVal_lvrg_scr_orgn_inv_t), (a - a_best))
            MED_quo[i] = np.exp(-(Delta_empirical_gap[i]) ** 2 / (gamma_t * (vVal_lev_score_a_and_best)))
        else:
            MED_quo[i] = 1
    #print(MED_quo)
    return MED_quo

def scale_arms(A,MED_quo):
    K, d = A.shape
    aug_A = A
    for i in range(K):
        aug_A[i,:] = np.sqrt(MED_quo[i]) * A[i,:]
    return aug_A
def calc_expected_regret(best_arm,theta_true, MED_prob_dist,A):
    mVal_reward= np.matmul(A, theta_true)
    mVal_exp_reward = np.dot(MED_prob_dist,mVal_reward)
    inst_regret = np.max(np.matmul(A, theta_true)) - np.sum(mVal_exp_reward)
    #print( np.argmax(np.matmul(A, theta_true)))
    #print("This is expected reward",np.sum(mVal_exp_reward))
    #print(inst_regret)
    return inst_regret


def Lin_SGMED_algo_ver1_EOPT_main(seed):
    d, K ,n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, A , theta_true,noise_sigma,delta,S,best_arm = init_end_of_optimism(seed)
    #print(theta_true)
    aug_A = A
    MED_quo = np.ones(K)
    empirical_best_ind = np.zeros(K)
    empirical_best_quo = 0
    opt_design_quo = 1
    mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn
    vVal_acc_reward_arm = 0
    acc_regret = 0
    acc_regret_arr = np.zeros(n)
    for t in range(n):
        prob_dist = calc_q_opt_design(aug_A)
        #print("Optimal design probability distribution", prob_dist)
        emp_bst_opt_prob_dist = empirical_best_quo*empirical_best_ind + opt_design_quo*prob_dist
        MED_prob_dist = np.multiply(emp_bst_opt_prob_dist, MED_quo)
        MED_prob_dist = MED_prob_dist/np.sum(MED_prob_dist)
        #print(MED_prob_dist)
        #print("Final probability distribution", MED_prob_dist)
        #print(np.sum(MED_prob_dist))
        #print(MED_prob_dist.shape)
        Arm_t = sample_action (A,MED_prob_dist)

        inst_regret = calc_expected_regret(best_arm,theta_true, MED_prob_dist, A)
        acc_regret = acc_regret + inst_regret
        acc_regret_arr[t] = acc_regret
        # print(Arm_t)
        reward_t = receive_reward(Arm_t,theta_true, noise_sigma)
        # print(reward_t)

        mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn_t + np.outer(Arm_t, Arm_t)
        #print(mVal_lvrg_scr_orgn)
        #print(mVal_lvrg_scr_orgn_t)
        mVal_lvrg_scr_orgn_inv_t = np.linalg.inv(mVal_lvrg_scr_orgn_t)
        #print(mVal_lvrg_scr_orgn_inv_t.shape)
        vVal_acc_reward_arm = vVal_acc_reward_arm + Arm_t*reward_t
        #print(vVal_acc_reward_arm.shape)
        theta_t = np.matmul(mVal_lvrg_scr_orgn_inv_t, vVal_acc_reward_arm.T)
        #print(theta_t)
        Delta_empirical_gap = estimate_empirical_reward_gap(theta_t, A)
        #print(Delta_empirical_gap)
        #print(Delta_empirical_gap.shape)
        #print(np.min(Delta_empirical_gap))
        if(t==1):
            empirical_best_quo = 0.5
            opt_design_quo = 0.5
        empirical_best_arm = np.where(Delta_empirical_gap == 0)[0][0]
        #print(empirical_best_arm)
        empirical_best_ind = np.zeros(K)
        empirical_best_ind[empirical_best_arm] = 1
        gamma_t = calc_gamma_t(t,d,sVal_lambda,delta,S,noise_sigma)
        #print(gamma_t)
        MED_quo = calc_MED_ver1_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t)
        #print(MED_quo)
        aug_A = scale_arms(A,MED_quo)

    print(acc_regret)
    print("Best reward is : " ,np.max(np.dot(A, theta_true)) )
    #plt.plot(np.arange(n), acc_regret_arr , label="Lin-SGMED")
    # Naming the x-axis, y-axis and the whole graph
    #plt.xlabel("Time")
    #plt.ylabel("Regret")
    #plt.title("Expected regret with time")
    #plt.legend()
    #plt.show()
    return n, acc_regret_arr

def Lin_SGMED_algo_ver2_EOPT_main(seed):
    d, K ,n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, A , theta_true,noise_sigma,delta,S,best_arm = init_end_of_optimism(seed)
    #print(theta_true)
    aug_A = A
    MED_quo = np.ones(K)
    empirical_best_ind = np.zeros(K)
    empirical_best_quo = 0
    opt_design_quo = 1
    mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn
    vVal_acc_reward_arm = 0
    acc_regret = 0
    acc_regret_arr = np.zeros(n)
    for t in range(n):
        prob_dist = calc_q_opt_design(aug_A)
        #print("Optimal design probability distribution", prob_dist)
        emp_bst_opt_prob_dist = empirical_best_quo*empirical_best_ind + opt_design_quo*prob_dist
        MED_prob_dist = np.multiply(emp_bst_opt_prob_dist, MED_quo)
        MED_prob_dist = MED_prob_dist/np.sum(MED_prob_dist)
        #print(MED_prob_dist)
        #print("Final probability distribution", MED_prob_dist)
        #print(np.sum(MED_prob_dist))
        #print(MED_prob_dist.shape)
        Arm_t = sample_action (A,MED_prob_dist)

        inst_regret = calc_expected_regret(best_arm,theta_true, MED_prob_dist, A)
        acc_regret = acc_regret + inst_regret
        acc_regret_arr[t] = acc_regret
        # print(Arm_t)
        reward_t = receive_reward(Arm_t,theta_true, noise_sigma)
        # print(reward_t)

        mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn_t + np.outer(Arm_t, Arm_t)
        #print(mVal_lvrg_scr_orgn)
        #print(mVal_lvrg_scr_orgn_t)
        mVal_lvrg_scr_orgn_inv_t = np.linalg.inv(mVal_lvrg_scr_orgn_t)
        #print(mVal_lvrg_scr_orgn_inv_t.shape)
        vVal_acc_reward_arm = vVal_acc_reward_arm + Arm_t*reward_t
        #print(vVal_acc_reward_arm.shape)
        theta_t = np.matmul(mVal_lvrg_scr_orgn_inv_t, vVal_acc_reward_arm.T)
        #print(theta_t)
        Delta_empirical_gap = estimate_empirical_reward_gap(theta_t, A)
        #print(Delta_empirical_gap)
        #print(Delta_empirical_gap.shape)
        #print(np.min(Delta_empirical_gap))
        if(t==1):
            empirical_best_quo = 0.5
            opt_design_quo = 0.5
        empirical_best_arm = np.where(Delta_empirical_gap == 0)[0][0]
        #print(empirical_best_arm)
        empirical_best_ind = np.zeros(K)
        empirical_best_ind[empirical_best_arm] = 1
        gamma_t = calc_gamma_t(t,d,sVal_lambda,delta,S,noise_sigma)
        #print(gamma_t)
        MED_quo = calc_MED_ver2_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t)
        #print(MED_quo)
        aug_A = scale_arms(A,MED_quo)

    print(acc_regret)
    print("Best reward is : " ,np.max(np.dot(A, theta_true)) )
    #plt.plot(np.arange(n), acc_regret_arr , label="Lin-SGMED")
    # Naming the x-axis, y-axis and the whole graph
    #plt.xlabel("Time")
    #plt.ylabel("Regret")
    #plt.title("Expected regret with time")
    #plt.legend()
    #plt.show()

    return n, acc_regret_arr


def Lin_SGMED_algo_ver1_main(seed,K,n,d,):
    d, K ,n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, A , theta_true,noise_sigma,delta,S,best_arm = init(seed,K,n,d)
    #print(theta_true)
    aug_A = A
    MED_quo = np.ones(K)
    empirical_best_ind = np.zeros(K)
    empirical_best_quo = 0
    opt_design_quo = 1
    mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn
    vVal_acc_reward_arm = 0
    acc_regret = 0
    acc_regret_arr = np.zeros(n)
    for t in range(n):
        prob_dist = calc_q_opt_design(aug_A)
        #print("Optimal design probability distribution", prob_dist)
        emp_bst_opt_prob_dist = empirical_best_quo*empirical_best_ind + opt_design_quo*prob_dist
        MED_prob_dist = np.multiply(emp_bst_opt_prob_dist, MED_quo)
        MED_prob_dist = MED_prob_dist/np.sum(MED_prob_dist)
        #print(MED_prob_dist)
        #print("Final probability distribution", MED_prob_dist)
        #print(np.sum(MED_prob_dist))
        #print(MED_prob_dist.shape)
        Arm_t = sample_action (A,MED_prob_dist)

        inst_regret = calc_expected_regret(best_arm,theta_true, MED_prob_dist, A)
        acc_regret = acc_regret + inst_regret
        acc_regret_arr[t] = acc_regret
        # print(Arm_t)
        reward_t = receive_reward(Arm_t,theta_true, noise_sigma)
        # print(reward_t)

        mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn_t + np.outer(Arm_t, Arm_t)
        #print(mVal_lvrg_scr_orgn)
        #print(mVal_lvrg_scr_orgn_t)
        mVal_lvrg_scr_orgn_inv_t = np.linalg.inv(mVal_lvrg_scr_orgn_t)
        #print(mVal_lvrg_scr_orgn_inv_t.shape)
        vVal_acc_reward_arm = vVal_acc_reward_arm + Arm_t*reward_t
        #print(vVal_acc_reward_arm.shape)
        theta_t = np.matmul(mVal_lvrg_scr_orgn_inv_t, vVal_acc_reward_arm.T)
        #print(theta_t)
        Delta_empirical_gap = estimate_empirical_reward_gap(theta_t, A)
        #print(Delta_empirical_gap)
        #print(Delta_empirical_gap.shape)
        #print(np.min(Delta_empirical_gap))
        if(t==1):
            empirical_best_quo = 0.5
            opt_design_quo = 0.5
        empirical_best_arm = np.where(Delta_empirical_gap == 0)[0][0]
        #print(empirical_best_arm)
        empirical_best_ind = np.zeros(K)
        empirical_best_ind[empirical_best_arm] = 1
        gamma_t = calc_gamma_t(t,d,sVal_lambda,delta,S,noise_sigma)
        #print(gamma_t)
        MED_quo = calc_MED_ver1_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t)
        #print(MED_quo)
        aug_A = scale_arms(A,MED_quo)

    print(acc_regret)
    print("Best reward is : " ,np.max(np.dot(A, theta_true)) )
    #plt.plot(np.arange(n), acc_regret_arr , label="Lin-SGMED")
    # Naming the x-axis, y-axis and the whole graph
    #plt.xlabel("Time")
    #plt.ylabel("Regret")
    #plt.title("Expected regret with time")
    #plt.legend()
    #plt.show()
    return n, acc_regret_arr

def Lin_SGMED_algo_ver2_main(seed,K,n,d):
    d, K ,n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, A , theta_true,noise_sigma,delta,S,best_arm = init(seed,K,n,d)
    #print(theta_true)
    aug_A = A
    MED_quo = np.ones(K)
    empirical_best_ind = np.zeros(K)
    empirical_best_quo = 0
    opt_design_quo = 1
    mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn
    vVal_acc_reward_arm = 0
    acc_regret = 0
    acc_regret_arr = np.zeros(n)
    for t in range(n):
        prob_dist = calc_q_opt_design(aug_A)
        #print("Optimal design probability distribution", prob_dist)
        emp_bst_opt_prob_dist = empirical_best_quo*empirical_best_ind + opt_design_quo*prob_dist
        MED_prob_dist = np.multiply(emp_bst_opt_prob_dist, MED_quo)
        MED_prob_dist = MED_prob_dist/np.sum(MED_prob_dist)
        #print(MED_prob_dist)
        #print("Final probability distribution", MED_prob_dist)
        #print(np.sum(MED_prob_dist))
        #print(MED_prob_dist.shape)
        Arm_t = sample_action (A,MED_prob_dist)

        inst_regret = calc_expected_regret(best_arm,theta_true, MED_prob_dist, A)
        acc_regret = acc_regret + inst_regret
        acc_regret_arr[t] = acc_regret
        # print(Arm_t)
        reward_t = receive_reward(Arm_t,theta_true, noise_sigma)
        # print(reward_t)

        mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn_t + np.outer(Arm_t, Arm_t)
        #print(mVal_lvrg_scr_orgn)
        #print(mVal_lvrg_scr_orgn_t)
        mVal_lvrg_scr_orgn_inv_t = np.linalg.inv(mVal_lvrg_scr_orgn_t)
        #print(mVal_lvrg_scr_orgn_inv_t.shape)
        vVal_acc_reward_arm = vVal_acc_reward_arm + Arm_t*reward_t
        #print(vVal_acc_reward_arm.shape)
        theta_t = np.matmul(mVal_lvrg_scr_orgn_inv_t, vVal_acc_reward_arm.T)
        #print(theta_t)
        Delta_empirical_gap = estimate_empirical_reward_gap(theta_t, A)
        #print(Delta_empirical_gap)
        #print(Delta_empirical_gap.shape)
        #print(np.min(Delta_empirical_gap))
        if(t==1):
            empirical_best_quo = 0.5
            opt_design_quo = 0.5
        empirical_best_arm = np.where(Delta_empirical_gap == 0)[0][0]
        #print(empirical_best_arm)
        empirical_best_ind = np.zeros(K)
        empirical_best_ind[empirical_best_arm] = 1
        gamma_t = calc_gamma_t(t,d,sVal_lambda,delta,S,noise_sigma)
        #print(gamma_t)
        MED_quo = calc_MED_ver2_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t)
        #print(MED_quo)
        aug_A = scale_arms(A,MED_quo)

    print(acc_regret)
    print("Best reward is : " ,np.max(np.dot(A, theta_true)) )
    #plt.plot(np.arange(n), acc_regret_arr , label="Lin-SGMED")
    # Naming the x-axis, y-axis and the whole graph
    #plt.xlabel("Time")
    #plt.ylabel("Regret")
    #plt.title("Expected regret with time")
    #plt.legend()
    #plt.show()

    return n, acc_regret_arr

def plotter_end_of_optimism(eps):
    acc_regret_arr_ver1_avg = 0
    acc_regret_arr_ver2_avg = 0
    sVal_trials = 50
    for i in range(sVal_trials):
        #seed = i * 189
        # np.random.seed(seed)
        n, acc_regret_arr_ver1 = Lin_SGMED_algo_ver1_EOPT_main(eps)
        acc_regret_arr_ver1_avg = acc_regret_arr_ver1_avg + acc_regret_arr_ver1
        n, acc_regret_arr_ver2 = Lin_SGMED_algo_ver2_EOPT_main(eps)
        acc_regret_arr_ver2_avg = acc_regret_arr_ver2_avg + acc_regret_arr_ver2

    plt.plot(np.arange(n), acc_regret_arr_ver1_avg / sVal_trials, label="Lin-SGMED-ver1")
    plt.plot(np.arange(n), acc_regret_arr_ver2_avg / sVal_trials, label="Lin-SGMED-ver2")
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.title("Expected regret with time")
    plt.legend()
    plt.show()


def plotter_arm_size(K,n,d):
    acc_regret_arr_ver1_avg = 0
    acc_regret_arr_ver2_avg = 0
    sVal_trials = 20
    for i in range(sVal_trials):
        seed = i * 189
        # np.random.seed(seed)
        n, acc_regret_arr_ver1 = Lin_SGMED_algo_ver1_main(seed,K,n,d)
        acc_regret_arr_ver1_avg = acc_regret_arr_ver1_avg + acc_regret_arr_ver1
        n, acc_regret_arr_ver2 = Lin_SGMED_algo_ver2_main(seed,K,n,d)
        acc_regret_arr_ver2_avg = acc_regret_arr_ver2_avg + acc_regret_arr_ver2

    plt.plot(np.arange(n), acc_regret_arr_ver1_avg / sVal_trials, label="Lin-SGMED-ver1")
    plt.plot(np.arange(n), acc_regret_arr_ver2_avg / sVal_trials, label="Lin-SGMED-ver2")
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.title("Expected regret with time")
    plt.legend()
    plt.show()

############################################End of Optimism Experiment##################################################

plotter_end_of_optimism(eps = 0.005)
plotter_end_of_optimism(eps = 0.01)
plotter_end_of_optimism(eps = 0.1)


############################################Varying arm size##################################################
#plotter_arm_size(K=10,n=10000,d=20)
#plotter_arm_size(K=100,n=3000,d=20)
#plotter_arm_size(K=1000,n=1000,d=20)


############################################End of Optimism Experiment##################################################
#plotter_arm_size(K=10,n=10000,d=20)
#plotter_arm_size(K=10,n=3000,d=50)
#plotter_arm_size(K=10,n=1000,d=100)