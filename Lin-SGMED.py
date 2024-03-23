from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

#all the scalar values will start with sVal
#all the vector values will start with vVal
#all the matrix values will start with mVal
#underscore will be used to divide words
#meaning full names will not have vowels in it e.g leverage = 'lvrg'

def init():
    noise_sigma = 1
    delta = 0.01
    S = 1
    sVal_dimension = d = 10
    sVal_arm_size = K = 1000
    sVal_horizon = n = 1
    sVal_lambda = 1
    mVal_I = np.eye(sVal_dimension)
    mVal_lvrg_scr_orgn = sVal_lambda*mVal_I
    sVal_arm_set = A = sample_random(sVal_arm_size,sVal_dimension)
    theta_true = np.random.randn(d, 1)
    theta_true = S*(theta_true/ (np.linalg.norm(theta_true, axis=0)))
    return sVal_dimension, sVal_arm_size,sVal_horizon, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, sVal_arm_set, theta_true, noise_sigma, delta, S

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


def calc_gamma_t(t,d,sVal_lambda,delta,S):
    gamma_t = (np.sqrt(d*np.log(1 + t/(sVal_lambda*d)) + 2*np.log(2/delta)) + np.sqrt(sVal_lambda)*S)**2
    return gamma_t

def calc_MED_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t):
    return 0

def Lin_SGMED_algo_main():
    d, K ,n, sVal_lambda, mVal_I, mVal_lvrg_scr_orgn, A , theta_true,noise_sigma,delta,S= init()
    #print(theta_true)
    aug_A = A
    MED_quo = np.ones(K)
    empirical_best_ind = np.zeros(K)
    empirical_best_quo = 0
    opt_design_quo = 1
    mVal_lvrg_scr_orgn_t = mVal_lvrg_scr_orgn
    vVal_acc_reward_arm = 0
    for t in range(n):
        prob_dist = calc_q_opt_design(aug_A)
        #print("Optimal design probability distribution", prob_dist)
        emp_bst_opt_prob_dist = empirical_best_quo*empirical_best_ind + opt_design_quo*prob_dist
        MED_prob_dist = np.multiply(emp_bst_opt_prob_dist, MED_quo)
        #print("Final probability distribution", MED_prob_dist)
        #print(np.sum(MED_prob_dist))
        #print(MED_prob_dist.shape)
        Arm_t = sample_action (A,MED_prob_dist)
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
        gamma_t = calc_gamma_t(t,d,sVal_lambda,delta,S)
        print(gamma_t)
        MED_quo = calc_MED_probability_distribution(Delta_empirical_gap,mVal_lvrg_scr_orgn_inv_t, A, empirical_best_arm, gamma_t)
        print(MED_quo)







Lin_SGMED_algo_main()


