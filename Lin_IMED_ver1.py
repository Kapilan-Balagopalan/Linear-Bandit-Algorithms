from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt
import ipdb 

import numpy.random as ra
import numpy.linalg as la
from Bandit_Env import *

#all the scalar values will start with sVal
#all the vector values will start with vVal
#all the matrix values will start with mVal
#underscore will be used to divide words
#meaning full names will not have vowels in it e.g leverage = 'lvrg'
class Lin_IMED(Bandit):
    ########################################
    def __init__(self, X, lam, R, S, flags):
        self.X = X
        self.R = R
        self.lam = lam
        self.delta = .01
        self.S = S
        self.flags = flags
    


        # more instance variables
        self.t = 1
        self.K, self.d = self.X.shape


        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.lam
       
        self.sqrt_beta = calc_sqrt_beta_det2(self.d, self.t, self.R, self.lam, self.delta, self.S, self.logdetV)
        self.theta_hat = np.zeros(self.d)
        self.Vt = self.lam * np.eye(self.d)
        self.beta_t = 1

       

        self.MED_quo = np.ones(self.K)
        self.empirical_best_quo = 0.5
        self.opt_design_quo = 0.5
        self.AugX = self.X.copy()
        self.empirical_best_ind = np.zeros(self.K)
        self.Delta_empirical_gap = np.ones(self.K)
        self.empirical_best_arm = 0
        self.gamma_t = 0

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.K), self.do_not_ask)
        if (self.t == 1):
            chosen = np.argmax(self.MED_quo)
            return chosen
        radius_sq = self.multiplier * (self.beta_t)
        if (self.subsample_func == None):
            chosen = np.argmin(self.MED_quo)
        else:
            raise NotImplementedError()  # todo: use valid_idx
        #             idx = self.subsample_func(self.N, self.subN);
        #             subX = self.X[idx,:];
        #             obj_func = np.dot(subX, self.theta_hat) + np.sqrt(radius_sq) * \
        #                     np.sqrt( mahalanobis_norm_sq_batch(subX, self.invVt));
        #
        #             chosen = idx[np.argmax(obj_func)];
        return chosen

    def estimate_empirical_reward_gap(self):
        # print(theta_t.shape)
        # print(A.shape)
        reward_A = np.matmul(self.X, self.theta_hat)
        self.Delta_empirical_gap = np.max(reward_A) - reward_A


    def calc_IMED_ver1_index(self):
        #a = self.X[self.empirical_best_arm, :]
        #vVal_lev_score_emp_best = np.matmul(np.matmul(a.T, self.invVt), a)
        # print(vVal_lev_score_emp_best)
        # print(a.shape)
        for i in range(self.K):
            a = self.X[i, :]
            vVal_lev_score_a = np.matmul(np.matmul(a.T, self.invVt), a)
            #print(vVal_lev_score_a)
            self.MED_quo[i] = ((self.Delta_empirical_gap[i]**2)/((self.beta_t)*vVal_lev_score_a)) - np.log((self.beta_t)*vVal_lev_score_a)

    def update(self, pulled_idx, y_t):
        #assert (y_t >= 0.0 and y_t <= 1.0);
        ##########################
        ## DEBUGGING
        ##########################
        # y_t = 2*y_t - 1;
        xt = self.X[pulled_idx, :]
        #print(y_t)
        #print(xt)
        self.XTy = self.XTy +  y_t * xt
        self.Vt =  self.Vt + np.outer(xt, xt)

        self.invVt = np.linalg.pinv(self.Vt )

        self.theta_hat = np.matmul(self.invVt, self.XTy.T)
        self.do_not_ask.append(pulled_idx)
        my_t = self.t + 1
        #self.sqrt_beta = calc_sqrt_beta_det2(self.d, my_t, self.R, self.lam, self.delta, self.S, self.logdetV)
        self.beta_t = calc_beta_t(self.t, self.d, self.lam, self.delta, self.S, self.R)
        self.estimate_empirical_reward_gap()

        #self.empirical_best_arm = np.where(self.Delta_empirical_gap == 0)[0][0]
        #self.empirical_best_ind[self.empirical_best_arm] = 1

        self.calc_IMED_ver1_index()
        #self.scale_arms()
        #self.gamma_t = calc_gamma_t(self.t,self.d,self.lam,self.delta,self.S,self.R)
        self.t = self.t +  1
