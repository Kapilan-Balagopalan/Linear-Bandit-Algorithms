from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt
import ipdb

import numpy.random as ra
import numpy.linalg as la
from Bandit_Env import *

def sample_action(A,MED_prob_dist):
    K,d = A.shape
    ind = np.random.choice(K, 1, p= MED_prob_dist)
    return A[ind,:], ind

def calc_q_opt_design(A):
    sVal_opt_design_arms, sampling_time = optimal_design_algo(A)
    prob_dist = optimal_probability(A, sVal_opt_design_arms)
    return prob_dist


#all the scalar values will start with sVal
#all the vector values will start with vVal
#all the matrix values will start with mVal
#underscore will be used to divide words
#meaning full names will not have vowels in it e.g leverage = 'lvrg'
class Lin_SGMED(Bandit):
    ########################################
    def __init__(self, X, lam, R, S, flags, subsample_func=None, subsample_rate=1.0, multiplier=1.0):
        self.X = X
        self.R = R
        self.lam = lam
        self.delta = .01
        self.S = S
        self.flags = flags
        self.multiplier = float(multiplier)


        # more instance variables
        self.t = 1
        self.K, self.d = self.X.shape

        # - subsampling aspect
        assert subsample_func == None
        self.subN = np.round(self.K * float(subsample_rate)).astype(int)
        self.subsample_func = subsample_func

        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.lam
        self.X_invVt_norm_sq = np.sum(self.X * self.X, axis=1) / self.lam
        self.logdetV = self.d * np.log(self.lam)
        self.sqrt_beta = calc_sqrt_beta_det2(self.d, self.t, self.R, self.lam, self.delta, self.S, self.logdetV)
        self.theta_hat = np.zeros(self.d)
        self.Vt = self.lam * np.eye(self.d)

        self.do_not_ask = []
        self.dbg_dict = {'multiplier': float(multiplier),
                         'subN': self.subN,
                         'subsample_func': self.subsample_func}

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
            prob_dist = calc_q_opt_design(self.AugX)

            emp_bst_opt_prob_dist =  prob_dist
            MED_prob_dist = np.multiply(emp_bst_opt_prob_dist, self.MED_quo)
            MED_prob_dist = MED_prob_dist / np.sum(MED_prob_dist)
            # print(MED_prob_dist)
            # print("Final probability distribution", MED_prob_dist)
            # print(np.sum(MED_prob_dist))
            # print(MED_prob_dist.shape)
            Arm_t, chosen = sample_action(self.X, MED_prob_dist)
            return chosen, np.nan
        radius_sq = self.multiplier * (self.gamma_t)
        if (self.subsample_func == None):
            prob_dist = calc_q_opt_design(self.AugX)
            emp_bst_opt_prob_dist = self.empirical_best_quo*self.empirical_best_ind  + self.opt_design_quo * prob_dist
            MED_prob_dist = np.multiply(emp_bst_opt_prob_dist, self.MED_quo)
            MED_prob_dist = MED_prob_dist / np.sum(MED_prob_dist)
            # print(MED_prob_dist)
            # print("Final probability distribution", MED_prob_dist)
            # print(np.sum(MED_prob_dist))
            # print(MED_prob_dist.shape)
            Arm_t, chosen = sample_action(self.X, MED_prob_dist)
            #return chosen, np.nan
        else:
            raise NotImplementedError()  # todo: use valid_idx
        #             idx = self.subsample_func(self.N, self.subN);
        #             subX = self.X[idx,:];
        #             obj_func = np.dot(subX, self.theta_hat) + np.sqrt(radius_sq) * \
        #                     np.sqrt( mahalanobis_norm_sq_batch(subX, self.invVt));
        #
        #             chosen = idx[np.argmax(obj_func)];
        return chosen, radius_sq

    def estimate_empirical_reward_gap(self):
        # print(theta_t.shape)
        # print(A.shape)
        reward_A = np.matmul(self.X, self.theta_hat)
        self.Delta_empirical_gap = np.max(reward_A) - reward_A

    def calc_MED_ver1_probability_distribution(self):
        a = self.X[self.empirical_best_arm, :]
        vVal_lev_score_emp_best = np.matmul(np.matmul(a.T, self.invVt), a)
        # print(vVal_lev_score_emp_best)
        # print(a.shape)
        for i in range(self.K):
            a = self.X[i, :]
            vVal_lev_score_a = np.matmul(np.matmul(a.T, self.invVt), a)
            self.MED_quo[i] = np.exp(
                -(self.Delta_empirical_gap[i]) ** 2 / ((self.gamma_t) * (vVal_lev_score_a + vVal_lev_score_emp_best)))

    def scale_arms(self):
        for i in range(self.K):
            self.AugX[i, :] = np.sqrt(self.MED_quo[i]) * self.X[i, :]

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
        #self.Vt.dropna(inplace=True)
        self.invVt = np.linalg.pinv(self.Vt )

        self.theta_hat = np.matmul(self.invVt, self.XTy.T)
        self.do_not_ask.append(pulled_idx)

        my_t = self.t + 1
        #self.sqrt_beta = calc_sqrt_beta_det2(self.d, my_t, self.R, self.lam, self.delta, self.S, self.logdetV)
        #self.gamma_t = calc_gamma_t(self.t, self.d, self.lam, self.delta, self.S, self.R)
        self.gamma_t = calc_gamma_t(self.t,self.d,self.lam,self.delta,self.S,self.R)
        self.estimate_empirical_reward_gap()

        self.empirical_best_arm = np.where(self.Delta_empirical_gap == 0)[0][0]
        self.empirical_best_ind = np.zeros(self.K)
        self.empirical_best_ind[self.empirical_best_arm] = 1

        self.calc_MED_ver1_probability_distribution()
        self.scale_arms()
        
        self.t = self.t +  1

    def getDoNotAsk(self):
        return self.do_not_ask

    def predict(self, X=None):
        if X is None:
            X = self.X
        return X.dot(self.theta_hat)

    def get_debug_dict(self):
        return self.dbg_dict