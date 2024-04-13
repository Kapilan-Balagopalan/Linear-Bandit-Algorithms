from optimaldesign import *
from arms_generator import *

from Bandit_Env import *

from scipy.optimize import brentq

def sample_action(A,MED_prob_dist):
    K,d = A.shape
    ind = np.random.choice(K, 1, p= MED_prob_dist)
    return A[ind,:], ind

def calc_q_opt_design(A):
    sVal_opt_design_arms, sampling_time = optimal_design_algo(A)
    prob_dist = optimal_probability(A, sVal_opt_design_arms)
    return prob_dist

def calc_gamma(N,d, delta):
    c_OPT = 4
    gamma = np.power(((3*N*np.sqrt(c_OPT)*np.sqrt(d))/(2*d*np.log(N)) + 64 *np.log(2/delta) ),2/3)
    return gamma

def calc_eta(gamma,d):
    c_OPT = 4
    eta = gamma / (c_OPT*d)
    return eta


class Lin_ZHU(Bandit):
    ########################################
    def __init__(self, X, lam, R, S,N, flags):
        self.X = X
        self.R = R
        self.lam = lam
        self.delta = .01
        self.S = S
        self.flags = flags
        self.N = N


        # more instance variables
        self.t = 1
        self.K, self.d = self.X.shape

        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.lam

        self.theta_hat = np.zeros(self.d)
        self.Vt = self.lam * np.eye(self.d)

        self.empirical_best_quo = 0.5
        self.opt_design_quo = 0.5
        self.AugX = self.X.copy()
        
        self.Delta_empirical_gap = np.ones(self.K)
        self.empirical_best_arm = 0

        self.gamma = calc_gamma(self.N,self.d,self.delta)
        self.eta = calc_eta(self.gamma, self.d)

    def calc_ZHU_probability_distribution(self,qt,lam_true):

        MED_quo = np.ones(self.K)
        for i in range(self.K):
            MED_quo[i] = qt[i] /(lam_true + self.eta*self.Delta_empirical_gap[i])
    
        return MED_quo

    def find_lambda(self,x,qt):
        temp = 0
        for i in range(self.K):
            temp = temp + qt[i] / (x+ self.eta * self.Delta_empirical_gap[i])
        return temp - 1

    def next_arm(self):

        prob_dist = calc_q_opt_design(self.AugX)
        if (self.t == 1):    
            MED_prob_dist = prob_dist
            Arm_t, chosen = sample_action(self.X, MED_prob_dist)
            return chosen
        
        qt =  self.opt_design_quo * prob_dist
        qt[self.empirical_best_arm] = qt[self.empirical_best_arm]  + self.empirical_best_quo
       
        lam_true = brentq(self.find_lambda, 0.4,1.2, args=(qt))
        
        MED_quo = self.calc_ZHU_probability_distribution(qt,lam_true)
        
        MED_prob_dist = MED_quo / np.sum(MED_quo)
  

        Arm_t, chosen = sample_action(self.X, MED_prob_dist)
        
    
        return chosen

    def estimate_empirical_reward_gap(self,X,theta_hat):
        reward_A = np.matmul(X, theta_hat)
        self.empirical_best_arm = np.argmax(reward_A)
        self.Delta_empirical_gap = np.max(reward_A) - reward_A



    def scale_arms(self):
        for i in range(self.K):
            self.AugX[i, :] = self.X[i,:]/(np.sqrt(1 + self.eta*self.Delta_empirical_gap[i]))

    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]

        self.XTy = self.XTy +  y_t * xt
        self.Vt =  self.Vt + np.outer(xt, xt)

        self.invVt = np.linalg.inv(self.Vt )

        theta_hat = np.matmul(self.invVt, self.XTy.T)

        self.estimate_empirical_reward_gap(self.X, theta_hat)

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