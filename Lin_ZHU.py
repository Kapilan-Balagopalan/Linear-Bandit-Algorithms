from optimaldesign import *
from arms_generator import *

from Bandit_Env import *

from scipy.optimize import brentq

class Lin_ZHU(Bandit):
    ########################################
    def __init__(self, X, R, S,N, flags):
        self.X = X
        self.R = R
        self.S = S
        self.flags = flags
        if(self.flags["type"] == "EOPT"):
            self.lam = (self.R**2)/self.S**2
        elif(self.flags["type"] == "Sphere"):
            self.lam =(self.R**2)/self.S**2
        self.delta = .01
        self.N = N


        # more instance variables
        self.t = 1
        self.K, self.d = self.X.shape

        self.XTy = np.zeros((1,self.d))
        self.invVt = np.eye(self.d) / self.lam
        self.Vt = self.lam * np.eye(self.d)

        self.empirical_best_quo = 0.5
        self.opt_design_quo = 0.5
        self.AugX = self.X.copy()
        
        self.Delta_empirical_gap = np.ones((self.K,1))
        self.empirical_best_arm = 0
        if(self.flags["version"] == "anytime"):
            self.gamma = calc_gamma_LinZHU(self.t + 1,self.d,self.delta)
        else:
            self.gamma = calc_gamma_LinZHU(self.N,self.d,self.delta)
        self.eta = calc_eta_LinZHU(self.gamma, self.d)

    def calc_ZHU_probability_distribution(self,qt,lam_true):

        MED_quo = np.ones((self.K,1))
        for i in range(self.K):
            MED_quo[i][0] = qt[i][0] /(lam_true + self.eta*self.Delta_empirical_gap[i][0])
    
        return MED_quo

    def find_lambda(self,x,qt):
        temp = 0
        for i in range(self.K):
            temp = temp + qt[i][0] / (x+ self.eta * self.Delta_empirical_gap[i][0])
        return temp - 1

    def next_arm(self):

        prob_dist = calc_q_opt_design(self.AugX)
        if (self.t == 1):    
            MED_prob_dist = prob_dist
            Arm_t, chosen = sample_action(self.X, MED_prob_dist)
            return chosen
        
        qt =  self.opt_design_quo * prob_dist
        qt[self.empirical_best_arm][0] = qt[self.empirical_best_arm][0]  + self.empirical_best_quo

        if(self.flags["version"] == "anytime"):
            self.gamma = calc_gamma_LinZHU(self.t + 1 ,self.d,self.delta)
            self.eta = calc_eta_LinZHU(self.gamma, self.d)

       
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
            self.AugX[i,:] = self.X[i,:]/(np.sqrt(1 + self.eta*self.Delta_empirical_gap[i][0]))

    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]

        self.XTy = self.XTy +  y_t * xt
        self.Vt =  self.Vt + np.outer(xt, xt)

        tempval1 = np.matmul(self.invVt, xt.T)  # d by 1, O(d^2)
        tempval2 = np.matmul(xt, tempval1)  # scalar, O(d)
        #self.invVt = np.linalg.inv(self.Vt )
        self.invVt = self.invVt - np.outer(tempval1, tempval1) / (1 + tempval2)

        #self.invVt = find_matrix_inverse_vt_method_fast(self.invVt, xt)

        theta_hat = np.matmul(self.invVt, self.XTy.T)

        self.estimate_empirical_reward_gap(self.X, theta_hat)

        self.scale_arms()

        self.t = self.t +  1
