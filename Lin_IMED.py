from optimaldesign import *
from arms_generator import *
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
       

        self.theta_hat = np.zeros(self.d)
        self.Vt = self.lam * np.eye(self.d)
        self.beta_t = calc_beta_t_LinIMED(self.t,self.d,self.lam,self.delta,self.S,self.R)

       

        self.MED_quo = np.ones(self.K)
        self.empirical_best_quo = 0.5
        self.opt_design_quo = 0.5
        self.AugX = self.X.copy()
 
        self.Delta_empirical_gap = np.ones(self.K)
        self.empirical_best_arm = 0
   

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.K), self.do_not_ask)
        if (self.t == 1):
            return np.random.randint(self.K) 
        chosen = np.argmin(self.MED_quo)
        return chosen

    def estimate_empirical_reward_gap(self,X,theta_hat):
        reward_A = np.matmul(X, theta_hat)
        self.Delta_empirical_gap = np.max(reward_A) - reward_A


    def calc_IMED_ver1_index(self):
        for i in range(self.K):
            a = self.X[i, :]
            vVal_lev_score_a = np.matmul(np.matmul(a.T, self.invVt), a)
            self.MED_quo[i] = ((self.Delta_empirical_gap[i]**2)/((self.beta_t)*vVal_lev_score_a)) - np.log((self.beta_t)*vVal_lev_score_a)

    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]

        self.XTy = self.XTy +  y_t * xt
        self.Vt =  self.Vt + np.outer(xt, xt)

        self.invVt = np.linalg.inv(self.Vt )

        theta_hat = np.matmul(self.invVt, self.XTy.T)

        self.beta_t = calc_beta_t_LinIMED(self.t,self.d,self.lam,self.delta,self.S,self.R)
        
        self.estimate_empirical_reward_gap(self.X,theta_hat)

        if(self.flags["version"] == 1):
            self.calc_IMED_ver1_index()  
        self.t = self.t +  1
