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
    def __init__(self, X, R, S, flags):
        self.X = X
        self.R = R
        self.S = S
        self.flags = flags
        if(self.flags["type"] == "EOPT"):
            self.lam = (self.R**2)/self.S**2
        elif(self.flags["type"] == "Sphere"):
            self.lam =(self.R**2)/self.S**2
        self.delta = .01
    


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


        self.best_ucb_arm = 0
        self.worst_ucb_arm = 0
        self.C = 1

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.K), self.do_not_ask)
        if (self.t == 1):
            return np.random.randint(self.K) 
        chosen = np.argmin(self.MED_quo)
        return chosen

    def estimate_empirical_reward_gap_ver1(self,X,theta_hat):
        reward_A = np.matmul(X, theta_hat)
        self.Delta_empirical_gap = np.max(reward_A) - reward_A

    def estimate_empirical_reward_gap_ver3(self,X,theta_hat):
        UCB_arr =  np.zeros(self.K)
        for i in range(self.K):
            a = self.X[i, :]
            vVal_lev_score_a = np.matmul(np.matmul(a.T, self.invVt), a)
            UCB_arr[i] = np.sqrt(((self.beta_t)*vVal_lev_score_a))
        reward_A = np.matmul(X, theta_hat)
        ver3_metric = reward_A + UCB_arr
        self.Delta_empirical_gap = np.max(ver3_metric) - ver3_metric
        self.best_ucb_arm = np.argmin(self.Delta_empirical_gap)
        self.worst_ucb_arm = np.argmax(self.Delta_empirical_gap)


    def calc_IMED_ver1_index(self):
        for i in range(self.K):
            a = self.X[i, :]
            vVal_lev_score_a = np.matmul(np.matmul(a.T, self.invVt), a)
            self.MED_quo[i] = ((self.Delta_empirical_gap[i]**2)/((self.beta_t)*vVal_lev_score_a)) - np.log((self.beta_t)*vVal_lev_score_a)

    def calc_IMED_ver3_index(self):
        for i in range(self.K):
            a = self.X[i, :]
            vVal_lev_score_a = np.matmul(np.matmul(a.T, self.invVt), a)
            self.MED_quo[i] = ((self.Delta_empirical_gap[i]**2)/((self.beta_t)*vVal_lev_score_a)) - np.log((self.beta_t)*vVal_lev_score_a)
        a = self.X[self.best_ucb_arm, :]
        vVal_lev_score_a = np.matmul(np.matmul(a.T, self.invVt), a)
        self.MED_quo[self.best_ucb_arm] = np.minimum(np.log(self.C/self.Delta_empirical_gap[self.worst_ucb_arm]) , -np.log((self.beta_t)*vVal_lev_score_a) )

    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]

        self.XTy = self.XTy +  y_t * xt
        self.Vt =  self.Vt + np.outer(xt, xt)

        self.invVt = np.linalg.inv(self.Vt )
        #self.invVt = find_matrix_inverse_vt_method_fast(self.invVt, xt)

        theta_hat = np.matmul(self.invVt, self.XTy.T)

        self.beta_t = calc_beta_t_LinIMED(self.t,self.d,self.lam,self.delta,self.S,self.R)
        
        

        if(self.flags["version"] == 1):
            self.estimate_empirical_reward_gap_ver1(self.X,theta_hat)
            self.calc_IMED_ver1_index() 
        elif (self.flags["version"] == 3):
            self.estimate_empirical_reward_gap_ver3(self.X,theta_hat)
            self.calc_IMED_ver3_index() 
        else:
            raise NotImplementedError() 
        self.t = self.t +  1
