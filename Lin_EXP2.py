from optimaldesign import *
from arms_generator import *
from Bandit_Env import *



class Lin_EXP2(Bandit):
    ########################################
    def __init__(self, X, R, S, N, flags):
        self.X = X
        self.R = R
        self.S = S
        self.flags = flags
        self.K, self.d = self.X.shape
        self.N = N
        self.delta = .01

        # more instance variables
        self.t = 1
 
        self.theta_est = np.zeros((self.d,1))
        #print(self.theta_est.shape)
        self.AugX = self.X.copy()

        self.EXP2_prob_dist = np.ones((self.K,1))/self.K

        self.gamma_t = 1 

        self.prob_dist_opt = calc_q_opt_design(self.AugX)

        self.eta_t = 1

        #print(self.prob_dist_opt.shape)
        self.XTy = np.zeros((self.d,1))
        
        
        #print("original shape is",self.theta_est.shape)

    

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.K), self.do_not_ask)
        if (self.t == 1):
            self.EXP2_prob_dist = self.prob_dist_opt

            Arm_t, chosen = sample_action(self.X, self.EXP2_prob_dist)
            return chosen
        #print(self.X.shape)
        #print(self.theta_est.shape)
        exploit_part =  np.matmul(self.X, self.theta_est)
        #print(exploit_part.shape)
        self.eta_t = calc_eta_t_EXP2(self.t + 1,self.d,self.K)
        if(self.eta_t > 1/self.d):
            self.eta_t  = 1/self.d
        self.gamma_t = self.d*self.eta_t     

        exploit_part = np.exp(exploit_part*self.eta_t)
        exploit_part = exploit_part/np.sum(exploit_part)

        self.EXP2_prob_dist = (1-self.gamma_t)*exploit_part + self.gamma_t*self.prob_dist_opt
        #print(self.EXP2_prob_dist.shape)
        Arm_t, chosen = sample_action(self.X, self.EXP2_prob_dist)

        return chosen
    

    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]
        qt = np.eye(self.d)
        XTy  = y_t*xt

        for i in range(self.K):
            aa_t = np.outer(self.X[i,:],self.X[i,:])
            qt = qt + self.EXP2_prob_dist[i][0]*aa_t

        #print(qt.shape, "The shape of Qt")
        

        q_t_inv = np.linalg.inv(qt )

        theta_est_inst = np.matmul(q_t_inv,XTy.T)
        #print(theta_est_inst.shape)

        self.theta_est  = self.theta_est + theta_est_inst

        self.t = self.t +  1
