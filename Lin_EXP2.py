from optimaldesign import *
from arms_generator import *
from Bandit_Env import *



#all the scalar values will start with sVal
#all the vector values will start with vVal
#all the matrix values will start with mVal
#underscore will be used to divide words
#meaning full names will not have vowels in it e.g leverage = 'lvrg'
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
 
        self.theta_est = np.zeros(self.d)
        
        self.AugX = self.X.copy()

        self.EXP2_prob_dist = np.ones(self.K)/self.K

        self.gamma_t = 1 

        self.prob_dist_opt = calc_q_opt_design(self.AugX)

        self.eta_t = 1


        self.XTy = np.zeros(self.d)
        
        
        #print("original shape is",self.theta_est.shape)

    

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.K), self.do_not_ask)
        if (self.t == 1):
            self.EXP2_prob_dist = self.prob_dist_opt

            Arm_t, chosen = sample_action(self.X, self.EXP2_prob_dist)
            return chosen
        
        exploit_part =  np.matmul(self.X, self.theta_est) 
        self.eta_t = calc_eta_t_EXP2(self.t + 1,self.d,self.K)
        self.gamma_t = self.d*self.eta_t     

        exploit_part = np.exp(exploit_part*self.eta_t)
        exploit_part = exploit_part/np.sum(exploit_part)
        #print(exploit_part.shape)
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
            qt = qt + self.EXP2_prob_dist[i]*aa_t

        #print(qt.shape, "The shape of Qt")
        

        q_t_inv = np.linalg.inv(qt )

        theta_est_inst = np.matmul(q_t_inv,XTy.T)


        self.theta_est  = self.theta_est + theta_est_inst[:,0]

        self.t = self.t +  1
