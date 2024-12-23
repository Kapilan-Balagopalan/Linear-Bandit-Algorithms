from optimaldesign import *
from arms_generator import *

from Bandit_Env import *

from scipy.optimize import brentq



class Lin_ZHU(Bandit):
    ########################################
    def __init__(self, X, R, S,N,d, delay_switch,delay_time,flags):
        self.d = d
        self.R = R
        self.S = S
        self.flags = flags
        self.lam = (self.R**2)/self.S**2
        self.delta = .01
        self.N = N


        # more instance variables
        self.t = 1

        self.XTy = np.zeros((1,self.d))
        self.theta_hat = np.zeros((self.d,1))
        self.invVt = np.eye(self.d) / self.lam
        self.Vt = self.lam * np.eye(self.d)

        self.empirical_best_quo = 0.5
        self.opt_design_quo = 0.5
        

        self.empirical_best_arm = 0
        if(self.flags["version"] == "anytime"):
            self.gamma = calc_gamma_LinZHU(self.t + 1,self.d,self.delta)
        else:
            self.gamma = calc_gamma_LinZHU(self.N,self.d,self.delta)

        self.eta = calc_eta_LinZHU(self.gamma, self.d)
        self.logdetV = self.d * np.log(self.lam)

        self.delay_switch = delay_switch
        self.delay_time = delay_time
        self.update_buffer_xt = Queue(maxsize=self.delay_time)
        self.update_buffer_y = Queue(maxsize=self.delay_time)

    def calc_ZHU_probability_distribution(self,qt,lam_true,X,Delta_empirical_gap):
        K = X.shape[0]
        MED_quo = np.ones((K,1))
        for i in range(K):
            MED_quo[i][0] = qt[i][0] /(lam_true + self.eta*Delta_empirical_gap[i][0])
    
        return MED_quo

    def find_lambda(self,x,qt,Delta_empirical_gap):
        temp = 0
        K = qt.shape[0]
        for i in range(K):
            temp = temp + qt[i][0] / (x+ self.eta * Delta_empirical_gap[i][0])
        return temp - 1

    def next_arm(self,X_t):
        AugX = X_t.copy()
        K = X_t.shape[0]

        if (self.t > 1):
            Delta_empirical_gap = self.estimate_empirical_reward_gap(X_t, self.theta_hat)
            AugX = self.scale_arms(X_t,Delta_empirical_gap)

        prob_dist = calc_q_opt_design(AugX)

        if (self.t == 1):    
            MED_prob_dist = prob_dist
            Arm_t, chosen = sample_action(X_t, MED_prob_dist)
            return chosen
        
        qt =  self.opt_design_quo * prob_dist
        qt[self.empirical_best_arm][0] = qt[self.empirical_best_arm][0]  + self.empirical_best_quo

        if(self.flags["version"] == "anytime"):
            self.gamma = calc_gamma_LinZHU(self.t + 1 ,self.d,self.delta)
            self.eta = calc_eta_LinZHU(self.gamma, self.d)

       
        lam_true = brentq(self.find_lambda, 0.4,1.2, args=(qt,Delta_empirical_gap))
        MED_quo = self.calc_ZHU_probability_distribution(qt,lam_true,X_t,Delta_empirical_gap)
        MED_prob_dist = MED_quo / np.sum(MED_quo)
        Arm_t, chosen = sample_action(X_t, MED_prob_dist)

        return chosen

    def estimate_empirical_reward_gap(self,X,theta_hat):
        reward_A = np.matmul(X, theta_hat)
        self.empirical_best_arm = np.argmax(reward_A)
        Delta_empirical_gap = reward_A[self.empirical_best_arm]- reward_A
        return Delta_empirical_gap



    def scale_arms(self,X,Delta_empirical_gap):
        K = X.shape[0]
        AugX = np.zeros((K,self.d))
        for i in range(K):
            AugX[i,:] = X[i,:]/(np.sqrt(1 + self.eta*Delta_empirical_gap[i][0]))
        return AugX


    def update(self, xt, y_t):

        self.XTy = self.XTy + y_t * xt
        self.Vt = self.Vt + np.outer(xt, xt)

        tempval1 = self.invVt @ xt.T
        tempval2 = xt @ tempval1
        self.logdetV += np.log(1 + tempval2)

        self.invVt = self.invVt - np.outer(tempval1, tempval1) / (1 + tempval2)

        self.theta_hat = self.invVt @ self.XTy.T



    def update_delayed(self, xt, y_t):
        if (self.delay_switch == False):
            self.update(xt, y_t)
        else:
            self.update_buffer_y.put(y_t)
            self.update_buffer_xt.put(xt)
            if (self.t % self.delay_time == 1):
                queue_sze = self.update_buffer_y.qsize()
                for i in range(queue_sze):
                    self.update(self.update_buffer_xt.get(), self.update_buffer_y.get())

        self.t = self.t + 1