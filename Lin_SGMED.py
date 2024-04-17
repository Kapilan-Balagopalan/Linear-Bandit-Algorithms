from optimaldesign import *
from arms_generator import *
from Bandit_Env import *



#all the scalar values will start with sVal
#all the vector values will start with vVal
#all the matrix values will start with mVal
#underscore will be used to divide words
#meaning full names will not have vowels in it e.g leverage = 'lvrg'
class Lin_SGMED(Bandit):
    ########################################
    def __init__(self, X, lam, R, S, opt_coeff, flags):
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
        self.Vt = self.lam * np.eye(self.d)

        self.empirical_best_quo = opt_coeff
        self.opt_design_quo = 1 - opt_coeff
        self.AugX = self.X.copy()

        self.MED_quo = np.ones(self.K)

        self.Delta_empirical_gap = np.ones(self.K)
        self.empirical_best_arm = 0
        self.gamma_t = calc_gamma_t_SGMED(self.t,self.d,self.lam,self.delta,self.S,self.R)

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.K), self.do_not_ask)
        prob_dist = calc_q_opt_design(self.AugX)
        if (self.t == 1):
            MED_prob_dist = prob_dist / np.sum(prob_dist)

            Arm_t, chosen = sample_action(self.X, MED_prob_dist)
            return chosen
        
        qt =  self.opt_design_quo * prob_dist
        qt[self.empirical_best_arm] = qt[self.empirical_best_arm]  + self.empirical_best_quo

        MED_prob_dist = np.multiply(qt, self.MED_quo)
        MED_prob_dist = MED_prob_dist / np.sum(MED_prob_dist)
            # print(MED_prob_dist)
            # print("Final probability distribution", MED_prob_dist)
            # print(np.sum(MED_prob_dist))
            # print(MED_prob_dist.shape)
        Arm_t, chosen = sample_action(self.X, MED_prob_dist)

        return chosen

    def estimate_empirical_reward_gap(self,X,theta_hat):
        reward_A = np.matmul(X, theta_hat)
        self.empirical_best_arm = np.argmax(reward_A)
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

    def calc_MED_ver2_probability_distribution(self):
        a_hat = self.X[self.empirical_best_arm, :]
        for i in range(self.K):
            a = self.X[i, :]
            vVal_lev_score_a = np.matmul(np.matmul((a-a_hat).T, self.invVt), (a-a_hat))
            if(vVal_lev_score_a != 0):
                self.MED_quo[i] = np.exp(
                    -(self.Delta_empirical_gap[i]) ** 2 / ((self.gamma_t) * (vVal_lev_score_a)))
            else:
                self.MED_quo[i] = 1

    def scale_arms(self):
        for i in range(self.K):
            self.AugX[i, :] = np.sqrt(self.MED_quo[i]) * self.X[i, :]

    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]

        self.XTy = self.XTy +  y_t * xt
        self.Vt =  self.Vt + np.outer(xt, xt)
      
        self.invVt = np.linalg.inv(self.Vt )

        theta_hat = np.matmul(self.invVt, self.XTy.T)
 
        self.gamma_t = calc_gamma_t_SGMED(self.t,self.d,self.lam,self.delta,self.S,self.R)

        self.estimate_empirical_reward_gap(self.X, theta_hat)
        if(self.flags["version"] == 1):
            self.calc_MED_ver1_probability_distribution()
        elif(self.flags["version"] == 2): 
            self.calc_MED_ver2_probability_distribution()
        else:
            raise NotImplementedError()
        
        self.scale_arms()
        
        self.t = self.t +  1
