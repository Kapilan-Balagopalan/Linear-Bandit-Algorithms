from optimaldesign import *
from arms_generator import *
from Bandit_Env import *



#all the scalar values will start with sVal
#all the vector values will start with vVal
#all the matrix values will start with mVal
#underscore will be used to divide words
#meaning full names will not have vowels in it e.g leverage = 'lvrg'
class Lin_SGMED_NOPT(Bandit):
    ########################################
    def __init__(self, X, R, S,N ,opt_coeff,emp_coeff,flags):
        self.X = X
        self.R = R
        self.S = S
        self.N = N
        self.delta = .01
        self.flags = flags
        self.K, self.d = self.X.shape
        if(self.flags["type"] == "EOPT"):
            self.lam = self.R**2/self.S**2
        elif(self.flags["type"] == "Sphere"):
            self.lam = self.R**2/self.S**2
        self.delta = .01

        # more instance variables
        self.t = 1
        


        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.lam
        self.Vt = self.lam * np.eye(self.d)

        self.empirical_best_quo = emp_coeff
        self.opt_design_quo = opt_coeff
        self.each_arm_coeff = (1 - emp_coeff - opt_coeff)/self.K
        self.AugX = self.X.copy()

        self.MED_quo = np.ones(self.K)
        self.All_arm = np.ones(self.K)

        self.Delta_empirical_gap = np.ones(self.K)
        self.empirical_best_arm = 0
        self.gamma_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)

        self.logdetV = self.d * np.log(self.lam)

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.K), self.do_not_ask)
        if (self.t == 1):
            return np.random.randint(self.K) 
        
        qt = self.All_arm

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

        tempval1 = np.matmul(self.invVt, xt.T)  # d by 1, O(d^2)
        tempval2 = np.dot(tempval1, xt)  # scalar, O(d)
        self.logdetV += np.log(1 + tempval2)
      
        #self.invVt = np.linalg.inv(self.Vt )
        self.invVt = self.invVt - np.outer(tempval1, tempval1) / (1 + tempval2)
        #self.invVt = find_matrix_inverse_vt_method_fast(self.invVt, xt)

        theta_hat = np.matmul(self.invVt, self.XTy.T)
 
        self.gamma_t =  calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)

        self.estimate_empirical_reward_gap(self.X, theta_hat)
        if(self.flags["version"] == 1):
            self.calc_MED_ver1_probability_distribution()
        elif(self.flags["version"] == 2): 
            self.calc_MED_ver2_probability_distribution()
        else:
            raise NotImplementedError()
        
       # self.scale_arms()
        
        self.t = self.t +  1
