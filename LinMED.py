from optimaldesign import *
from arms_generator import *
from Bandit_Env import *
from queue import Queue

class Lin_SGMED(Bandit):
    ########################################
    def __init__(self, X, R, S,N ,d,opt_coeff,emp_coeff,delay_switch,delay_time, flags):
        self.d = d
        self.R = R
        self.S = S
        self.N = N
        self.delta = .01
        self.flags = flags


        self.lam = self.R**2/self.S**2

        # more instance variables
        self.t = 1

        self.XTy = np.zeros((1,self.d))

        self.invVt = np.eye(self.d) / self.lam


        self.Vt = self.lam * np.eye(self.d)

        self.empirical_best_quo = emp_coeff
        self.opt_design_quo = opt_coeff
        self.each_arm_coeff = (1 - emp_coeff - opt_coeff)

        self.theta_hat = np.zeros((self.d,1))


        self.empirical_best_arm = 0

        self.gamma_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)


        self.logdetV = self.d * np.log(self.lam)

        self.prob_chosen = 0

        self.delay_switch = delay_switch
        self.delay_time = delay_time
        self.update_buffer_idx = Queue(maxsize = self.delay_time)
        self.update_buffer_y = Queue(maxsize = self.delay_time)


    def check_event_E(self, X_t):
        K = X_t.shape[0]
        for i in range(K):
            a = X_t[i,:]
            vVal_lev_score_a = np.matmul(np.matmul(a, self.invVt), a.T)
            if (vVal_lev_score_a > 1):
                return i,True

        return 0, False




    def next_arm(self,X_t):
        AugX = X_t.copy()
        K = X_t.shape[0]

        if (self.t > 1):
            Delta_empirical_gap = self.estimate_empirical_reward_gap(X_t , self.theta_hat)

            MED_quo = self.calc_MED_ver2_probability_distribution(X_t,Delta_empirical_gap)

            for i in range(K):
                AugX[i, :] = np.sqrt(MED_quo[i][0]) * X_t[i, :]


        prob_dist = calc_q_opt_design(AugX)

        if (self.t == 1):
            MED_prob_dist = (prob_dist / np.sum(prob_dist))*self.opt_design_quo + (1-self.opt_design_quo)/K

            Arm_t, chosen = sample_action(X_t, MED_prob_dist)
            self.prob_chosen = MED_prob_dist[chosen][0]
            return chosen


        
        qt =  self.opt_design_quo * prob_dist + self.each_arm_coeff/K
        qt[self.empirical_best_arm] = qt[self.empirical_best_arm] + self.empirical_best_quo

        MED_prob_dist = np.multiply(qt, MED_quo)
        MED_prob_dist = MED_prob_dist / np.sum(MED_prob_dist)

        index, check_event_E = self.check_event_E(X_t)
        check_event_E = False

        if(check_event_E == True):
            MED_prob_dist =MED_prob_dist/2
            MED_prob_dist[index] = MED_prob_dist[index] + 0.5
            Arm_t, chosen = sample_action(X_t, MED_prob_dist)
            #print("The right class is implemented")
        else:
            Arm_t, chosen = sample_action(X_t, MED_prob_dist)
        self.prob_chosen = MED_prob_dist[chosen][0]
        return chosen


    def estimate_empirical_reward_gap(self,X,theta_hat):
        reward_A = np.matmul(X, theta_hat)
        self.empirical_best_arm = np.argmax(reward_A)
        return reward_A[self.empirical_best_arm] - reward_A


    def calc_MED_ver2_probability_distribution(self, X_t, Delta_empirical_gap):
        K, d = X_t.shape
        a_hat = X_t[self.empirical_best_arm, :]
        MED_quo = np.zeros((K, 1))
        for i in range(K):
            a = X_t[i,:]
            vVal_lev_score_a = ((a- a_hat) @ self.invVt) @ (a- a_hat)
            if (vVal_lev_score_a != 0):
                MED_quo [i,0] = np.exp(-(Delta_empirical_gap[i,0]) ** 2 / ((self.gamma_t) * (vVal_lev_score_a)))
            else :
                MED_quo[i,0] = 1

        return MED_quo





    #@profile
    def get_probability_arm(self,X_t):
        return self.prob_chosen



    def update(self, xt, y_t):

        self.XTy = self.XTy +  y_t * xt
        self.Vt =  self.Vt + np.outer(xt, xt)

        tempval1 = self.invVt @ xt.T
        tempval2 = xt @ tempval1
        self.logdetV += np.log(1 + tempval2)


        self.invVt = self.invVt - np.outer(tempval1, tempval1) / (1 + tempval2)

        self.theta_hat  = np.matmul(self.invVt, self.XTy.T)


        self.gamma_t = calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)


    def update_delayed(self,xt, y_t):
        if (self.delay_switch == False):
            self.update(xt, y_t)
        else:
            self.update_buffer_y.put(y_t)
            self.update_buffer_idx.put(xt)
            if (self.t % self.delay_time == 1):
                queue_sze = self.update_buffer_y.qsize()
                for i in range(queue_sze):
                    self.update(self.update_buffer_idx.get(), self.update_buffer_y.get())

        self.t = self.t + 1