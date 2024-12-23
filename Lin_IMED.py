from optimaldesign import *
from arms_generator import *
from Bandit_Env import *


class Lin_IMED(Bandit):
    ########################################
    def __init__(self, X, R, S,N,d,delay_switch,delay_time, flags):
        self.d = d
        self.R = R
        self.S = S
        self.flags = flags
        self.lam = (self.R**2)/self.S**2
        self.delta = .01

        self.delay_switch = delay_switch
        self.delay_time = delay_time
        self.update_buffer_xt = Queue(maxsize=self.delay_time)
        self.update_buffer_y = Queue(maxsize=self.delay_time)

        # more instance variables
        self.t = 1

        self.theta_hat = np.zeros((d,1))

        self.XTy = np.zeros((1,self.d))
        self.invVt = np.eye(self.d) / self.lam
       

        self.theta_hat = np.zeros((self.d,1))
        self.Vt = self.lam * np.eye(self.d)

        self.beta_t = calc_sqrt_beta_det2_initial(self.R, self.lam, self.delta, self.S)


        self.empirical_best_quo = 0.5
        self.opt_design_quo = 0.5

        self.empirical_best_arm = 0


        self.best_ucb_arm = 0
        self.worst_ucb_arm = 0
        self.C = 30

        self.logdetV = self.d * np.log(self.lam)

    def next_arm(self,X_t):

        K = X_t.shape[0]

        if(self.t > 1):
            if (self.flags["version"] == 1):
                Delta_empirical_gap = self.estimate_empirical_reward_gap_ver1(X_t, self.theta_hat)
                MED_quo = self.calc_IMED_ver1_index(X_t, Delta_empirical_gap)
            elif (self.flags["version"] == 3):
                Delta_empirical_gap = self.estimate_empirical_reward_gap_ver3(X_t, self.theta_hat)
                MED_quo = self.calc_IMED_ver3_index(X_t, Delta_empirical_gap)
            else:
                raise NotImplementedError()

            self.t = self.t + 1

        if (self.t == 1):
            return np.random.randint(K)

        chosen = np.argmin(MED_quo)
        #print(chosen)
        return chosen

    def estimate_empirical_reward_gap_ver1(self,X,theta_hat):
        reward_A = X @ theta_hat
        return np.max(reward_A) - reward_A

    def estimate_empirical_reward_gap_ver3(self,X,theta_hat):
        K = X.shape[0]
        UCB_arr =  np.zeros((K,1))
        for i in range(K):
            a = X[i,:]
            vVal_lev_score_a = (a @ self.invVt) @ a
            UCB_arr[i][0] = np.sqrt(((self.beta_t)*vVal_lev_score_a))
        reward_A = X @ theta_hat
        ver3_metric = reward_A + UCB_arr
        Delta_empirical_gap = np.max(ver3_metric) - ver3_metric
        self.best_ucb_arm = np.argmin(Delta_empirical_gap)
        self.worst_ucb_arm = np.argmax(Delta_empirical_gap)
        return Delta_empirical_gap


    def calc_IMED_ver1_index(self,X,Delta_empirical_gap):
        K = X.shape[0]
        MED_quo = np.zeros((K,1))
        for i in range(K):
            a = X[i, :]
            vVal_lev_score_a = (a @ self.invVt) @ a
            #print(vVal_lev_score_a)
            MED_quo[i] = ((Delta_empirical_gap[i]**2)/((self.beta_t)*vVal_lev_score_a)) - np.log((self.beta_t)*vVal_lev_score_a)
        return MED_quo


    def calc_IMED_ver3_index(self,X,Delta_empirical_gap):
        K = X.shape[0]
        MED_quo = np.zeros((K, 1))
        for i in range(K):
            a = X[i, :]
            vVal_lev_score_a = (a @ self.invVt) @ a
            #print(self.beta_t)
            MED_quo[i] = ((Delta_empirical_gap[i]**2)/((self.beta_t)*vVal_lev_score_a)) - np.log((self.beta_t)*vVal_lev_score_a)
        a = X[self.best_ucb_arm, :]
        vVal_lev_score_a = (a @ self.invVt) @ a
        MED_quo[self.best_ucb_arm] = np.minimum(np.log(self.C/(Delta_empirical_gap[self.worst_ucb_arm]**2)) , -np.log((self.beta_t)*vVal_lev_score_a) )


    def update(self, xt, y_t):

        self.XTy += (y_t) * xt
        self.Vt += np.outer(xt, xt)

        tempval1 = self.invVt @ xt.T
        tempval2 = xt @ tempval1
        self.logdetV += np.log(1 + tempval2)

        self.invVt = self.invVt - np.outer(tempval1, tempval1) / (1 + tempval2)

        self.theta_hat = self.invVt @ self.XTy.T

        self.beta_t = calc_sqrt_beta_det2(self.d, self.R, self.lam, self.delta, self.S, self.logdetV)

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