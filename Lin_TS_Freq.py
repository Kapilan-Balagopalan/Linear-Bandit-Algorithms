from optimaldesign import *
from arms_generator import *
from Bandit_Env import *
from scipy import linalg


class Lin_TS_FREQ(Bandit):
    ########################################
    def __init__(self, X, R, S, N,d,delay_switch,delay_time, flags,n_mc_samples):
        self.d = d
        self.R = R
        self.S = S
        self.flags = flags

        self.N = N
        self.delta = .01
        self.delta_ts = self.delta
        self.oversample_coeff = 1

        self.lam = (self.R**2)/(self.S**2)


        self.t = 1

        self.multi_var_mean = np.zeros((self.d,1))
        self.multi_var_var = np.eye(self.d)

        self.XTy = np.zeros((1,self.d))
        self.invVt = np.eye(self.d) / self.lam
        self.invVt_sqrt = np.eye(self.d) /np.sqrt(self.lam)

        self.theta_hat = np.zeros((self.d,1))
        self.Vt = self.lam * np.eye(self.d)

        self.beta_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)


        self.logdetV = self.d * np.log(self.lam)

        self.prob_chosen = 0
        self.arm_chosen = 0
        self.n_mc_samples = n_mc_samples

        self.delay_switch = delay_switch
        self.delay_time = delay_time
        self.update_buffer_idx = Queue(maxsize=self.delay_time)
        self.update_buffer_y = Queue(maxsize=self.delay_time)

    #@profile
    def next_arm(self,X_t):
        K = X_t.shape[0]
        if (self.t == 1):
            return np.random.randint(K)

        eta_t = np.random.randn(self.d)

        temp = self.invVt_sqrt @ eta_t
        temp = temp.reshape((-1,1))

        if (self.flags["version"] == 2):
            theta_tilde = self.theta_hat + self.oversample_coeff*temp
        else:
            theta_tilde = self.theta_hat + self.oversample_coeff * temp *self.beta_t

        obj_func = X_t @ theta_tilde

        chosen = np.argmax(obj_func)

        self.arm_chosen = chosen
        return chosen



    def get_probability_arm(self,X_t):
        K = X_t.shape[0]
        if (self.t == 1):
            return 1/K

        eta_t = np.random.randn(self.d, self.n_mc_samples)
        # print(eta_t.shape)
        temp = np.matmul(self.invVt_sqrt, eta_t)
        #temp = temp.reshape((-1, 1))
        # print(temp.shape)
        theta_rep = np.tile(self.theta_hat,(1,self.n_mc_samples))
        theta_tilde = theta_rep + self.oversample_coeff * self.beta_t * temp
        obj_func = np.matmul(X_t, theta_tilde)
        argmax_arr = np.argmax(obj_func,axis = 0)
        argmax_count = np.bincount(argmax_arr)
        return argmax_count[self.arm_chosen]/self.n_mc_samples


   # @profile
    def update(self, xt, y_t):

        self.XTy += (y_t) * xt
        self.Vt += np.outer(xt,xt)

        tempval1 = self.invVt @ xt
        tempval2 = xt @ tempval1
        self.logdetV += np.log(1 + tempval2)

        evalues, evectors = np.linalg.eig(self.Vt)
        # Ensuring square root matrix exists
        assert (evalues >= 0).all()
        self.invVt = evectors * np.reciprocal(evalues) @ evectors.T

        self.invVt_sqrt = evectors * np.sqrt(np.reciprocal(evalues)) @ evectors.T
        

        self.theta_hat = self.invVt @ self.XTy.T


        self.beta_t = calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)


    def update_delayed(self, xt, y_t):
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