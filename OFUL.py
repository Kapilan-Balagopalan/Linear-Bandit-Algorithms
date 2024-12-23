from Bandit_Env import *
from queue import Queue

class Oful(Bandit):
########################################
    def __init__(self, X, R, S,N,d, delay_switch,delay_time, flags):
        self.d = d
        self.R = R
        self.S = S
        self.flags = flags

        self.lam = self.R**2/self.S**2

        self.delta = .01
        # more instance variables
        self.t = 1
        self.over_sampling_rate = 1

        self.delay_switch = delay_switch
        self.delay_time = delay_time
        self.update_buffer_xt = Queue(maxsize=self.delay_time)
        self.update_buffer_y = Queue(maxsize=self.delay_time)

        self.XTy = np.zeros((1,self.d))
        self.invVt = np.eye(self.d) / self.lam

        self.logdetV = self.d*np.log(self.lam)
        self.theta_hat = np.zeros((self.d,1))
        self.Vt = self.lam * np.eye(self.d)

        self.beta_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)


    def next_arm(self, X_t):
        K = X_t.shape[0]
        if (self.t == 1):
            return np.random.randint(K)

        obj_func = np.zeros((K,1))
        for i in range(K):
            obj_func[i][0] = X_t[i] @ self.theta_hat + self.over_sampling_rate*np.sqrt(self.beta_t) * np.sqrt((X_t[i] @ self.invVt)@ X_t[i])

        chosen = np.argmax(obj_func)
        return chosen

    def update(self, xt , y_t):

        self.XTy += (y_t) * xt
        self.Vt += np.outer(xt,xt)

        tempval1 = self.invVt @ xt.T
        tempval2 = xt @ tempval1
        self.logdetV += np.log(1 + tempval2)


        self.invVt = self.invVt - np.outer(tempval1, tempval1) / (1 + tempval2)

        self.theta_hat = self.invVt @ self.XTy.T

        self.beta_t = calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)


    def update_delayed(self, xt,y_t):
        if (self.delay_switch == False):
            self.update(xt,y_t)
        else:
            self.update_buffer_y.put(y_t)
            self.update_buffer_xt.put(xt)
            if (self.t % self.delay_time == 1):
                queue_sze = self.update_buffer_y.qsize()
                for i in range(queue_sze):
                    self.update(self.update_buffer_xt.get(), self.update_buffer_y.get())

        self.t = self.t + 1