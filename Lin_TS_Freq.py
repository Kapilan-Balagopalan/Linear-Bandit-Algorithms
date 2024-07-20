from optimaldesign import *
from arms_generator import *
from Bandit_Env import *
from scipy import linalg


#all the scalar values will start with sVal
#all the vector values will start with vVal
#all the matrix values will start with mVal
#underscore will be used to divide words
#meaning full names will not have vowels in it e.g leverage = 'lvrg'
class Lin_TS_FREQ(Bandit):
    ########################################
    def __init__(self, X, R, S, N, flags):
        self.X = X
        self.R = R
        self.S = S
        self.flags = flags
        self.K, self.d = self.X.shape
        self.N = N
        self.delta = .01
        self.delta_ts = self.delta
        self.oversample_coeff = 1
        if(self.flags["type"] == "EOPT"):
            self.lam = (self.R**2)/(self.S**2)
        elif(self.flags["type"] == "Sphere"):
            self.lam = (self.R**2)/(self.S**2)

        # more instance variables
        self.t = 1

        self.multi_var_mean = np.zeros(self.d)
        self.multi_var_var = np.eye(self.d)

        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.lam
        self.invVt_sqrt = np.eye(self.d) 
        self.X_invVt_norm_sq = np.sum(self.X * self.X, axis=1) / self.lam
        self.logdetV = self.d*np.log(self.lam)
        self.sqrt_beta = calc_sqrt_beta_det2(self.d,self.t,self.R,self.lam,self.delta,self.S,self.logdetV)
        self.theta_hat = np.zeros(self.d)
        self.Vt = self.lam * np.eye(self.d)

        if (self.flags["type"] == "EOPT") :
            self.beta_t = calc_beta_t_OFUL(self.t, self.d, self.lam, self.delta_ts, self.S, self.R)
        elif(self.flags["type"] == "Sphere") :
            self.beta_t = calc_beta_t_OFUL(self.t, self.d, self.lam, self.delta_ts, self.S, self.R)
        else:
             raise NotImplementedError() # todo: use valid_idx
 
        
        
        #print("original shape is",self.theta_est.shape)

    

    def next_arm(self):

        if (self.t == 1):
            return np.random.randint(self.K) 
        

        eta_t = np.random.multivariate_normal(self.multi_var_mean , self.multi_var_var, 1)
        temp = np.matmul(self.invVt_sqrt, eta_t.T)
        theta_tilde = self.theta_hat + self.oversample_coeff*self.beta_t*temp[:,0]
        #obj_func = np.dot(self.X, self.theta_hat) + np.sqrt(radius_sq) * np.sqrt(self.X_invVt_norm_sq)
        obj_func = np.zeros(self.K)
        for i in range(self.K):
            obj_func[i] = np.dot(self.X[i], theta_tilde.T)
            #print(np.dot(self.X[i], theta_tilde.T))

        chosen = np.argmax(obj_func)

        return chosen
    

    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]

        self.XTy += (y_t) * xt
        self.Vt += np.outer(xt,xt)

        evalues, evectors = np.linalg.eig(self.Vt)
        # Ensuring square root matrix exists
        assert (evalues >= 0).all()
        self.invVt = evectors * np.reciprocal(evalues) @ np.linalg.inv(evectors)
        #self.invVt_sqrt = linalg.sqrtm(self.invVt )
        self.invVt_sqrt = evectors * np.sqrt(np.reciprocal(evalues)) @ np.linalg.inv(evectors)
        
        #self.invVt = np.linalg.inv(self.Vt)
        self.theta_hat = np.dot(self.invVt, self.XTy)



        if (self.flags["type"] == "EOPT") :
            self.beta_t = calc_beta_t_OFUL(self.t, self.d, self.lam, self.delta_ts, self.S, self.R)
        elif(self.flags["type"] == "Sphere") :
            self.beta_t = calc_beta_t_OFUL(self.t, self.d, self.lam, self.delta_ts, self.S, self.R)
        else:
             raise NotImplementedError() # todo: use valid_idx
    

        

        self.t = self.t  + 1