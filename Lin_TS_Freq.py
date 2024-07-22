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
    def __init__(self, X, R, S, N, flags,n_mc_samples):
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

        self.multi_var_mean = np.zeros((self.d,1))
        self.multi_var_var = np.eye(self.d)

        self.XTy = np.zeros((1,self.d))
        self.invVt = np.eye(self.d) / self.lam
        self.invVt_sqrt = np.eye(self.d) /np.sqrt(self.lam)

        self.theta_hat = np.zeros((self.d,1))
        self.Vt = self.lam * np.eye(self.d)

        if (self.flags["type"] == "EOPT") :
            self.beta_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)
        elif(self.flags["type"] == "Sphere") :
            self.beta_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)
        else:
             raise NotImplementedError() # todo: use valid_idx

        self.logdetV = self.d * np.log(self.lam)
        #print("original shape is",self.theta_est.shape)
        self.prob_chosen = 0
        self.arm_chosen = 0
        self.n_mc_samples = n_mc_samples

    def next_arm(self):

        if (self.t == 1):
            return np.random.randint(self.K) 
        

        eta_t = np.random.multivariate_normal(self.multi_var_mean.ravel() , self.multi_var_var, 1)
        #print(eta_t.shape)
        temp = np.matmul(self.invVt_sqrt, eta_t.T)
        #print(temp.shape)
        theta_tilde = self.theta_hat + self.oversample_coeff*self.beta_t*temp
        #obj_func = np.dot(self.X, self.theta_hat) + np.sqrt(radius_sq) * np.sqrt(self.X_invVt_norm_sq)
        obj_func = np.zeros((self.K,1))
        for i in range(self.K):
            obj_func[i][0] = np.matmul(self.X[i], theta_tilde)
            #print(np.dot(self.X[i], theta_tilde.T))

        chosen = np.argmax(obj_func)
        self.arm_chosen = chosen
        return chosen

    def get_probability_arm(self):
        if (self.t == 1):
            return 1/self.K
        prob_list = np.zeros((self.K,1))
        for i in range(self.n_mc_samples):
            eta_t = np.random.multivariate_normal(self.multi_var_mean.ravel(), self.multi_var_var, 1)
            # print(eta_t.shape)
            temp = np.matmul(self.invVt_sqrt, eta_t.T)
            # print(temp.shape)
            theta_tilde = self.theta_hat + self.oversample_coeff * self.beta_t * temp
            # obj_func = np.dot(self.X, self.theta_hat) + np.sqrt(radius_sq) * np.sqrt(self.X_invVt_norm_sq)
            obj_func = np.zeros((self.K, 1))
            for i in range(self.K):
                obj_func[i][0] = np.matmul(self.X[i], theta_tilde)
                # print(np.dot(self.X[i], theta_tilde.T))

            chosen = np.argmax(obj_func)
            prob_list[chosen][0] = prob_list[chosen][0] + 1

        return prob_list[self.arm_chosen]/self.n_mc_samples


    def update(self, pulled_idx, y_t):

        xt = self.X[pulled_idx, :]

        self.XTy += (y_t) * xt
        self.Vt += np.outer(xt,xt)

        tempval1 = np.matmul(self.invVt, xt.T)  # d by 1, O(d^2)
        tempval2 = np.matmul(xt,tempval1)  # scalar, O(d)
        self.logdetV += np.log(1 + tempval2)

        evalues, evectors = np.linalg.eig(self.Vt)
        # Ensuring square root matrix exists
        assert (evalues >= 0).all()
        self.invVt = evectors * np.reciprocal(evalues) @ np.linalg.inv(evectors)
        #self.invVt_sqrt = linalg.sqrtm(self.invVt )
        self.invVt_sqrt = evectors * np.sqrt(np.reciprocal(evalues)) @ np.linalg.inv(evectors)
        
        #self.invVt = np.linalg.inv(self.Vt)
        self.theta_hat = np.matmul(self.invVt, self.XTy.T)



        if (self.flags["type"] == "EOPT") :
            self.beta_t = calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)
        elif(self.flags["type"] == "Sphere") :
            self.beta_t = calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)
        else:
             raise NotImplementedError() # todo: use valid_idx
    

        

        self.t = self.t  + 1