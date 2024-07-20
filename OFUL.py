from Bandit_Env import *

class Oful(Bandit):
########################################
    def __init__(self, X, R, S,N, flags):
        self.X = X
        self.R = R
        self.S = S
        self.flags = flags
        if(self.flags["type"] == "EOPT"):
            self.lam = self.R**2/self.S**2
        elif(self.flags["type"] == "Sphere"):
            self.lam = self.R**2/self.S**2
        self.delta = .01
        # more instance variables
        self.t = 1
        self.K, self.d = self.X.shape


        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.lam
        self.X_invVt_norm_sq = np.sum(self.X * self.X, axis=1) / self.lam
        self.logdetV = self.d*np.log(self.lam)
        self.theta_hat = np.zeros(self.d)
        self.Vt = self.lam * np.eye(self.d)

        if (self.flags["type"] == "EOPT") :
            self.beta_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)
        elif(self.flags["type"] == "Sphere") :
            self.beta_t = calc_sqrt_beta_det2_initial( self.R, self.lam, self.delta, self.S)
        else:
             raise NotImplementedError() # todo: use valid_idx

    def next_arm(self):
        if (self.t == 1):
            return np.random.randint(self.K) 
        

        radius_sq = self.beta_t
        #obj_func = np.dot(self.X, self.theta_hat) + np.sqrt(radius_sq) * np.sqrt(self.X_invVt_norm_sq)
        obj_func = np.zeros(self.K)
        for i in range(self.K):
            obj_func[i] = np.dot(self.X[i], self.theta_hat) + np.sqrt(radius_sq) * np.sqrt(
            np.matmul(np.matmul(self.X[i].T, self.invVt), self.X[i]))
        chosen = np.argmax(obj_func)
        return chosen

    def update(self, pulled_idx, y_t):

        ##########################
        ## DEBUGGING
        ##########################
        # y_t = 2*y_t - 1;
        xt = self.X[pulled_idx, :]

        self.XTy += (y_t) * xt
        self.Vt += np.outer(xt,xt)

        tempval1 = np.dot(self.invVt, xt)  # d by 1, O(d^2)
        tempval2 = np.dot(tempval1, xt)  # scalar, O(d)
        self.logdetV += np.log(1 + tempval2)

        self.invVt = find_matrix_inverse_vt_method_fast(self.invVt, xt)

        #self.invVt  = find_matrix_inverse_vt_method_conventional(self.Vt)

        

        #self.invVt = np.linalg.inv(self.Vt)
        self.theta_hat = np.dot(self.invVt, self.XTy)

        if (self.flags["type"] == "EOPT") :
            self.beta_t = calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)
        elif(self.flags["type"] == "Sphere") :
            self.beta_t = calc_sqrt_beta_det2(self.d,self.R, self.lam, self.delta, self.S,self.logdetV)
        else:
             raise NotImplementedError() # todo: use valid_idx
    

        

        self.t = self.t  + 1




