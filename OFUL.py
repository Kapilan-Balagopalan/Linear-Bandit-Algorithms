import numpy as np

import numpy.random as ra
import numpy.linalg as la

from Bandit_Env import *

class Oful(Bandit):
########################################
    def __init__(self, X, lam, R, S, flags, subsample_func=None, subsample_rate=1.0, multiplier=1.0):
        self.X = X
        self.R = R
        self.lam = lam
        self.delta = .01
        self.S = S
        self.flags = flags
        self.multiplier = float(multiplier)

        # more instance variables
        self.t = 1
        self.N, self.d = self.X.shape

        #- subsampling aspect
        assert subsample_func == None
        self.subN = np.round(self.N * float(subsample_rate)).astype(int)
        self.subsample_func = subsample_func

        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.lam
        self.X_invVt_norm_sq = np.sum(self.X * self.X, axis=1) / self.lam
        self.logdetV = self.d*np.log(self.lam)
        self.sqrt_beta = calc_sqrt_beta_det2(self.d,self.t,self.R,self.lam,self.delta,self.S,self.logdetV)
        self.theta_hat = np.zeros(self.d)
        self.Vt = self.lam * np.eye(self.d)

        self.do_not_ask = []
        self.dbg_dict = {'multiplier':float(multiplier),
                'subN': self.subN,
                'subsample_func': self.subsample_func}

    def next_arm(self):
        #valid_idx = np.setdiff1d(np.arange(self.N),self.do_not_ask)
        if (self.t == 1):
            return ra.randint(self.N), np.nan
        #my_c = self.dbg_dict['multiplier']
        #radius_sq = self.multiplier * (self.sqrt_beta)**2

        radius_sq = self.multiplier * self.gamma_t
        if (self.subsample_func == None):
            #obj_func = np.dot(self.X, self.theta_hat) + np.sqrt(radius_sq) * np.sqrt(self.X_invVt_norm_sq)
            obj_func = np.zeros(self.N)
            for i in range(self.N):
                obj_func[i] = np.dot(self.X[i], self.theta_hat) + np.sqrt(radius_sq) * np.sqrt(
                    np.matmul(np.matmul(self.X[i].T, self.invVt), self.X[i]))


            #chosen_inner = np.argmax(obj_func[valid_idx])
            chosen_inner = np.argmax(obj_func)
            #chosen = valid_idx[chosen_inner]
        else:
            raise NotImplementedError() # todo: use valid_idx
#             idx = self.subsample_func(self.N, self.subN);
#             subX = self.X[idx,:];
#             obj_func = np.dot(subX, self.theta_hat) + np.sqrt(radius_sq) * \
#                     np.sqrt( mahalanobis_norm_sq_batch(subX, self.invVt));
#
#             chosen = idx[np.argmax(obj_func)];
        #return chosen, radius_sq
        return chosen_inner,radius_sq

    def update(self, pulled_idx, y_t):

        ##########################
        ## DEBUGGING
        ##########################
        # y_t = 2*y_t - 1;
        xt = self.X[pulled_idx, :]

        self.XTy += (y_t) * xt
        self.Vt += np.outer(xt,xt)

        #tempval1 = np.dot(self.invVt, xt)    # d by 1, O(d^2)
        #tempval2 = np.dot(tempval1, xt)      # scalar, O(d)
        #self.logdetV += np.log(1 + tempval2)

        #if (self.t % 20 == 0 & False):
            #self.invVt = la.inv(self.Vt)
            # FIXME: would this be a concern for the simulation?
            #if (self.subsample_func == None):
    #            self.X_invVt_norm_sq = mahalanobis_norm_sq_batch(self.X, self.invVt);  # O(Nd^2)
                #self.X_invVt_norm_sq -= (np.dot(self.X, tempval1) ** 2) / (1 + tempval2) # efficient update, O(Nd)
        #else:
            #self.invVt -= np.outer(tempval1, tempval1) / (1 + tempval2)
            #if (self.subsample_func == None):
                #self.X_invVt_norm_sq -= (np.dot(self.X, tempval1) ** 2) / (1 + tempval2) # efficient update, O(Nd)
        self.invVt = np.linalg.inv(self.Vt)
        self.theta_hat = np.dot(self.invVt, self.XTy)


        #self.do_not_ask.append( pulled_idx )

        my_t = self.t + 1
        self.gamma_t = calc_gamma_t(self.t, self.d, self.lam, self.delta, self.S, self.R)
        #self.sqrt_beta = calc_sqrt_beta_det2(self.d,my_t,self.R,self.lam,self.delta,self.S,self.logdetV)

        self.t += 1

    def getDoNotAsk(self):
        return self.do_not_ask

    def predict(self, X=None):
        if X is None:
            X = self.X
        return X.dot(self.theta_hat)

    def get_debug_dict(self):
        return self.dbg_dict



