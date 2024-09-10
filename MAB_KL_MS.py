from optimaldesign import *
from arms_generator import *
from Bandit_Env import *
from scipy import linalg


class MAB_KL_MS(Bandit):
    ########################################
    def __init__(self,K,X,n_mc_samples):
        self.K = K
        self.t = 1
        self.X = X
        self.S = np.zeros((self.K,1))
        self.F = np.zeros((self.K, 1))
        self.estimated_means = np.zeros((self.K, 1))
        self.prob_vals = np.zeros((self.K, 1))
        self.pulled_freqs = np.zeros((self.K, 1))
        self.chosen = 0
        self.prob_chosen = 0
        self.n_mc_samples = n_mc_samples

    def kl_div_est(self,mu_1,mu_2):
        if (mu_2 == 0 or mu_2 == 1 or mu_1 ==1 or mu_1 == 0):
            return 0
        #print(mu_2)
        Kl_div = mu_1*np.log((mu_1)/(mu_2)) + (1-mu_1)* np.log((1-mu_1)/ (1-mu_2) )
        #print(Kl_div)
        return Kl_div

    def binomial_KL_divergence(self,a , b):
        if b == 1:
            b = 1 - np.finfo(float).eps
        if b == 0:
            b = np.finfo(float).eps
        if a == 1:
            a = 1 - np.finfo(float).eps
        if a == 0:
            a = np.finfo(float).eps

        kl = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
        return kl

    def next_arm(self):
        if(self.t <= self.K):
            self.chosen =self.t - 1
            #self.pulled_freqs[self.t][0] = self.pulled_freqs[self.t][0] + 1
            #print(self.chosen)
            #print(self.estimated_means[self.chosen][0])
            return self.chosen

        for i in range(self.K):
            mu_max = np.max(self.estimated_means)
            Kl_div = self.binomial_KL_divergence(self.estimated_means[i,0],mu_max)
            self.prob_vals[i,0] = np.exp(-self.pulled_freqs[i,0]*Kl_div)

        self.prob_vals = self.prob_vals/np.sum(self.prob_vals)
        ind = np.random.choice(self.K, 1, p=self.prob_vals .ravel())

        self.chosen = ind
        #print(self.chosen)
        #print(self.estimated_means[self.chosen][0])
        return self.chosen


    def get_probability_arm(self):
        if(self.t <= self.K):
            return 1/self.K
        else:
            self.prob_chosen = self.prob_vals[self.chosen,0]
            return self.prob_chosen

    def update(self, pulled_idx, y_t):
        self.pulled_freqs[pulled_idx,0] = self.pulled_freqs[pulled_idx,0] + 1
        #print(self.pulled_freqs[pulled_idx,0])
        self.estimated_means[pulled_idx,0] = ((self.estimated_means[pulled_idx,0])*(self.pulled_freqs[pulled_idx,0] - 1) + y_t)/(self.pulled_freqs[pulled_idx,0] )
        #print(self.estimated_means[pulled_idx][0] )
        self.t = self.t + 1 