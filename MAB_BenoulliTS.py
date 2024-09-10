from optimaldesign import *
from arms_generator import *
from Bandit_Env import *
from scipy import linalg
from line_profiler import profile

class MAB_TS_Bernoulli(Bandit):
    ########################################
    def __init__(self,K,X,n_mc_samples):
        self.K = K
        self.t = 1
        self.X = X
        self.S = np.ones((self.K,1))*0.5
        self.F = np.ones((self.K, 1))*0.5
        #print(self.S )
        self.reward_samples = np.zeros((self.K, 1))
        self.chosen = 0
        self.prob_chosen = 0
        self.n_mc_samples = n_mc_samples

    def next_arm(self):
        for i in range(self.K):
            self.reward_samples[i,0] = np.random.beta(self.S[i,0], self.F[i,0], size=None)

        self.chosen = int(np.argmax(self.reward_samples))

        return self.chosen

    @profile
    def get_probability_arm(self):
        temp_reward_samples = np.zeros((self.K, 1))
        count = 0
        if(self.t==1):
            return 1/self.K
        for i in range(self.n_mc_samples):
            for j in range(self.K):
                temp_reward_samples[j,0] = np.random.beta(self.S[j,0], self.F[j,0], size=None)
            temp_chosen = int(np.argmax(temp_reward_samples))
            if (temp_chosen == self.chosen):
                count = count + 1
        return count/self.n_mc_samples

    def update(self, pulled_idx, y_t):
        if (y_t == 1):
            self.S[pulled_idx,0] = self.S[pulled_idx,0] + 1
        else:
            self.F[pulled_idx,0] = self.F[pulled_idx,0] + 1
        self.t = self.t + 1
