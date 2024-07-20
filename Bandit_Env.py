import numpy as np
import ipdb


def calc_gamma_LinZHU(N,d, delta):
    c_OPT = 4
    gamma = np.sqrt((c_OPT*d*N)/(d*np.log(N) + 32*np.log(2/delta)))
    return gamma

def calc_eta_LinZHU(gamma,d):
    c_OPT = 4
    eta = gamma / (c_OPT*d)
    return eta

def sample_action(A,MED_prob_dist):
    K,d = A.shape
    ind = np.random.choice(K, 1, p= MED_prob_dist)
    return A[ind,:], ind

def calc_eta_t_EXP2(t,d,K):
        return np.sqrt(np.log(K)/(3*d*t))

def calc_sqrt_beta_det2(d,t,R,ridge,delta,S,logdetV):
  return R * np.sqrt( logdetV - d*np.log(ridge) + np.log (1/(delta**2)) ) + np.sqrt(ridge) * S

def find_matrix_inverse_vt_method_conventional(Vt):
    invVt = np.linalg.inv(Vt)
    return invVt
def find_matrix_inverse_vt_method_fast(invVt, xt):
    tempval1 = np.dot(invVt, xt)    
    tempval2 = np.dot(tempval1, xt)   
    invVt -= np.outer(tempval1, tempval1) / (1 + tempval2) 
    return invVt


def calc_expected_regret(best_arm,theta_true, MED_prob_dist,A):
    mVal_reward= np.matmul(A, theta_true)
    mVal_exp_reward = np.dot(MED_prob_dist,mVal_reward)
    inst_regret = np.max(np.matmul(A, theta_true)) - np.sum(mVal_exp_reward)
    #print( np.argmax(np.matmul(A, theta_true)))
    #print("This is expected reward",np.sum(mVal_exp_reward))
    #print(inst_regret)
    return inst_regret

def calc_regret(chosen_arm, theta_true, A):
    mVal_reward = np.matmul(A, theta_true)
    inst_regret = np.max(mVal_reward) - mVal_reward[chosen_arm]
    # print( np.argmax(np.matmul(A, theta_true)))
    # print("This is expected reward",np.sum(mVal_exp_reward))
    # print(inst_regret)
    return inst_regret

def calc_gamma_t_SGMED(t,d,sVal_lambda,delta,S,noise_sigma):
    gamma_t = (noise_sigma*np.sqrt(d*np.log(1 + t/(sVal_lambda*d)) + 2*np.log(2/delta)) + np.sqrt(sVal_lambda)*S)**2
    return gamma_t

def calc_gamma_t_EXP2(t,d,sVal_lambda,delta,S,noise_sigma):
    gamma_t = (noise_sigma*np.sqrt(d*np.log(1 + t/(sVal_lambda*d)) + 2*np.log(2/delta)) + np.sqrt(sVal_lambda)*S)**2
    return gamma_t

def calc_beta_t_OFUL(t,d,sVal_lambda,delta,S,noise_sigma):
    beta_t = (noise_sigma*np.sqrt(d*np.log(1 + t/(sVal_lambda*d)) + 2*np.log(1/delta)) + np.sqrt(sVal_lambda)*S)**2
    return beta_t

def calc_beta_t_LinIMED(t,d,sVal_lambda,delta,S,noise_sigma):
    beta_t = (noise_sigma*np.sqrt(3*d*np.log(1 + t) ) + np.sqrt(2))**2
    return beta_t

def receive_reward(chosen,theta_true, noise_sigma, A):
    #print(Arm_t.shape)
    #print(theta_true.shape)
    noise = np.random.normal(0,noise_sigma, 1)
    #print(noise)
    reward = np.dot(A[chosen,:],theta_true)
    #print(reward)
    final_reward = reward + noise
    #print(final_reward)
    return final_reward


class Bandit(object):
    def __init__(self, X, theta):
        raise NotImplementedError()

    def next_arm(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def get_debug_dict(self):
        raise NotImplementedError()
