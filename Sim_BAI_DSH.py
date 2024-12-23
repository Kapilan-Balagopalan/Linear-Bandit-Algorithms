import numpy as np
import matplotlib.pyplot as plt
import math
no_non_stops = 0
cum_non_stops = 0

no_non_stops2 = 0
cum_non_stops2 = 0

variance = 1
mu_best = 1
mu_sub1 =0.900
mu_sub2 = mu_sub1
n_trials = 1000
delta = 0.05

histogram_vector = np.zeros(n_trials)

#np.random.seed(100)
big_trials = 1
K = 3
Means = [mu_best, mu_sub1, mu_sub2]
eps = 0.0

for i in range(n_trials):
    T1 = int(K * math.log(K,2))
    L = int(math.ceil(math.log(K,2)))
    t = 1
    samples = np.zeros(K)
    pull_count = np.zeros(K)
    m = 0
    Arms = np.arange(K)
    b_arr = np.zeros(K)
    while(True):
        samples = np.zeros(K)
        pull_count = np.zeros(K)
        Arms = np.arange(K)
        b_arr = np.zeros(K)
        m = m + 1
        T_m = T1*math.pow(2,m-1)
        #print(T_m)
        #print(L)
        Elim_arm_count = K
        Elim_arms = np.zeros(Elim_arm_count)
        Elim_arms[:] = Arms[:]
        Emp_means = [0,0,0]
        for l in range(L):
            N_m_l = int(math.floor(T_m/(K*math.pow(2,-l -1 )*math.ceil(math.log(K,2) ))))
            #print(N_m_l)
            for p in range(K):
                for j in range(N_m_l):
                    if (Emp_means[p] != float('-inf')):
                        samples[p] = samples[p]  +  np.random.normal(Means[p], variance, 1)
                        pull_count[p] = pull_count[p] + 1
                if (Emp_means[p] !=  float('-inf')):
                    Emp_means[p] =  samples[p]/pull_count[p]
            Elim_arm_count = int(math.ceil(Elim_arm_count/2))
            Elim_arms = np.zeros(Elim_arm_count)
            for j in range(Elim_arm_count):
                Elim_arms[j] = np.argmax(Emp_means)
                Emp_means[np.argmax(Emp_means)] = float('-inf')

        for p in range(K):
            b_arr[p] = np.sqrt((2/pull_count[p])*np.log((6*K*math.log(K,2)*(m**2))/delta ))/1000
        count = 0
        for p in range(K):
            if(p != int(Elim_arms[0])):
                if(samples[int(Elim_arms[0])]/pull_count[int(Elim_arms[0])] - b_arr[int(Elim_arms[0])] >  samples[p]/pull_count[p] + b_arr[p] - eps):
                    count = count + 1
                else:
                    continue
        #print(count)
        if(count == K-1):
            print(m)
            break





#print(histogram_vector)
plt.hist(histogram_vector, bins=100, color='skyblue', alpha=0.5, edgecolor='black', label='Stopping times')

plt.xlabel('Stopping time' ,fontsize=20)
plt.ylabel('Number of Trials',  fontsize=20)
#plt.title('Histogram of stopping times')
#plt.legend()
plt.savefig('Img-2.png', format='png')
plt.savefig('Img-2.pdf', format='pdf')
plt.show()
print(cum_non_stops/big_trials)
print(cum_non_stops2/big_trials)






