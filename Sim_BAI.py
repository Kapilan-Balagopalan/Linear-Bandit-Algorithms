import numpy as np
import matplotlib.pyplot as plt

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

for k in range(big_trials):
    no_non_stops = 0
    no_non_stops2 = 0
    for i in range(n_trials):
        switch = True
        Arms = [0, 1, 2]
        Means = [mu_best, mu_sub1, mu_sub2]
        t = 1
        samples = np.zeros(3)
        for j in range(3):
            samples[j] = np.random.normal(Means[j], variance, 1)
        t = t + 1
        while (len(Arms) > 1):
            for j in Arms:
                samples[j] = (samples[j] * (t - 1) + np.random.normal(Means[j], variance, 1)) / t
            for j in Arms:
                # print(np.max(samples) - samples[j])
                # print( np.sqrt(2* np.log(3.3*(t**2)/delta)/t))
                # print("end")
                if (np.max(samples) - samples[j] >= np.sqrt(2 * np.log(3.3 * (t ** 2) / delta) / t)):
                    # Means.remove(Means[j])
                    samples[j] = float('-inf')
                    Arms.remove(j)
                    break
            t = t + 1
            if (t > 5000 and switch == True):
                no_non_stops2 = no_non_stops2 + 1
                switch = False
            if (t > 30000):
                no_non_stops = no_non_stops + 1
                break
        #print(t-1)
        histogram_vector[i] = t-1
    cum_non_stops = cum_non_stops +  no_non_stops
    cum_non_stops2 = cum_non_stops2 + no_non_stops2


#print(histogram_vector)
plt.hist(histogram_vector, bins=100, color='skyblue', alpha=0.5, edgecolor='black', label='Stopping times')

plt.xlabel('Stopping time')
plt.ylabel('Number of Trials')
plt.title('Histogram of stopping times')
#plt.legend()
plt.savefig('Img-2.png', format='png')
plt.show()
print(cum_non_stops/big_trials)
print(cum_non_stops2/big_trials)



