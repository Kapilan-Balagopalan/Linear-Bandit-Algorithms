from optimaldesign import *
from arms_generator import *
import numpy as np
import matplotlib.pyplot as plt

from BanditFactory import *
import ipdb

from datetime import datetime

import os

from tqdm import tqdm

n_algo = 3
algo_list = [None] * n_algo
algo_names = ["OFUL","Lin-TS-Freq","LinMED"]
algo_names_plot = ["OFUL","Lin-TS-Freq","LinMED"]
test_type = "Sphere"
emp_coeff = [0.99, 0.9, 0.5]
opt_coeff = [0.005, 0.05, 0.25]

n_cpu = 10
n_trials = 100
n = 5000

algo_color = ["r","b","g", "c","m","k", "gray", "black", "orange"]

filename1 = '333.npy'
filename2 = '444.npy'

current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/'
completeName1 = os.path.join(prefix , filename1)
completeName2 = os.path.join(prefix , filename2)

with open(completeName1, 'rb') as f:

    cum_regret_arr1 = np.load(f)

with open(completeName2, 'rb') as f:
    cum_regret_arr2 = np.load(f)

t_alpha = 1
cum_regret_mean1 = np.sum(cum_regret_arr1, axis=0)/n_trials
cum_regret_mean_std1 = np.std(cum_regret_arr1, axis=0, ddof=1)



cum_regret_confidence_up1 = cum_regret_mean1 + (t_alpha * cum_regret_mean_std1)/np.sqrt(n_trials)
cum_confidence_down1 = cum_regret_mean1 - (t_alpha * cum_regret_mean_std1)/np.sqrt(n_trials)

cum_regret_mean2 = np.sum(cum_regret_arr2, axis=0)/n_trials
cum_regret_mean_std2 = np.std(cum_regret_arr2, axis=0, ddof=1)



cum_regret_confidence_up2 = cum_regret_mean2 + (t_alpha * cum_regret_mean_std2)/np.sqrt(n_trials)
cum_confidence_down2 = cum_regret_mean2 - (t_alpha * cum_regret_mean_std2)/np.sqrt(n_trials)

i = 0
for name in algo_names_plot:
    plt.plot(np.arange(n), np.log(cum_regret_mean1[:, i]), color=algo_color[i], label=algo_names_plot[i], linewidth=3)
    plt.plot(np.arange(n), np.log(cum_regret_mean2[:, i]),linestyle='dashed', color=algo_color[i], linewidth=3)

    #plt.fill_between(np.arange(n), cum_regret_confidence_up1[:, i], cum_confidence_down1[:, i], color=algo_color[i],
                     #alpha=.3)
    #plt.fill_between(np.arange(n), cum_regret_confidence_up2[:, i], cum_confidence_down2[:, i], color=algo_color[i],
                     #alpha=.3)
    i = i + 1

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time",fontsize = 15)
plt.ylabel("Regret",fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
#plt.title("Delayed rewards experiment")
legend_handles = [
    plt.Line2D([0], [0], color=algo_color[0], label='OFUL', linewidth=3),
    plt.Line2D([0], [0], color=algo_color[1], label='LinTS', linewidth=3),
    plt.Line2D([0], [0], color=algo_color[2], label='LinMED', linewidth=3),
    plt.Line2D([0], [0], color='black', linestyle='-', label='Delay = 0',linewidth=3 ),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Delay = 10', linewidth=3)
]
#plt.legend()
plt.legend(handles=legend_handles,fontsize = 15)
plt.savefig(prefix + 'PDE11.eps', format='eps')
plt.savefig(prefix + 'PDE11.png', format='png')
plt.savefig(prefix + 'PDE11.pdf', format='pdf')
plt.show()