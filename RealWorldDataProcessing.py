import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import numpy.random as ra
import numpy.linalg as la

# Reading from u .data file
file_path = 'C:/Users/Kapilan/OneDrive - University of Arizona/Academia_Kapilan/Research/Source_code/RealWorldDataSet/ml-100k/ml-100k/u.data'

with open(file_path, 'r') as file:
    Lines = file.readlines()




n_users = 943
n_movies = 1682

user_movie_rating = np.zeros((n_users, n_movies))

count = 0
# Strips the newline character
for line in Lines:
    count += 1
    line_array = line.split()
    #print(line_array[2])
    user_movie_rating[int(line_array[0])-1,int(line_array[1])-1] = int(int(line_array[2]))
    #print("Line{}: {}".format(count, line.strip()))

# Reading from u.item  file
file_path = 'C:/Users/Kapilan/OneDrive - University of Arizona/Academia_Kapilan/Research/Source_code/RealWorldDataSet/ml-100k/ml-100k/u.item'

with open(file_path, 'r') as file:
    Lines = file.readlines()




n_movie_features = 19

movie_features = np.zeros((n_movies,n_movie_features))

count = 0
# Strips the newline character
for line in Lines:
    count += 1
    #print(line)
    line_array = line.split("|")
    for i in range(n_movie_features):
        movie_features[int(line_array[0])-1, i ] = line_array[len(line_array) - i - 1]


# Reading from u.occupation  file
file_path = 'C:/Users/Kapilan/OneDrive - University of Arizona/Academia_Kapilan/Research/Source_code/RealWorldDataSet/ml-100k/ml-100k/u.occupation'

with open(file_path, 'r') as file:
    Lines = file.readlines()


dict_occupation = {}
count = 0
# Strips the newline character
for line in Lines:

    #print(line)
    line_array = line.split()
    #line.replace("\n", "")
    #line.strip()
    dict_occupation[line_array[0]] = count
    count += 1


dict_gender = {"M" : 1, "F" : 2}



# Reading from u.user  file
file_path = 'C:/Users/Kapilan/OneDrive - University of Arizona/Academia_Kapilan/Research/Source_code/RealWorldDataSet/ml-100k/ml-100k/u.user'

with open(file_path, 'r') as file:
    Lines = file.readlines()

count = 0
# Strips the newline character
n_user_features = 4

user_features = np.zeros((n_users,n_user_features))

print(dict_occupation)
for line in Lines:
    count += 1
    print(line)
    line_array = line.split("|")
    user_features[int(line_array[0])-1,0]    = int(line_array[1])
    user_features[int(line_array[0]) - 1, 1] = dict_gender[line_array[2]]
    user_features[int(line_array[0]) - 1, 2] = dict_occupation[line_array[3]]
    temp = [ord(c) for c in line_array[4]]
    user_features[int(line_array[0]) - 1, 3] = sum(temp)

print(user_features)







