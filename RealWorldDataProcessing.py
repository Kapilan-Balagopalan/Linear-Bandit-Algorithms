import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import numpy.random as ra
import numpy.linalg as la
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Reading from u .data file

def generate_real_world_data(d,n_users_aug, n_movies_aug):

    file_path = './ml-100k/u.data'

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
        # print(line_array[2])
        user_movie_rating[int(line_array[0]) - 1, int(line_array[1]) - 1] = int(int(line_array[2]))
        # print("Line{}: {}".format(count, line.strip()))

    # Reading from u.item  file
    file_path = './ml-100k/u.item'

    with open(file_path, 'r') as file:
        Lines = file.readlines()

    n_movie_features = 19

    movie_features = np.zeros((n_movies, n_movie_features))

    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        # print(line)
        line_array = line.split("|")
        for i in range(n_movie_features):
            movie_features[int(line_array[0]) - 1, i] = line_array[len(line_array) - i - 1]

    # Reading from u.occupation  file
    file_path = './ml-100k/u.occupation'

    with open(file_path, 'r') as file:
        Lines = file.readlines()

    dict_occupation = {}
    count = 0
    # Strips the newline character
    for line in Lines:
        # print(line)
        line_array = line.split()
        # line.replace("\n", "")
        # line.strip()
        dict_occupation[line_array[0]] = count
        count += 1

    dict_gender = {"M": 1, "F": 2}

    # Reading from u.user  file
    file_path = './ml-100k/u.user'

    with open(file_path, 'r') as file:
        Lines = file.readlines()

    count = 0
    # Strips the newline character
    n_user_features = 4

    user_features = np.zeros((n_users, n_user_features))

    # print(dict_occupation)
    for line in Lines:
        count += 1
        # print(line)
        line_array = line.split("|")
        user_features[int(line_array[0]) - 1, 0] = int(line_array[1])
        user_features[int(line_array[0]) - 1, 1] = dict_gender[line_array[2]]
        user_features[int(line_array[0]) - 1, 2] = dict_occupation[line_array[3]]
        temp = [ord(c) for c in line_array[4]]
        user_features[int(line_array[0]) - 1, 3] = sum(temp)

    #print(user_features.shape)
    #print(movie_features.shape)



    chosen_users = np.random.choice(n_users,size=n_users_aug,replace=False)


    chosen_movies = np.random.choice(n_movies, size=n_movies_aug, replace=False)

    #print(len(chosen_users))
    #print(len(chosen_movies))

    chosen_user_mat = user_features[chosen_users]
    chosen_movie_mat = movie_features[chosen_movies]
    #chosen_movie_mat = chosen_movie_mat + 0.1
    scaler = StandardScaler()
    chosen_movie_mat_scaled = scaler.fit_transform(chosen_movie_mat)

    #chosen_user_mat = scaler.fit_transform(chosen_user_mat)

    # Initialize PCA and reduce to 2 components
    n_components_movies = int(np.sqrt(d))
    pca = PCA(n_components= n_components_movies)
    chosen_movie_mat_final = pca.fit_transform(chosen_movie_mat_scaled)
    #print(chosen_movie_mat_final.shape)

    scaler = StandardScaler()
    chosen_user_mat_scaled = scaler.fit_transform(chosen_user_mat)

    # chosen_user_mat = scaler.fit_transform(chosen_user_mat)

    # Initialize PCA and reduce to 2 components
    n_components_users = int(np.sqrt(d))
    pca = PCA(n_components=n_components_users)
    chosen_user_mat_final = pca.fit_transform(chosen_user_mat_scaled)

    for i in range(n_components_users):
        temp_min = np.min(chosen_user_mat_final[:,i])
        temp_max = np.max(chosen_user_mat_final[:, i])

        chosen_user_mat_final[:,i] = (chosen_user_mat_final[:,i] - temp_min)/(temp_max - temp_min)

    for i in range(n_components_movies):
        temp_min = np.min(chosen_movie_mat_final[:,i])
        temp_max = np.max(chosen_movie_mat_final[:, i])
        if (temp_max - temp_min == 0):
            continue
        else:
            chosen_movie_mat_final[:,i] = (chosen_movie_mat_final[:,i] - temp_min)/(temp_max - temp_min)

    #print(chosen_user_mat)
    #print(chosen_movie_mat_final)




    theta_true = np.random.normal(0,1,int(d))
    #print(theta_true)
    theta_true = theta_true/np.linalg.norm(theta_true)

    #as(final_arm_set.shape)
    return chosen_movie_mat_final, chosen_user_mat_final, theta_true

#generate_real_world_data(n_users_aug = 20, n_movies_aug = 20)











