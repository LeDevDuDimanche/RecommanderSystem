# Useful starting lines
import numpy as np
import scipy
import scipy.io

import matplotlib.pyplot as plt  
from helpers import load_data, preprocess_data 
from plots import plot_raw_data, plot_train_test_data
from ex10_methods import *

path_dataset = "Datasets/data_train.csv" #"movielens100k.csv"
ratings = load_data(path_dataset)


num_items_per_user, num_users_per_item = plot_raw_data(ratings)

print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))


 

valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)
plot_train_test_data(train, test)

baseline_global_mean(train, test)
baseline_user_mean(train, test)
baseline_item_mean(train, test)
print("Matrix factorization SGD ", matrix_factorization_SGD(train, test))

print("ALS ", ALS(train, test))