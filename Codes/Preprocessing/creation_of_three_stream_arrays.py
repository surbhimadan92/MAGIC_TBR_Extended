import numpy as np
import pandas as pd
import os, glob
from tqdm import tqdm
import math


train_samples = list(pd.read_csv("/home/lasii/TAPNET/train_samples_updated.csv")["sample_id"]) #path for train labels
val_samples = list(pd.read_csv("/home/lasii/TAPNET/val_samples_updated.csv")["sample_id"]) #path for validation labels

tracking_all_files = glob.glob("/home/lasii/TAPNET/tracking/*") #Tracking arrays created for each video


indexes = [22,20,18,16,14,12,11,13,15,17,19,21,24,26,28,23,25,27,8,6,5,4,0,1,2,3,7,9,10]



#Creation of C Array
final_list = []
for t in tqdm(val_samples):
    file_path = "/home/lasii/TAPNET/tracking/" + f"{t:05d}" + "_cor.npy"
    arr = np.load(file_path).transpose(1,0,2)
    main_li = []
    for k in range(64):
        lis = []
        for val in indexes:
            vv = arr[k,val,:]
            lis.append(vv[0])
            lis.append(vv[1])
        main_li.append(lis)
    final_list.append(main_li)
    
train_array = np.array(final_list)




#Creation of T Array

final_list = []
for t in tqdm(val_samples):
    file_path = "/home/lasii/TAPNET/tracking/" + f"{t:05d}" + "_cor.npy"
    arr = np.load(file_path).transpose(1,0,2)
    main_li = []
    for i in range(1,64):
        lis = []
        for val in indexes:
            vv1 = list(arr[i,val,:])
            vv2 = list(arr[i-1,val,:])
            lis.append(math.dist(vv1, vv2))
        main_li.append(lis)
    final_list.append(main_li)
    
train_array = np.array(final_list)




#Creation of S Array
indexes = [22,20,18,16,14,12,11,13,15,17,19,21,26,28,25,27,8,6,5,4,0,1,2,3,7,9,10]

final_list = []
for t in tqdm(train_samples):
    file_path = "/home/lasii/TAPNET/tracking/" + f"{t:05d}" + "_cor.npy"
    arr = np.load(file_path).transpose(1,0,2)
    main_li = []
    for i in range(64):
        lis = []
        reference_point = list((arr[i,23,:]+arr[i,24,:])/2)
        for val in indexes:
            current_point = list(arr[i,val,:])
            lis.append(math.dist(reference_point, current_point))
        main_li.append(lis)
    final_list.append(main_li)
    
train_array = np.array(final_list)














