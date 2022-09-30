import mne
import os
import scipy.io as scio
import numpy as np
def Normalize(raw_list):
    np_data = np.array(raw_list)
    length = len(raw_list)
    means = np.mean(np_data, axis=0)
    sigmas = np.std(np_data, axis=0)
    mean_matrix = np.tile(means, (length, 1))
    sigma_metrix = np.tile(sigmas, (length, 1))
    np_data = (np_data - mean_matrix) / sigma_metrix
    return np_data
def readGDF2(fileAddr):
    raw = mne.io.read_raw_gdf(fileAddr)
    raw.load_data()
    raw.filter(8, 30, fir_design='firwin')
    mark = raw.annotations.description
    markTime = raw.annotations.onset
    leftnum = 0
    rightnum = 0
    footnum = 0
    tonguenum = 0
    dataSet_left = np.zeros((1, 22, 1500))
    dataSet_right = np.zeros((1, 22, 1500))
    dataSet_foot = np.zeros((1, 22, 1500))
    dataSet_tongue = np.zeros((1, 22, 1500))
    for i in range(len(mark)):
        if i == len(mark)-1:
            if mark[i] == '769':
                data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
                dataSet_left = np.vstack((dataSet_left, Normalize(data[0:22, :]).reshape((1, 22, 1500))))
                leftnum += 1
            elif mark[i] == '770':
                data, times = raw[:, int(markTime[i]) * 250:(int(markTime[i]) + 6) * 250]
                dataSet_right = np.vstack((dataSet_right, Normalize(data[0:22, :]).reshape((1, 22, 1500))))
                print(dataSet_right.shape)
                rightnum += 1
            elif mark[i] == '771':
                data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
                if data.shape[1] == 1500:
                    dataSet_foot = np.vstack((dataSet_foot, Normalize(data[0:22, :]).reshape((1, 22, 1500))))
                else:
                    data_1 = data[0:22, :]
                    data_1 = np.hstack((data_1, data[0:22, -(1500 - data.shape[1]):]))
                    data_1 = Normalize(data_1).reshape((1, 22, 1500))
                    dataSet_foot = np.vstack((dataSet_foot, data_1))
                footnum += 1
            elif mark[i] == '772':
                data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
                dataSet_tongue = np.vstack((dataSet_tongue, Normalize(data[0:22, :]).reshape((1, 22, 1500))))
                tonguenum += 1
        else:
            if mark[i] == '769' and mark[i-1]!='1023':
                data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
                dataSet_left = np.vstack((dataSet_left,Normalize(data[0:22, :]).reshape((1,22,1500))))
                leftnum += 1
            elif mark[i] == '770' and mark[i-1]!='1023':
                data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
                dataSet_right = np.vstack((dataSet_right,Normalize(data[0:22, :]).reshape((1,22,1500))))
                rightnum += 1
            elif mark[i] == '771' and mark[i-1]!='1023':
                data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
                if data.shape[1] == 1500:
                    dataSet_foot = np.vstack((dataSet_foot,Normalize(data[0:22, :]).reshape((1,22,1500))))
                else:
                    data_1 = data[0:22, :]
                    data_1 = np.hstack((data_1,data[0:22, -(1500 - data.shape[1]):]))
                    data_1 = Normalize(data_1).reshape((1,22,1500))
                    dataSet_foot = np.vstack((dataSet_foot,data_1))
                footnum += 1
            elif mark[i] == '772' and mark[i-1]!='1023':
                data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
                dataSet_tongue = np.vstack((dataSet_tongue,Normalize(data[0:22, :]).reshape((1,22,1500))))
                tonguenum += 1
    return dataSet_left[1:], dataSet_right[1:], dataSet_foot[1:], dataSet_tongue[1:]
def readRaw_test2(fileAddr):
    raw = mne.io.read_raw_gdf(fileAddr)
    raw.load_data()
    raw.filter(8, 30, fir_design='firwin')
    mark = raw.annotations.description
    markTime = raw.annotations.onset
    trial_num = 0
    test_data = np.zeros((288, 22, 1500))
    for i in range(len(mark)):
        if mark[i] == '783':
            data, times = raw[:, int(markTime[i] ) * 250:(int(markTime[i]) + 6) * 250]
            test_data[trial_num, :, :] = Normalize(data[0:22, :])
            trial_num += 1
    return test_data
def readONE_train(i):
    path  = 'data2'
    fileAddr = path + '/A0' + str(i) + 'T.gdf'
    left, right, foot, tongue = readGDF2(fileAddr)
    label = np.zeros(((len(left) + len(right) + len(foot) + len(tongue)),1))
    for i in range(len(left)+len(right)+len(foot)+len(tongue)):
        if i < len(left):
            label[i] = 0
        elif i >= len(left) and i < len(left)+len(right):
            label[i] = 1
        elif i >= len(left)+len(right) and i <  len(left)+len(right)+len(foot):
            label[i] = 2
        elif i >= len(left)+len(right)+len(foot):
            label[i] = 3
    train_data = np.vstack((left,right,foot,tongue))
    return train_data,label
def readONE_test(i):
    path  = 'data2'
    fileAddr = path + '/A0' + str(i) + 'E.gdf'
    test_data = readRaw_test2(fileAddr)
    return test_data
def read_trainANDtest(i):
    train_data,train_label = readONE_train(i)
    test_date = readONE_test(i)
    fileAddr = 'data2' + '/A0' + str(i) + 'E.mat'
    mat = scio.loadmat(fileAddr)
    test_label = mat['classlabel'] - 1
    return train_data,train_label,test_date,test_label