import numpy as np
from dataset import EEGDataset
def DA4(EEGdata):
    # 0-4s，0.2-4.2s，0.4,-4.4s，0.6-4.6s，0.8-4.8s，1-5s，
    # 1.2-5.2s，1.4-5.4s，1.6-5.6s，1.8-5.8s，2-6s
    label = EEGdata.label
    data = EEGdata.data
    print("Before data augmentation:", data.shape)
    # print(data.shape)
    length = int(len(data)/4)+10
    a,b,c,d = 0,0,0,0
    left,right,foot,tongue = np.zeros((length,22,1500)),np.zeros((length,22,1500)),np.zeros((length,22,1500)),np.zeros((length,22,1500))
    for i in range(len(label)):
        if label[i]==0:
            left[a] = data[i]
            a=a+1
        elif label[i]==1:
            right[b] = data[i]
            b+=1
        elif label[i]==2:
            foot[c] = data[i]
            c+=1
        elif label[i]==3:
            tongue[d] = data[i]
            d+=1
    left = left[0:a]
    right = right[0:b]
    foot = foot[0:c]
    tongue = tongue[0:d]
    print('size of 4 classes:',len(left),len(right),len(foot),len(tongue))
    newleft = np.zeros((len(left)*11,22,1000))
    newright = np.zeros((len(right) *11, 22, 1000))
    newfoot = np.zeros((len(foot)*11, 22, 1000))
    newtongue= np.zeros((len(tongue)*11, 22, 1000))
    a2,b2,c2,d2 = 0,0,0,0
    for i in range(0,len(left)):
        if i+1<len(left):
            for j in range(11):
                # print(int(j*0.2*250),int((4+j*0.2)*250))
                newleft[a2+j] = left[i,:,int(j*0.2*250):int((4+j*0.2)*250)]
            a2+=11
    # print(len(right))
    for i in range(0,len(right)):
        if i+1<len(right):
            # print('i',i)
            for j in range(11):
                newright[b2 + j] = right[i,:,int(j*0.2*250):int((4+j*0.2)*250)]
            b2+=11
    for i in range(0,len(foot)):
        if i+1<len(foot):
            for j in range(11):
                newfoot[c2 + j] = foot[i,:,int(j*0.2*250):int((4+j*0.2)*250)]
            c2+=11
    for i in range(0,len(tongue)):
        if i+1<len(tongue):
            for j in range(11):
                newtongue[d2 + j] = tongue[i,:,int(j*0.2*250):int((4+j*0.2)*250)]
            d2+=11
    left = newleft
    right = newright
    foot = newfoot
    tongue = newtongue
    label = np.zeros((len(left) + len(right) + len(foot) + len(tongue), 1))
    for i in range(len(label)):
        if i < len(left):
            label[i] = 0
        elif i >= len(left) and i < len(left) + len(right):
            label[i] = 1
        elif i >= len(left) + len(right) and i < len(left) + len(right) + len(tongue):
            label[i] = 2
        else:
            label[i] = 3
    data = np.vstack((left,right,foot,tongue))
    print("After data augmentation:",data.shape)
    newEEGdata = EEGDataset(label,data)
    return newEEGdata