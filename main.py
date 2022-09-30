import os
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model_def2 import *
import dataset
from trainANDtest import *
from dataset import *
import data_agumentation as DA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def ten_fold():
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    np.random.seed(1234)
    batch_size = 64
    n_epoch = 2

    # for patient in np.unique(Patient_id):
    result_sum = np.zeros((9, 5))
    for i in range(1, 10):
        if i>7:
            batch_size = 68
        if i == 88:
            Result = []
            res = [0, 0, 0, 0]
            Result.append(res)
            Result = np.mean(Result, axis=0)
            result_sum[i] = Result
            continue
        else:
            Result = np.zeros((5))
            print('\nBegin Training for Subject_' + str(i))
            EEG = dataset.Subject9_i_10fold(i)

            print(EEG.label.shape)
            print(EEG.data.shape)
            index = index_of_ten_folds(len(EEG.label))
            for j in range(10):
                number = 0
                if j != 0:
                    for m in range(j):
                        number = number + index[m]
                    number = int(number)
                    Train_data = np.vstack((EEG.data[:number], EEG.data[number+int(index[j]):]))
                    Train_label = np.vstack((EEG.label[:number], EEG.label[number+int(index[j]):]))
                    Test_data = EEG.data[number:number+int(index[j])]
                    Test_label = EEG.label[number:number+int(index[j])]
                else:
                    Train_data = EEG.data[int(index[0]):]
                    Train_label = EEG.label[int(index[0]):]
                    Test_data = EEG.data[:int(index[0])]
                    Test_label = EEG.label[:int(index[0])]
                print('Fold_' + str(j + 1))

                Train = EEGDataset(label=Train_label, data=Train_data)
                Test = EEGDataset(label=Test_label, data=Test_data)
                Train = DA.DA4(Train)
                Trainloader = DataLoader(Train, batch_size=batch_size, shuffle=True,pin_memory=True)
                Testloader = DataLoader(Test, batch_size=1, shuffle=True,pin_memory=True)
                res = TrainTest_Model(ArcCNN, Trainloader, Testloader, opti='Adam', n_epoch=n_epoch, learning_rate=0.001,
                                      print_epoch=-1, name=i)
                Result = Result + res
            Result = Result / 10
            print('-' * 100)
            print('End Training with \t loss:%.4f\tAccuracy : %0.4f\t\tval-loss:%.4f\tval-Accuracy : %.4f\tval-Kappa : %.4f' %
                  (Result[0], Result[1], Result[2], Result[3],Result[4]))

            print('\n' + '-' * 100)
            result_sum[i - 1] = Result

    for k in range(len(result_sum)):
        print('Subject_', str(k + 1))
        print('\t\tloss:%.4f\tAccuracy : %0.4f\t\tval-loss:%.4f\tval-Accuracy : %.4f' %
              (result_sum[k, 0], result_sum[k, 1], result_sum[k, 2], result_sum[k, 3]))
    with open('result.txt', 'w') as outfile:
        for slice_2d in result_sum:
            np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')

ten_fold()