import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model_def2 import ArcFaceNet,CenterLossNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plot
from sklearn.metrics import confusion_matrix
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def Test_Model(net,Testloader,criterion,arcface_net,centerloss_net,alpha,is_cuda=True):
    model = net.to(device)
    # model.forward()
    model.eval()
    running_loss = 0.0
    evaluation = 0.0
    time = 0
    truelabel = [-1]*100
    predictedlabel = [-1]*100


    for i,data in enumerate(Testloader,0):
        input_img,labels = data
        input_imgs = input_img.reshape(len(input_img),1,22, 1500)
        new_imgs = np.zeros((11,1,22,1000))
        num = [0, 0, 0, 0]
        for j in range(11):
            new_imgs[j] = input_imgs[:,:,:,int(j*0.2*250):int((4+j*0.2)*250)]
        for k in range(11):
            input_imgs = new_imgs[k].reshape(1,1,22, 1000)
            # 分割线
            input_imgs = torch.as_tensor(input_imgs)
            input_imgs = input_imgs.to(torch.float32)
            if is_cuda:
                input_imgs = input_imgs.to(device)
                labels = labels.to(device)
            labels = labels.view(-1)
            features,_ = net(input_imgs)
            out = arcface_net(features).to(device)
            arcface_loss = criterion(out,labels.long())
            center_loss = centerloss_net(features,labels)
            loss = alpha*arcface_loss + (1-alpha)*center_loss
            outputs = out
            predicted= torch.argmax(outputs, dim=1)
            # voting
            if predicted == 0:
                num[0] += 1
            elif predicted == 1:
                num[1] += 1
            elif predicted== 2:
                num[2] += 1
            else:
                num[3] += 1
            labels = labels.squeeze()
            running_loss+=loss.item()
        time = time + 1
        predicted3 = np.argmax(num)
        truelabel[i] = labels.cpu()
        predictedlabel[i] = predicted3
        evaluation += torch.sum(predicted3 == labels.data)
    # compute kappa
    truelabel = truelabel[:i+1]
    predictedlabel = predictedlabel[:i+1]
    cm = confusion_matrix(truelabel, predictedlabel)
    n = np.sum(cm)
    sum_po = 0
    sum_pe = 0
    for i in range(len(cm[0])):
        sum_po += cm[i][i]
        row = np.sum(cm[i, :])
        col = np.sum(cm[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    kappa = (po - pe) / (1 - pe)
    running_loss = running_loss/(i+1)
    running_loss = running_loss/11
    running_acc = evaluation/time
    return running_loss,running_acc,kappa,truelabel,predictedlabel
def TrainTest_Model(model, trainloader, testloader, n_epoch=30, opti='Adam', learning_rate=0.001, is_cuda=True,
                    print_epoch=5, verbose=False, name='null', fold=-1):
    if is_cuda:
        net = model(device=device).to(device)
    else:
        net = model()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    """
      centerloss和arcfaceloss需
      """
    cls_num, feature_dim = 4, 128
    arcface_net = ArcFaceNet(cls_num, feature_dim).to(device)
    centerloss_net = CenterLossNet(cls_num, feature_dim).to(device)
    criterion = nn.NLLLoss()

    if opti == 'SGD':
        optimizer = optim.SGD([{'params': net.parameters()},
                               {'params': arcface_net.parameters()},
                               {'params': centerloss_net.parameters()}], lr=0.001)
    elif opti == 'Adam':
        optimizer = optim.Adam([{'params': net.parameters()},
                                {'params': arcface_net.parameters()},
                                {'params': centerloss_net.parameters()}], lr=0.001)
    else:
        print('Optimizer:' + opti + 'not implemented.')
    train_loss = np.zeros((n_epoch,))
    test_loss = np.zeros((n_epoch,))
    train_acc = np.zeros((n_epoch,))
    test_acc = np.zeros((n_epoch,))
    bestAcc = 0
    bestkappa = 0
    alpha = 0.95
    print('the value of w：', alpha)
    for epoch in range(n_epoch):
        net.train()
        running_loss = 0.0
        evaluation = 0.0
        time = 0
        for i, data in enumerate(trainloader, 0):
            input, labels = data
            optimizer.zero_grad()
            inputs = input.reshape(len(input), 1, 22, 1000)
            inputs = torch.as_tensor(inputs)
            inputs = torch.as_tensor(inputs)
            """
            without arcface
            """
            # outputs = net(inputs.to(torch.float32).to(device))
            """
            without center loss
            """
            # outputs = net(inputs.to(torch.float32).to(device), labels.to(torch.int64).to(device))
            '''
            arcface loss + center loss
            '''
            labels = labels.view(-1)
            labels = labels.squeeze()
            labels = labels.to(device)
            features, _ = net(inputs.to(torch.float32).to(device))
            out = arcface_net(features).to(device)
            arcface_loss = criterion(out, labels.long())
            center_loss = centerloss_net(features, labels).to(device)
            loss = alpha * arcface_loss + (1 - alpha) * center_loss
            outputs = out
            predicted = torch.argmax(outputs, dim=1)
            evaluation += torch.sum(predicted == labels.data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            time = time + len(labels)

        running_loss = running_loss / (i + 1)
        running_acc = evaluation / time
        validation_loss, validation_acc, kappa, truelabel, predictedlabel = Test_Model(net, testloader, criterion,
                                                                                       arcface_net, centerloss_net,
                                                                                       alpha, True)

        if validation_acc > bestAcc:
            bestAcc = validation_acc
        if kappa > bestkappa:
            bestkappa = kappa
            # filepath = 'BCI_IV2a/trueANDpreLabel/Subject' + str(name) + '_' + str(fold + 1) + '.mat'
            # scio.savemat(filepath, {'truelabel': truelabel,
            #                         'predictedlabel': predictedlabel})
            # torch.save(net,'BCI_IV2a/trueANDpreLabel/Subject' + str(name) + '_' + str(fold + 1) + '.pkl')
            # state = {'model': net.state_dict(),
            #          'arcface_net': arcface_net.state_dict(),
            #          'centerloss_net': centerloss_net.state_dict(),}
            # torch.save(state, 'BCI_IV2a/trueANDpreLabel/Subject' + str(name) + '_' + str(fold + 1) + '.pth')
        if epoch % 10 == 0:
            print(
                '[%d,%.3d]\tloss:%.4f\tAccuracy:%.4f\t\tval-loss:%.4f\tval_Accuracy:%.4f\tval_Kappa:%.4f\t\tbest-Accuracy:%.4f\tbest-Kappa:%.4f' %
                (epoch + 1, n_epoch, running_loss, running_acc, validation_loss, validation_acc, kappa, bestAcc,
                 bestkappa))
        train_loss[epoch], test_loss[epoch], train_acc[epoch], test_acc[
            epoch] = running_loss, validation_loss, running_acc, validation_acc
    if verbose:
        print(
            'Finished Training \n loss:%.4f\tAccuray:%.4f\t\tval-Accuracy:%.4f\t\tbest-Accuracy:%.4f\tbest-Kappa:%.4f' %
            (running_loss, running_acc, validation_loss, validation_acc, bestAcc, bestkappa))

    plot.PLOT4(train_acc, test_acc, train_loss, test_loss, n_epoch, name)
    return (running_loss, running_acc, validation_loss, bestAcc, bestkappa)