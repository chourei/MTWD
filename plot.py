from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import manifold
from matplotlib.backends.backend_pdf import PdfPages
def PLOT4(trainAcc, testAcc , trainLoss , testLoss,epochs,name):
    result = np.vstack((trainAcc, testAcc))
    result = np.vstack((result,trainLoss))
    result = np.vstack((result, testLoss))
    # print(result)
    np.savetxt('resultOF'+str(name)+'.csv', result, delimiter=",", fmt='%.03f')
    X = np.arange(1, epochs + 1, 1)
    plt.suptitle("Subject_"+str(name))
    # 第一个图：acc
    plt.subplot(1, 2, 1)
    # plt.plot(X, trainLoss, c="r", label="train_loss")
    plt.plot(X, trainAcc, c="r", label='trainAcc')
    plt.plot(X, testAcc, c="g", label='testAcc')
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()
    # 第二个图：loss
    plt.subplot(1, 2, 2)
    # plt.title("折线图")
    plt.plot(X, trainLoss, c="r", label='trainLoss')
    plt.plot(X, testLoss, c="g", label='testLoss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    plt.tight_layout(pad=1.08)
    plt.savefig('acc_loss_of_'+str(name)+'.jpg')
    plt.show()