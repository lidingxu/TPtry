import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

data = scio.loadmat('data_orsay_2017.mat')
datakeys = list(data.keys())
print(datakeys)
Xtest = data['Xtest']
ytest = data['ytest']
Xtrain = data['Xtrain']
ytrain = data['ytrain']
wtest_square = data['wtest_square']
wtest_logistic = data['wtest_logistic']
wtest_hinge = data['wtest_hinge']
#Xtest = Xtest.reshape((100, -1))
#Xtrain = Xtrain.reshape((100, -1))
print(Xtest.shape)
print(ytest.shape)
print(Xtrain.shape)
print(ytrain.shape)
print(wtest_square.shape)
print(wtest_logistic.shape)
print(wtest_hinge.shape)

max_iter = 2500
step_size = 0.05

class Squared:
    name = "squared"

    def loss(self, X, y, w):
        res = X.dot(w)-y
        return np.transpose(res).dot(res)/ y.size


    def gd(self, Xtrain, ytrain, Xtest, ytest):
        train_loss = np.zeros(max_iter)
        test_loss = np.zeros(max_iter)
        w = np.ones((100,1))
        train_size = ytrain.size
        for i in range(max_iter):
            train_loss[i] = self.loss(Xtrain, ytrain, w)
            test_loss[i] = self.loss(Xtest, ytest, w)
            res = Xtrain.dot(w)-ytrain
            gradient = np.transpose(Xtrain).dot(res)  / train_size
            w = w - step_size * gradient
            if i % 5 == 0:
                print('iter:%d train loss:%2f test_loss:%2f'%(i, train_loss[i], test_loss[i]))
        return  train_loss, test_loss

class Logistic:

    name = "logistic"
    
    def empirical_prob(self, X, w):
        return 1/ (1  + np.exp(-X.dot(w)))

    def loss(self, X, y, w):
        h = self.empirical_prob(X, w)
        loss = np.transpose(y).dot(np.log(h)) + np.transpose(1 - y).dot(np.log(1 - h))
        return -sum(loss)/ y.size


    def gd(self, Xtrain, ytrain, Xtest, ytest):
        train_loss = np.zeros(max_iter)
        test_loss = np.zeros(max_iter)
        w = np.ones((100,1))
        train_size = ytrain.size
        for i in range(max_iter):
            train_loss[i] = self.loss(Xtrain, ytrain, w)
            test_loss[i] = self.loss(Xtest, ytest, w)
            h = self.empirical_prob(Xtrain, w) 
            gradient = np.transpose(Xtrain).dot(h-ytrain) / train_size
            w = w - step_size * gradient 
            if i % 5 == 0:
                print('iter:%d train loss:%2f test_loss:%2f'%(i, train_loss[i], test_loss[i]))
        return  train_loss, test_loss


def plt_algo(Xtrain, ytrain, Xtest, ytest, algorithm):
    train_loss, test_loss = algorithm.gd(Xtrain, ytrain, Xtest, ytest)
    plt.figure()
    plt.plot(range(1, max_iter + 1), train_loss, label='train data', color='r')
    plt.plot(range(1, max_iter + 1), test_loss, label='test data', color='b')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.title('{} loss, train size{}'.format(algorithm.name, ytrain.size))
    plt.show()

Xtrain_small = Xtrain[::10,:]
ytrain_small = ytrain[::10,:]
plt_algo(Xtrain, ytrain, Xtest, ytest, Squared())
plt_algo(Xtrain, ytrain, Xtest, ytest, Logistic())
plt_algo(Xtrain_small, ytrain_small, Xtest, ytest, Squared())
plt_algo(Xtrain_small, ytrain_small, Xtest, ytest, Logistic())


