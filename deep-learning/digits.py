from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle as cPickle
import io
import os
import random
from scipy.io import loadmat

import os


##Constant


M = loadmat("mnist_all.mat")
train = ["train0","train1", "train2", "train3", "train4", "train5", "train6", "train7", "train8", "train9"]

test =  ["test0","test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9"]

one_hot_encode = [array([[1] + [0] * 9]),
    array([[0] * 1 + [1] + [0] * 8]),
    array([[0] * 2 + [1] + [0] * 7]),
    array([[0] * 3 + [1] + [0] * 6]),
    array([[0] * 4 + [1] + [0] * 5]),
    array([[0] * 5 + [1] + [0] * 4]),
    array([[0] * 6 + [1] + [0] * 3]),
    array([[0] * 7 + [1] + [0] * 2]),
    array([[0] * 8 + [1] + [0] * 1]),
    array([[0] * 9 + [1]])]

##Functions

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))


def load_data():
    for s in train:
        M[s]=M[s]/255.
    for s in test:
        M[s]=M[s]/255.
        
    training_x = np.empty((0,28*28))
    test_x = np.empty((0,28*28))
    training_y = np.empty((0,10))
    test_y = np.empty((0,10))
    validation_x = np.empty((0,28*28))
    validation_y = np.empty((0,10))
    for i in range(10):
        training_x = np.vstack((training_x,M[train[i]]))
        temp_train_y = np.zeros((M[train[i]].shape[0],10))
        temp_train_y[:,i]=1
        training_y = np.vstack((training_y,temp_train_y))
        
        test_x = np.vstack((test_x,M[test[i]]))
        temp_test_y = np.zeros((M[test[i]].shape[0],10))
        temp_test_y[:,i]=1
        test_y = np.vstack((test_y,temp_test_y))
    return training_x, training_y,validation_x, validation_y, test_x, test_y

    
def forward(x, Wb):
    W1 = Wb[:,:-1]
    b1 = Wb[:,-1:]
    L1 = np.matmul(W1,x.T)+b1
    output = softmax(L1)
    return output
    
def NLL(y, output):
    """
    Negative log loss function
    """
    return -sum(y*log(output)) 

def backward(y,output,x):
    dy = output-y
    dW = np.matmul(dy,x)
    db = np.matmul(dy, np.ones([x.shape[0],1]))
    return np.hstack((dW, db))

##Part1
def get_num():
    for i in range(10):
        digit = "train" + str(i)
        for j in range(10):
            number = M[digit][j].reshape((28,28))
            mpimg.imsave("num/"+digit + "No" + str(j) + ".jpg", number, cmap = cm.gray)   
get_num()

def make_plot():
    fig = plt.figure()
    i = 1
    for f in os.listdir("num"):
        num = imread("num/" + f)
        plt.subplot(10,10,i)
        implot = plt.imshow(num)
        plt.axis('off')
        i += 1
    plt.savefig("part1.jpg", papertype='a4',orientation='portrait')
make_plot()

##Part2
def part2(x, W, b):
    o = dot(W.T, x) +b
    return softmax(o)
    
##Part3
#(a)
def cost_function(x,y_, W, b):
    return -sum(y_*log(part2(x, W, b)))
    
def d_cost_function(x,y_, W, b):
    return dot((softmax(part2(x, w, b)) - y.T),x).T
    

def gradient_W(x,y_, W):
    o = dot(W.T, x) #+b
    p = softmax(o)
    dCdo = p - y_

    dCdW = dot(x, dCdo.T)
    return dCdW
    
#(b)
def finite_differences(x, y, w, b, i, j):
    """
    Use finite difference to calculate the gradient on w_ij with h.
    x: input
    y: expected output
    w: weight
    b: bias
    """
    h = 10e-5
    c = cost_function(x,y, w, b) 
    w1 = w
    w1[i, j] += h
    c1 = cost_function(x,y, w1, b) 
    return (c1 - c)/h
    
def part3(x, y, b, w):
    """
    Compare the result of my gradient function NLL_gradient and the gradient get 
    from finite differences method. Test three times on different w_ij.
    """
    index1 = [377, 488, 599]  
    index2 = [5, 6, 7]
    
    for k in range(3):
        i = index1[k]
        j = index2[k]
        fd = finite_differences(x, y, w, b, i, j)
        g = gradient_W(x,y, w)[i, j]
        diff = abs(g - fd)
        print("Test "+str(k)+":")
        print("finite differences = "+str(fd))
        print("gradient = "+str(g))
        print("difference = "+str(diff))
        print("\n")   


# x1 = M["train8"][0]
# #x1.reshape(784,1)
# x2 = M["train8"][1]
# #x2.reshape(784,1)
# x = vstack((x1, x2)).T
# y = array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
# w = zeros((x.shape[0], 10))
# b = zeros((10, 2))
# part3(x, y, b, w)

##Part4
def test_performance(M, w, set):
    total = 0
    count = 0
    for ind in range(0, len(set)):
        for img in M[set[ind]]:
            img = np.array([img]);
            img = vstack((ones((1, img.shape[0])), img.T))
            if(argmax(softmax(dot(w.T, img))) == ind):
                count+=1
            total+=1
    print("count is ", count)
    print("total is ", total)
    performance = float(count)/float(total)
    print(performance)
    return performance

def grad_descent(df, x, y, init_w, alpha):
    EPS = 1e-15   #EPS = 10**(-5)
    max_iter = 7000
    iter  = 1
    t = init_w
    #t = vstack((b, init_w)) 
    prev_t = t-10*EPS;
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        iter += 1
    return t  

def grad_descent_p4(df, x, y, init_w, alpha):
    EPS = 1e-15   #EPS = 10**(-5)
    max_iter = 7000
    iter  = 1
    t = init_w
    #t = vstack((b, init_w)) 
    prev_t = t-10*EPS;
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        if(iter % 500 == 0):
            filename ='w_'+str(iter)+'.txt'
            np.savetxt("part4_theta/" + filename, t)
            print(t);
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        iter += 1
    return t
    

##Part5
def grad_descent_p5(df, x, y, init_w, alpha, gamma = 0.9):
    EPS = 1e-5   #EPS = 10**(-5)
    max_iter = 7000
    iter = 1
    v = 0
    t = init_w
    #t = vstack((b, init_w)) 
    prev_t = t-10*EPS;
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        if(iter % 500 == 0):
            filename ='w_'+str(iter)+'_p5.txt'
            np.savetxt("part5_theta/"+filename, t)
            print(t);
        prev_t = t.copy()
        v = gamma*v + alpha*df(x, y, t)
        t -= v
        iter += 1
    return t



##Part6
    
def grad_descent_p6(f, df, loss_function, x, y, init_t, gamma = 0.9, alpha=0.0001):
    EPS = 1e-5
    max_iter=700
    iter = 1
    row, column = init_t.shape[0],init_t.shape[1]
    v = np.zeros((row, column))

    t = init_t.copy()
    prev_t = init_t-10*EPS
    current_alpha = alpha
    current_loss = loss_function(y, f(x,t))

    while norm(t - prev_t) >  EPS and iter <= max_iter:
        
        prev_t = t.copy()
        v = gamma*v + current_alpha*df(y, f(x,t), x)
        t -= v
        
        if iter % 5 == 0:
            if current_loss < loss_function(y, f(x,t)):
                current_alpha = current_alpha / 2
            current_loss = loss_function(y, f(x,t))
            print "iteration", iter
            print current_loss
        iter += 1
    return t


def grad_descent_p6_v2(x, y, init_t, w1,w2,w1r,w1c,w2r,w2c, f, df, loss_function, gamma = 0.6):
    
    EPS = 1e-5
    max_iter=20
    
    enable_momentum = gamma
    row, column = init_t.shape[0],init_t.shape[1]
    v = np.zeros((row, column)) 
    trajectory = []
    
    t = init_t.copy()
    prev_t = t-10*EPS
    
    t[w1r,w1c] = w1
    t[w2r,w2c] = w2

    current_alpha = 0.0035
    current_loss = loss_function(y, f(x,t))
    
    iter  = 1
    while norm(t - prev_t) >  EPS and iter <= max_iter:
        trajectory.append((t[w1r,w1c],t[w2r,w2c]))
        prev_t = t.copy()
        
        v = enable_momentum*v + current_alpha*df(y, f(x,t), x)
        t[w1r,w1c] -= v[w1r,w1c]
        t[w2r,w2c] -= v[w2r,w2c]
        
        if iter % 5 == 0:
            current_loss = loss_function(y, f(x,t))
            print "iteration", iter
            print current_loss
        iter += 1
    
    return trajectory


def loss_function_p6(Wb, x, y, w1, w2, w1r, w1c, w2r, w2c):
    Wb_cp = Wb.copy()
    Wb_cp[w1r,w1c] = w1
    Wb_cp[w2r,w2c] = w2
    b1 = Wb_cp[:,-1:]
    W1 = Wb_cp[:,:-1]
    L1 = np.matmul(W1,x.T)+b1
    output = softmax(L1)
    
    return NLL(y.T, output)

def part6(w1r,w1c,w2r,w2c):    
    np.random.seed(0)
    train_x_p6, train_y_p6,validation_x,validation_y, test_x, test_y = load_data()
    np.random.seed(1)
    Wb = np.random.normal(0.,0.001,[10,785])
    
    if not os.path.exists("Wb.txt"):
        Wb = grad_descent_p6(forward, backward, NLL, train_x_p6, train_y_p6.T, Wb,alpha=0.000001)
        np.savetxt("Wb.txt",Wb)
    else:
        Wb = np.loadtxt("Wb.txt")

    temp1 = Wb[w1r,w1c]
    temp2 = Wb[w2r,w2c]
    w1s = np.arange(temp1-2, temp1+2, 0.1)
    w2s = np.arange(temp2-2, temp2+2, 0.1)
    
    Cost = np.zeros([w1s.size, w2s.size])
    for ind1, w1 in enumerate(w1s):
        for ind2, w2 in enumerate(w2s):
            Cost[ind2,ind1] = loss_function_p6(Wb,train_x_p6,train_y_p6, w1,w2,w1r,w1c,w2r,w2c)
    w1_, w2_ = np.meshgrid(w1s, w2s)
    without_mo = grad_descent_p6_v2(train_x_p6, train_y_p6.T, Wb, temp1+0.8,temp2-1.8,w1r,w1c,w2r,w2c,forward, backward, NLL, gamma = 0)
    with_mo = grad_descent_p6_v2(train_x_p6, train_y_p6.T, Wb, temp1+0.8,temp2-1.8,w1r,w1c,w2r,w2c,forward, backward, NLL, gamma = 0.4)


    contour_plot = plt.contour(w1_, w2_, Cost, camp=cm.coolwarm)
    plt.plot([a for a, b in without_mo], [b for a,b in without_mo], 'y', label="without Momentum")
    plt.plot([a for a, b in with_mo], [b for a,b in with_mo], 'g', label="with Momentum")
    plt.title('plot contour')
    plt.savefig('part6.jpg')
    plt.show()
    
    
# np.random.seed(1)
# w1c = 28*(14  + int(round(6*np.random.rand())) - 3) + 14 + int(round(6*np.random.rand())) - 3
# w2c = 28*(14  + int(round(6*np.random.rand())) - 3) + 14 + int(round(6*np.random.rand())) - 3
# w1r = int(round((10-1)*np.random.rand()))
# w2r = int(round((10-1)*np.random.rand()))
# 
# part6(w1r,w1c,w2r,w2c)
    
    