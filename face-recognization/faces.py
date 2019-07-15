import os
os.mkdir("cropped/")
os.mkdir("uncropped/")

os.mkdir("cropped/bracco/")
os.mkdir("cropped/gilpin/") 
os.mkdir("cropped/harmon/")
os.mkdir("cropped/baldwin/") 
os.mkdir("cropped/hader/") 
os.mkdir("cropped/carell/") 

os.mkdir("cropped/radcliffe/") 
os.mkdir("cropped/butler/") 
os.mkdir("cropped/vartan/") 
os.mkdir("cropped/chenoweth/") 
os.mkdir("cropped/drescher/") 
os.mkdir("cropped/ferrera/")   

os.mkdir("uncropped/bracco/")
os.mkdir("uncropped/gilpin/") 
os.mkdir("uncropped/harmon/")
os.mkdir("uncropped/baldwin/") 
os.mkdir("uncropped/hader/") 
os.mkdir("uncropped/carell/") 

os.mkdir("uncropped/radcliffe/") 
os.mkdir("uncropped/butler/") 
os.mkdir("uncropped/vartan/") 
os.mkdir("uncropped/chenoweth/") 
os.mkdir("uncropped/drescher/") 
os.mkdir("uncropped/ferrera/")   

os.mkdir("train/")
os.mkdir("valid/")
os.mkdir("test/") 

from pylab import *
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from scipy.ndimage import filters

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import glob
import urllib
import shutil
import sys

reload(sys)
sys.setdefaultencoding('utf8')


###### global ######
train_set = {}
valid_set = {}
test_set = {}

act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Lorraine Bracco', 'Angie Harmon', 'Peri Gilpin']

act_test = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'America Ferrera', 'Fran Drescher', 'Kristin Chenoweth']

act_all = act + act_test

one = np.array([[1]])
zero = np.array([[0]])

###### Part1 ######
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
        
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.
    

def get_images(file, testfile):
    for a in act_all:
        name = a.split()[1].lower()
        i = 0
        for line in open(file):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], 
                    "uncropped/"+name+"/"+filename), {}, 60)
                if not os.path.isfile("uncropped/"+name+"/"+filename):
                    continue
                try:
                    image = imread("uncropped/"+name+"/"+filename)
                    x1,y1,x2,y2 = map(int, line.split()[5].split(","))
                    crop = image[y1:y2, x1:x2]
                    crop_gray = rgb2gray(crop)
                    crop_gray_resize = imresize(crop_gray, (32,32))
                    imsave("cropped/"+name+"/"+filename, crop_gray_resize)
                except:
                    continue
                print filename
                i += 1

testfile = urllib.URLopener()
get_images("facescrub_actors.txt", testfile)
get_images("facescrub_actresses.txt", testfile)


###### Part2 ######
def Part2():

    for a in act:
        name = a.split()[1].lower()
        act_images = os.listdir("cropped/" + name + "/")
        random.shuffle(act_images)
        if name not in test_set:
            train_set[name] = []
            test_set[name] = []
            valid_set[name] = [] 
        for img in act_images:
            if len(test_set[name]) < 10:
                test_set[name].append(img)
            elif len(valid_set[name]) < 10:
                valid_set[name].append(img)
            elif (len(train_set[name]) < 100 and 
            len(train_set[name]) < len(os.listdir("cropped/" + name + "/")) - 20):
                train_set[name].append(img)
            else:
                break 
                
    for a in act:  
        name = a.split()[1].lower()
        path = "cropped/" + name + "/"
        
        for img in train_set[name]:
            shutil.copy(path + img, "train/")
        for img in valid_set[name]:
            shutil.copy(path + img, "valid/") 
        for img in test_set[name]:
            shutil.copy(path + img, "test/")
            
Part2()

###### Part3 ######
def f(x, y, theta):
    return sum((y - dot(x, theta))**2)
    
def df(x, y, theta):
    return -2 * (dot(x.T , (y - dot(x, theta))))

def grad_descent(x, y, init_t, df, iter_time, a):
    alpha = a
    t = init_t.copy()
    max_iteration = iter_time
    iter  = 0
    while iter < max_iteration:
        t -= alpha*df(x, y, t)
        #if iter%20000 == 0:
        #    print df(x, y, t)
        iter += 1
    return t

def get_accuracy(x, theta, length, type):
    result = np.dot(x, theta)
    count = 0
    for i in range(0, 2*length):
        if result[i] >= 0.5 and i < length:
            count = count + 1
        elif result[i] < 0.5 and i >= length:
            count = count + 1    
    print type + " accuracy: %.4f" % (float(count)/float(2*length))
    accu = float(count)/float(2*length)
    return count 


def Part3_1():
    x = np.array([[]])
    x_valid = np.array([[]])
    x_test = np.array([[]])
    y = np.array([[]])
    y_valid = np.array([[]])
    
    for i in range(0, 200):
        if i < 100:
            flat_img = np.reshape(imread("train/"+train_set['carell'][i]),(1, 1024))
            x = np.concatenate((x, np.concatenate((one, flat_img/255.0),1)), 1)      
        else:
            flat_img = np.reshape(imread("train/"+train_set['baldwin'][i - 100]),(1, 1024))
            x = np.concatenate((x, np.concatenate((one, flat_img/255.0),1)), 1)  
    
    for i in range(0, 20):
        if i < 10:
            flat_img = np.reshape(imread("valid/"+valid_set['carell'][i]),(1, 1024))
            x_valid = np.concatenate((x_valid, np.concatenate((one, flat_img/255.0),1)), 1)
        else:
            flat_img = np.reshape(imread("valid/"+valid_set['baldwin'][i - 10]),(1, 1024))
            x_valid = np.concatenate((x_valid, np.concatenate((one, flat_img/255.0),1)), 1)
        
    for i in range(0, 20):
        if i < 10:
            flat_img = np.reshape(imread("test/"+test_set['carell'][i]),(1, 1024))
            x_test = np.concatenate((x_test, np.concatenate((one, flat_img/255.0),1)), 1)
        else:
            flat_img = np.reshape(imread("test/"+test_set['baldwin'][i - 10]),(1, 1024))
            x_test = np.concatenate((x_test, np.concatenate((one, flat_img/255.0),1)), 1)
    
    for i in range(0, 200):
        if i < 100:
            y = np.concatenate((y, one), 1)
        else:
            y = np.concatenate((y, zero), 1) 
            
    for i in range(0, 20):
        if i < 10:
            y_valid = np.concatenate((y_valid, one), 1)
        else:
            y_valid = np.concatenate((y_valid, zero), 1) 
                     
    x = np.reshape(x, (200, 1025))
    x_valid = np.reshape(x_valid, (20, 1025))
    x_test = np.reshape(x_test, (20, 1025))
    y = np.reshape(y, (200, 1))
    y_valid = np.reshape(y_valid, (20, 1))
    
    return x, x_valid, x_test, y, y_valid

x, x_valid, x_test, y, y_valid = Part3_1()

def Part3_2(x, x_valid, x_test, y):
    init_t = np.zeros((1025,1))
    theta = grad_descent(x, y, init_t, df, 100000, 9e-6)
    
    square_theta = np.reshape(theta[1:], (32,32))
    imsave("100img_theta.jpg", square_theta)
    
    count = get_accuracy(x, theta, 100, "train set")
    
    count = get_accuracy(x_valid, theta, 10, "validate set")
    
    count = get_accuracy(x_test, theta, 10, "test set")
    
Part3_2(x, x_valid, x_test, y)


init_t = np.zeros((1025,1))
iter_times = []

train_cost, validate_cost= [], []

for ind in range(1,41):
    iter_times.append(2500*ind)

for iter in iter_times:
    theta = grad_descent(x, y, init_t, df, iter, 9e-7)
    train_cost.append(f(x,y,theta)/200)
    validate_cost.append(f(x_valid,y_valid, theta)/20)
    
    
cx = iter_times
plt.plot(cx, train_cost, 'b', cx, validate_cost, 'g')
plt.savefig("part3.jpg")


    

###### Part4(a) ######
visualized_theta = imread("100img_theta.jpg")

def Part4_a():
    x = np.array([[]])
    y = np.array([[]])
    
    random.shuffle(train_set['carell'])
    random.shuffle(train_set['baldwin'])
    
    for i in range(0, 4):
        if i < 2:
            flat_img = np.reshape(imread("train/"+train_set['carell'][i]),(1, 1024))
            print train_set['carell'][i]
            x = np.concatenate((x, np.concatenate((one, flat_img/255.0),1)), 1)            
        else:
            flat_img = np.reshape(imread("train/"+train_set['baldwin'][i - 2]),(1, 1024))
            print train_set['baldwin'][i-2]
            x = np.concatenate((x, np.concatenate((one, flat_img/255.0),1)), 1)  
    
    for i in range(0, 4):
        if i < 2:
            y = np.concatenate((y, one), 1)
        else:
            y = np.concatenate((y, zero), 1)
            
    x = np.reshape(x, (4, 1025))
    y = np.reshape(y, (4, 1))
    
    init_t = np.zeros((1025,1))
    theta = grad_descent(x, y, init_t, df, 100000, 9e-6)
    
    square_theta = np.reshape(theta[1:], (32,32))
    imsave("2img_theta.jpg", square_theta)
    
    return x, y, theta
    
x2, y2, theta2 = Part4_a()

###### Part4(b) ######
def Part4_b(x, y):
    iter_times = [1, 10, 100, 1000, 10000, 100000]
    for iter in iter_times:
        theta = grad_descent(x, y, init_t, df, iter, 9e-6)
        square_theta = np.reshape(theta[1:], (32,32))
        imsave(str(iter)+".jpg", square_theta)

Part4_b(x, y)

###### Part5 ######

train_line, validate_line, test_line = [], [], []

def Part5_1(size, iter_time, a):
    x = np.array([[]])
    y = np.array([[]])

    for i in range(0, size*6):
        if i < size*3:
            y = np.concatenate((y, one), 1)
        else:
            y = np.concatenate((y, zero), 1)

    for name in act:
        for i in range(0,size):
            flat_img = imread("train/"+train_set[name.split()[1].lower()][i])
            flat_img = np.reshape(flat_img,(1, 1024))
            row = np.concatenate((one, flat_img/255.0),1)
            x = np.concatenate((x, row), 1)
            
    y = np.reshape(y, (size*6, 1))
    x = np.reshape(x, (size*6, 1025))
    
    init_t = np.zeros((1025,1))
    theta = grad_descent(x, y, init_t, df, iter_time, a)
    return theta, x, y

#t, xi,yi = Part5_1(50, 100000, 9e-6)
#t, xi, yi = Part5_1(95, 140000, 18e-9)
  
def Part5_2():
    x_valid_p5 = np.array([[]])
    for name in act:
        for i in range(0,10):
            flat_img = imread("valid/"+valid_set[name.split()[1].lower()][i])
            flat_img = np.reshape(flat_img,(1, 1024))
            row = np.concatenate((one, flat_img/255.0),1)
            x_valid_p5 = np.concatenate((x_valid_p5, row), 1)
    
    x_test_p5 = np.array([[]])
    for name_test in act_test:
        for j in range(0,20):
            act_images = os.listdir("cropped/" + name_test.split()[1].lower() + "/")
            random.shuffle(act_images)
            flat_img = imread("cropped/" + name_test.split()[1].lower() + "/" + act_images[i])
            flat_img = np.reshape(flat_img,(1, 1024))
            row = np.concatenate((one, flat_img/255.0),1)
            x_test_p5 = np.concatenate((x_test_p5, row), 1)
        
    x_valid_p5 = np.reshape(x_valid_p5, (60, 1025))
    x_test_p5 = np.reshape(x_test_p5, (120, 1025))
    
    return x_valid_p5, x_test_p5
#  
def Part5_3(x, x_valid_p5, x_test_p5, theta, size): 
    count_train = 0
    result_p5 = np.dot(x, theta)
    for i in range(0, size*6):
        if result_p5[i] >= 0.5 and i < 3*size:
            count_train += 1
        elif result_p5[i] < 0.5 and i >= 3*size:
            count_train += 1
    accu = float(count_train)/float(size*6)
    print "part5: size=%d, train set accuracy: %.4f" % (size, accu)      
    train_line.append(float(count_train)/float(6*size))
    
    count_valid = 0
    result_p5 = np.dot(x_valid_p5, theta)
    for i in range(0, 60):
        if result_p5[i] >= 0.5 and i < 30:
            count_valid += 1
        elif result_p5[i] < 0.5 and i >= 30:
            count_valid += 1
    accu = float(count_valid)/float(60)
    print "part5: size=%d, validate set accuracy: %.4f" % (size, accu)
    validate_line.append(float(count_valid)/float(60))


    count_test = 0
    result_p5 = np.dot(x_test_p5, theta)
    for i in range(0, 120):
        if result_p5[i] >= 0.5 and i < 60:
            count_test += 1
        elif result_p5[i] < 0.5 and i >= 60:
            count_test += 1
    accu = float(count_test)/float(120)
    print "part5: size=%d, test set accuracy: %.4f" % (size, accu)
    test_line.append(float(count_test)/float(120))

x_valid_p5, x_test_p5 = Part5_2()
#x_ind = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
x_ind = []
for ind in range(1,51):
    x_ind.append(2*ind)
    
for s in x_ind:
    t, xi, yi = Part5_1(s, 100000, 9e-7)
    Part5_3(xi, x_valid_p5, x_test_p5, t, s)

    
cx = x_ind
plt.plot(cx, train_line, 'g', cx, validate_line, 'y', cx, test_line, 'r')
plt.savefig("part5.jpg")

###### Part6 ######
def f_p6(x, y, theta):
    return sum((y - dot(theta.T, x))**2)

def df_p6(x, y, theta):
    return 2*np.dot(x, (dot(theta.T, x) - y).T)


def part6d(n,m):
    y = np.ones([6, 600])
    x = np.ones([1025,600])
    h = 1e-5
    theta = np.ones([1025, 6])
    theta_h = np.ones([1025,6])
    theta_h[n,m] = theta_h[n,m]+h
    
    print (f_p6(x, y, theta_h) - f_p6(x, y, theta))/h
    print df_p6(x, y, theta)[n,m]
part6d(1,2)



###### Part7 ######

def Part7_1():
    x_p7 = np.array([[]])
    y_p7 = np.array([[]])
    x_valid_p7 = np.array([[]])
    y_valid_p7 = np.array([[]])
    
    
    for i in range(0, 6):
        for j in range(0,100):
            if i == 0:
                y_p7 = np.concatenate((y_p7, np.array([[1,0,0,0,0,0]])), 1)
            if i == 1:
                y_p7 = np.concatenate((y_p7, np.array([[0,1,0,0,0,0]])), 1)
            if i == 2:
                y_p7 = np.concatenate((y_p7, np.array([[0,0,1,0,0,0]])), 1)
            if i == 3:
                y_p7 = np.concatenate((y_p7, np.array([[0,0,0,1,0,0]])), 1)
            if i == 4:
                y_p7 = np.concatenate((y_p7, np.array([[0,0,0,0,1,0]])), 1)
            if i == 5:
                y_p7 = np.concatenate((y_p7, np.array([[0,0,0,0,0,1]])), 1)
    y_p7 = np.reshape(y_p7, (600, 6))
    
    for i in range(0, 6):
        for j in range(0,10):
            if i == 0:
                y_valid_p7 = np.concatenate((y_valid_p7, np.array([[1,0,0,0,0,0]])), 1)
            if i == 1:
                y_valid_p7 = np.concatenate((y_valid_p7, np.array([[0,1,0,0,0,0]])), 1)
            if i == 2:
                y_valid_p7 = np.concatenate((y_valid_p7, np.array([[0,0,1,0,0,0]])), 1)
            if i == 3:
                y_valid_p7 = np.concatenate((y_valid_p7, np.array([[0,0,0,1,0,0]])), 1)
            if i == 4:
                y_valid_p7 = np.concatenate((y_valid_p7, np.array([[0,0,0,0,1,0]])), 1)
            if i == 5:
                y_valid_p7 = np.concatenate((y_valid_p7, np.array([[0,0,0,0,0,1]])), 1)
    y_valid_p7 = np.reshape(y_valid_p7, (60, 6))
    
    
    for name in act:
        for i in range(0,100):
            flat_img = imread("train/"+train_set[name.split()[1].lower()][i])
            flat_img = np.reshape(flat_img,(1, 1024))
            row = np.concatenate((one, flat_img/255.0),1)
            x_p7 = np.concatenate((x_p7, row), 1)
            
    x_p7 = np.reshape(x_p7, (600, 1025))
    
    for name in act:
        for i in range(0,10):
            flat_img = imread("valid/"+valid_set[name.split()[1].lower()][i])
            flat_img = np.reshape(flat_img,(1, 1024))
            row = np.concatenate((one, flat_img/255.0),1)
            x_valid_p7 = np.concatenate((x_valid_p7, row), 1)
            
    x_valid_p7 = np.reshape(x_valid_p7, (60, 1025))
    return x_p7, x_valid_p7, y_p7, y_valid_p7

x_p7, x_valid_p7, y_p7, y_valid_p7 = Part7_1()

init_t = np.zeros((1025,6))
iter_times = []


theta = grad_descent(x_p7, y_p7, init_t, df, 100000, 9e-7)
result = np.dot(x_p7, theta)
count = 0
for i in range(600):
    if np.argmax(y_p7[i]) == np.argmax(result[i]):
        count = count + 1
        
accu = float(count)/float(600)
print accu

result = np.dot(x_valid_p7, theta)
count = 0
for i in range(60):
    if np.argmax(y_valid_p7[i]) == np.argmax(result[i]):
        count = count + 1
        
accu = float(count)/float(60)
print accu

x_p7, x_valid_p7, y_p7, y_valid_p7 = Part7_1()

init_t = np.zeros((1025,6))
iter_times = []


train_line_p7, validate_line_p7, test_line_p7 = [], [], []

for ind in range(1,51):
    iter_times.append(1000*ind)

for iter in iter_times:
    theta = grad_descent(x_p7, y_p7, init_t, df, iter, 9e-7)
    result = np.dot(x_p7, theta)
    count = 0
    for i in range(600):
        if np.argmax(y_p7[i]) == np.argmax(result[i]):
            count = count + 1
            
    train_line_p7.append(float(count)/float(600))

    result = np.dot(x_valid_p7, theta)
    count = 0
    for i in range(60):
        if np.argmax(y_valid_p7[i]) == np.argmax(result[i]):
            count = count + 1
            
    validate_line_p7.append(float(count)/float(60))
    
cx = iter_times
plt.plot(cx, train_line_p7, 'g', cx, validate_line_p7, 'y')
plt.savefig("part7.jpg")
# 
# ###### Part8 ######

theta_baldwin = np.reshape(np.array([theta.T[0]]),(1025,1))
t = np.reshape(theta_baldwin[1:], (32,32))
imsave("theta_baldwin.jpg", t)
theta_hader = np.reshape(np.array([theta.T[1]]),(1025,1))
t = np.reshape(theta_hader[1:], (32,32))
imsave("theta_hader.jpg", t)
theta_carell = np.reshape(np.array([theta.T[2]]),(1025,1))
t = np.reshape(theta_carell[1:], (32,32))
imsave("theta_carell.jpg", t)
theta_bracco = np.reshape(np.array([theta.T[3]]),(1025,1))
t = np.reshape(theta_bracco[1:], (32,32))
imsave("theta_bracco.jpg", t)
theta_harmon = np.reshape(np.array([theta.T[4]]),(1025,1))
t = np.reshape(theta_harmon[1:], (32,32))
imsave("theta_harmon.jpg", t)
theta_gilpin = np.reshape(np.array([theta.T[5]]),(1025,1))
t = np.reshape(theta_gilpin[1:], (32,32))
imsave("theta_gilpin.jpg", t)
