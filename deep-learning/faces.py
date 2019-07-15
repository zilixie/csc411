import os
from pylab import *
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from scipy.ndimage import filters
from scipy.io import loadmat
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import glob
import urllib
from torch.autograd import Variable
import torch
import torch.utils.data as data_utils
import shutil
from torch.utils.data import Dataset, DataLoader,TensorDataset
import sys


reload(sys)
sys.setdefaultencoding('utf8')

###### global ######
train_set = {}
valid_set = {}
test_set = {}
file_list = []

act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Lorraine Bracco', 'Angie Harmon', 'Peri Gilpin']

act_test = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'America Ferrera', 'Fran Drescher', 'Kristin Chenoweth']

act_all = act + act_test

one_hot_encoding = {'Alec Baldwin':[[0],[0],[0],[1],[0],[0]],'Bill Hader':[[0],[0],[0],[0],[1],[0]],'Lorraine Bracco':[[1],[0],[0],[0],[0],[0]], 'Peri Gilpin':[[0],[1],[0],[0],[0],[0]], 'Angie Harmon':[[0],[0],[1],[0],[0],[0]], 'Steve Carell':[[0],[0],[0],[0],[0],[1]]}

###### download data ######
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

def resize_gray(image):
    try:
        image = imresize(image, (32,32))
        output = rgb2gray(image)
    except IndexError:
        pass
    return output
                

def get_data(file):
    for a in act_all:
        name = a.split()[1].lower()
        i = 0
        for line in open(file):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                try:
                    image = imread("uncropped/"+filename)
                    x1,y1,x2,y2 = map(int, line.split()[5].split(","))
                    face = image[y1:y2, x1:x2]
                    im_gray = rgb2gray(face)
                    im_resize = imresize(im_gray, (32,32))
                    imsave("cropped/"+filename, im_resize)
                except:
                    continue
                
                print filename
                i += 1

testfile = urllib.URLopener()
#get_data("facescrub_actors.txt")
#get_data("facescrub_actresses.txt")


###### separate data ######
def actor_name(name):
    name = ''.join([i for i in name if not i.isdigit()]).lower()
    name = name.replace(".jpg", "")
    name = name.replace(".jpeg", "")
    name = name.replace(".png", "")
    name = name.replace(".JPG", "")
    return name  
    
def shuffle_list():
    for f in os.listdir("cropped"):
        f_name = actor_name(f)
        if f_name in ['baldwin', 'hader', 'carell', 'bracco', 'harmon', 'gilpin']:
            file_list.append(f)
    random.shuffle(file_list)

shuffle_list()

def separate():
    for a in act:  
        for f in file_list:
            f_name = actor_name(f)
            name = a.split()[1].lower()
            if name == f_name:
                if name not in train_set:
                    train_set[name] = []
                    test_set[name] = []
                    valid_set[name] = []
                if len(train_set[name]) < 60:
                    train_set[name].append(f)
                elif len(valid_set[name]) < 10:
                    valid_set[name].append(f)
                elif len(test_set[name]) < 20:
                    test_set[name].append(f)
                else:
                    break
            
separate()


train_set['Alec Baldwin'] = train_set.pop('baldwin')
train_set['Bill Hader'] = train_set.pop('hader')
train_set['Steve Carell'] = train_set.pop('carell')
train_set['Lorraine Bracco'] = train_set.pop('bracco')
train_set['Angie Harmon'] = train_set.pop('harmon')
train_set['Peri Gilpin'] = train_set.pop('gilpin')

valid_set['Alec Baldwin'] = valid_set.pop('baldwin')
valid_set['Bill Hader'] = valid_set.pop('hader')
valid_set['Steve Carell'] = valid_set.pop('carell')
valid_set['Lorraine Bracco'] = valid_set.pop('bracco')
valid_set['Angie Harmon'] = valid_set.pop('harmon')
valid_set['Peri Gilpin'] = valid_set.pop('gilpin')

test_set['Alec Baldwin'] = test_set.pop('baldwin')
test_set['Bill Hader'] = test_set.pop('hader')
test_set['Steve Carell'] = test_set.pop('carell')
test_set['Lorraine Bracco'] = test_set.pop('bracco')
test_set['Angie Harmon'] = test_set.pop('harmon')
test_set['Peri Gilpin'] = test_set.pop('gilpin')


def construct_set(target):
    y = np.empty((6,0))
    data = np.empty((1024,0))
    for a in one_hot_encoding.keys():
        for image in target[a]:
            y = np.hstack((y,np.array(one_hot_encoding[a])))
            im = imread("cropped/"+image)
            flat_im = (im/255.).reshape(1024,1)
            data = np.hstack((data,flat_im))
            
    return data.T,y.T

def produce():
    train_data,train_label = construct_set(train_set)
    test_data,test_label = construct_set(test_set)
    valid_data,valid_label = construct_set(valid_set)
    
    return train_data,train_label,test_data,test_label,valid_data,valid_label 

###### Part 8 ######
def part8(alpha, epoch_times):
    torch.manual_seed(0)
    train_data,train_label,test_data,test_label,valid_data,valid_label = produce()
    
    TRAIN = np.hstack((train_data,np.argmax(train_label, 1).reshape((train_data.shape[0],1))))
    VALID = np.hstack((valid_data,np.argmax(valid_label, 1).reshape((valid_data.shape[0],1))))
    TEST = np.hstack((test_data,np.argmax(test_label, 1).reshape((test_data.shape[0],1))))
    
    dim_x = 32*32
    dim_h = 12
    dim_out = 6
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    
    model = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_h, dim_out),
            torch.nn.Softmax()
            )
    
    model[0].weight.data.normal_(0.0,0.01)
    model[2].weight.data.normal_(0.0,0.01)
    model[0].bias.data.normal_(0.0,0.01)
    model[2].bias.data.normal_(0.0,0.01)
    
    dataloader = DataLoader(TRAIN, batch_size=32,shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    
    train_x = Variable(torch.from_numpy(TRAIN[:,:-1]), requires_grad=False).type(dtype_float)
    validation_x = Variable(torch.from_numpy(VALID[:,:-1]), requires_grad=False).type(dtype_float)
    test_x = Variable(torch.from_numpy(TEST[:,:-1]), requires_grad=False).type(dtype_float)
    
    tr,va, te, cx= [],[],[],[]
    
    for epoch in range(epoch_times):
        for i, data in enumerate(dataloader):
            x = Variable(data[:,:-1], requires_grad=False).type(dtype_float)
            y_classes = Variable(data[:,-1], requires_grad=False).type(dtype_long)
            y_pred = model(x)
            loss = loss_fn(y_pred, y_classes)
        
            model.zero_grad()
            loss.backward() 
            optimizer.step()
        
        if epoch % 5 == 0:
            cx.append(epoch)
            
            train_set_result = model(train_x).data.numpy()
            tr.append((np.mean(np.argmax(train_set_result, 1) == TRAIN[:,-1])))
            
            valid_set_result = model(validation_x).data.numpy()
            va.append((np.mean(np.argmax(valid_set_result, 1) == VALID[:,-1])))
            
            test_set_result = model(test_x).data.numpy()
            te.append((np.mean(np.argmax(test_set_result, 1) == TEST[:,-1])))
            
    # plt.plot(cx, tr,'g',cx, va,'r',cx, te, 'y')
    # plt.xlabel('epoches')
    # plt.ylabel('accuracy')
    # plt.savefig('part8.jpg')
    # plt.show()

    y_pred = model(test_x).data.numpy()
    print ("accuracy:")
    print (np.mean(np.argmax(y_pred, 1) == TEST[:,-1]))
    
    return model

model = part8(2e-4, 2000)

#### Part 9 ####
for i in range(12):
    plt.imshow(model[0].weight.data.numpy()[i, :].reshape((32, 32)), cmap=plt.cm.coolwarm)
    plt.savefig('part9'+str(i)+'.jpg')
