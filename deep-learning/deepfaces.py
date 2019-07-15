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
from torch.utils.data import Dataset, DataLoader,TensorDataset
import hashlib
import torchvision.models as models
import torchvision
import torch.nn as nn


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

one_hot_encoding = {'Alec Baldwin':[[1],[0],[0],[0],[0],[0]],'Bill Hader':[[0],[1],[0],[0],[0],[0]], 'Steve Carell':[[0],[0],[1],[0],[0],[0]], 'Lorraine Bracco':[[0],[0],[0],[1],[0],[0]], 'Angie Harmon':[[0],[0],[0],[0],[1],[0]], 'Peri Gilpin':[[0],[0],[0],[0],[0],[1]]}

part10_act = {0:'Alec Baldwin', 1:'Bill Hader', 2:'Steve Carell', 3:'Lorraine Bracco', 4:'Angie Harmon', 5:'Peri Gilpin'}


######
def actor_name(name):
    name = ''.join([i for i in name if not i.isdigit()]).lower()
    name = name.replace(".jpg", "")
    name = name.replace(".jpeg", "")
    name = name.replace(".png", "")
    name = name.replace(".JPG", "")
    return name  
    
def shuffle_list():
    for f in os.listdir("cropped_227/"):
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
                elif len(test_set[name]) < 10:
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



def construct_set(target,model):
    data = np.empty((0,9216))
    y = np.empty((6,0))
    for a in one_hot_encoding.keys():
        images_set = target[a]
        for image in images_set:
            y = np.hstack((y,np.array(one_hot_encoding[a])))
            img = imread("cropped_227/" + image)[:,:,:3]
            img = imresize(img,(227,227))
            img = img - np.mean(img.flatten())
            img = img/np.max(np.abs(img.flatten()))
            img = np.rollaxis(img, -1).astype(np.float32)
            img = Variable(torch.from_numpy(img).unsqueeze_(0), requires_grad=False)
            img = model.process(img)
            data = np.vstack((data,img))
    return data,y.T


class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
        
        classifier_weight_i = [0]
        for i in classifier_weight_i:
            self.classifier[i].weight.data.normal_(0.0,0.01)
            self.classifier[i].bias.data.normal_(0.0,0.01)

    def __init__(self, num_classes = 6):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 6),
            nn.Softmax()
            )
        
        self.load_weights()
        
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
        
    def process(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        feature = x.data.numpy()
        return feature
    
def produce(model):
    train_data,train_label = construct_set(train_set,model)
    test_data,test_label = construct_set(test_set,model)
    valid_data,valid_label = construct_set(valid_set,model)
    
    return train_data,train_label,test_data,test_label,valid_data,valid_label
    

def part10(alpha, epoch_times):    
    torch.manual_seed(0)
    model = MyAlexNet()
    
    train_data,train_label,test_data,test_label,valid_data,valid_label = produce(model)
    
    TRAIN = np.hstack((train_data,np.argmax(train_label, 1).reshape((train_data.shape[0],1))))
    VALID = np.hstack((valid_data,np.argmax(valid_label, 1).reshape((valid_data.shape[0],1))))
    TEST = np.hstack((test_data,np.argmax(test_label, 1).reshape((test_data.shape[0],1))))
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    dataloader = DataLoader(TRAIN, batch_size=32,shuffle=True)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=alpha)
    
    train_x = Variable(torch.from_numpy(TRAIN[:,:-1]), requires_grad=False).type(dtype_float)
    validation_x = Variable(torch.from_numpy(VALID[:,:-1]), requires_grad=False).type(dtype_float)
    test_x = Variable(torch.from_numpy(TEST[:,:-1]), requires_grad=False).type(dtype_float)
    
    tr = []
    va = []
    te = []
    cx = []
    
    for epoch in range(epoch_times):
        for i, data in enumerate(dataloader):
            x = Variable(data[:,:-1], requires_grad=False).type(dtype_float)
            y_classes = Variable(data[:,-1], requires_grad=False).type(dtype_long)
            y_predit_result = model.classifier(x)
            loss = loss_fn(y_predit_result, y_classes)
            model.classifier.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            cx.append(epoch)
            
            train_set_result = model.classifier(train_x).data.numpy()
            tr.append((np.mean(np.argmax(train_set_result, 1) == TRAIN[:,-1])))
            
            valid_set_result = model.classifier(validation_x).data.numpy()
            va.append((np.mean(np.argmax(valid_set_result, 1) == VALID[:,-1])))
            
            test_set_result = model.classifier(test_x).data.numpy()
            te.append((np.mean(np.argmax(test_set_result, 1) == TEST[:,-1])))
            
    plt.plot(cx, tr,'g',cx, va,'r', cx, te, 'y')
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.savefig('part10.jpg')
    plt.show()
        
    test_x = Variable(torch.from_numpy(TEST[:,:-1]), requires_grad=False).type(dtype_float)
    y_predit_result = model.classifier(test_x).data.numpy()
    print ("accuracy:" + str(np.mean(np.argmax(y_predit_result, 1) == TEST[:,-1])))
    
    return model
    
model = part10(1e-4, 400)
test_img_list = ['hader0.jpg', 'harmon0.jpg']

for image in test_img_list:
    img = imread("cropped_227/"+image)[:,:,:3]
    img = imresize(img,(227,227))
    img = img - np.mean(img.flatten())
    img = img/np.max(np.abs(img.flatten()))
    img = np.rollaxis(img, -1).astype(np.float32)
    img = Variable(torch.from_numpy(img).unsqueeze_(0), requires_grad=False)    
    
    prob = model.forward(img).data.numpy()[0]
    answer = np.argsort(prob)
    
    for i in range(6):
        print("-----------------------")
        print("answer:", part10_act[answer[i]])
        print("probability:", prob[answer[i]])

    print("========================")
    ind = np.argmax(model.forward(img).data.numpy())
    prob = model.forward(img).data.numpy()[0][ind]
    print("best candidate is: " + part10_act[ind] + " with prob: "+ str(prob))



    

