from pylab import *
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import tree
from stop_words import *
import cPickle
import os

def build_sets():
    clean_fake = "clean_fake.txt"
    clean_real = "clean_real.txt"
    
    fake_news = []
    real_news = []
    
    f_fake = open(clean_fake)
    f_real = open(clean_real)
    
    lines_fake = f_fake.readlines()
    lines_real = f_real.readlines()
    
    for line in lines_fake:
        fake_news.append(str.split(line))
    
    for line in lines_real:
        real_news.append(str.split(line))
    
    np.random.seed(20)
    np.random.shuffle(fake_news)
    np.random.shuffle(real_news)
    
    train_fake = int(len(fake_news)*0.7)
    valid_fake = int(len(fake_news)*0.85)
    test_fake  = len(fake_news)
    
    train_real = int(len(real_news)*0.7)
    valid_real = int(len(real_news)*0.85)
    test_real  = len(real_news)
    
    train_set = fake_news[0:train_fake]
    train_set = train_set + real_news[0:train_real]
    
    valid_set = fake_news[train_fake: valid_fake]
    valid_set = valid_set + real_news[train_real: valid_real]
    
    test_set = fake_news[valid_fake: test_fake]
    test_set = test_set + real_news[valid_real: test_real]
    
    train_target = [0] * train_fake
    train_target = train_target + ([1] * train_real)
    
    valid_target = [0] * (valid_fake - train_fake)
    valid_target = valid_target + ([1]* (valid_real - train_real))
    
    test_target = [0] * (test_fake - valid_fake)
    test_target = test_target + ([1]* (test_real - valid_real))
    
    return train_set, valid_set, test_set, train_target, valid_target, test_target, train_fake, train_real

def words_appearance(train_set,train_target):
    # dict for every words appear in train_set of fake news
    dict_fake_appear = {}
    # dict for every words appear in train_set of real news
    dict_real_appear = {}
    
    for i in range(len(train_set)):
        news = train_set[i]
        no_dup_news = list(set(news))
        no_dup_news.sort(key=news.index)
        if train_target[i] == 0:
            for word in no_dup_news:
                if word not in dict_fake_appear:
                    dict_fake_appear[word] = 1
                else:
                    dict_fake_appear[word] += 1
        else:
            for word in no_dup_news:
                if word not in dict_real_appear:
                    dict_real_appear[word] = 1
                else:
                    dict_real_appear[word] += 1
    
    return dict_fake_appear, dict_real_appear

def naive_bayes(headline, m, p_hat, dict_fake_appear, dict_real_appear,pfake,preal,f_count, r_count):
    line = list(set(headline))
    line.sort(key=headline.index)
    probs_fake = []
    probs_real = []
    for word in line:
        if word not in dict_fake_appear:
            prob_fake = (float(m* p_hat)/float(f_count + m))
        else:
            prob_fake = (float(dict_fake_appear[word] + m* p_hat)/float(f_count + m))

        if word not in dict_real_appear:
            prob_real = (float(m* p_hat)/float(r_count + m))
        else:
            prob_real = (float(dict_real_appear[word] + m* p_hat)/float(r_count + m))

        probs_fake.append(math.log(prob_fake))
        probs_real.append(math.log(prob_real))
    
    for word in dict_fake_appear:
        if word not in line:
            prob_fake = 1 - (float(dict_fake_appear[word] + m* p_hat)/float(f_count + m))
        probs_fake.append(math.log(prob_fake))
    for word in dict_real_appear:
        if word not in line:
            prob_real = 1 - (float(dict_real_appear[word] + m* p_hat)/float(r_count + m))
        probs_real.append(math.log(prob_real))

    p_fake = math.exp(sum(probs_fake))*pfake
    p_real = math.exp(sum(probs_real))*preal

    if p_fake > p_real:
        return 0
    else:
        return 1

def part2_rate(train_set,train_target, valid_set,valid_target,m,p_hat,pfake,preal,f_count,r_count):

    dict_fake_appear, dict_real_appear = words_appearance(train_set,train_target)
    result = 0
    for i in range(len(valid_target)):
        line = valid_set[i]
        res = naive_bayes(line,m,p_hat,dict_fake_appear,dict_real_appear,pfake,preal,f_count,r_count)
        if res == valid_target[i]:
            result += 1
    
    res = float(result)/float(len(valid_target))
    res = float(format(res, '.3f'))
    
    return res      


def part3_absence(dict_real_appear, dict_fake_appear,train_fake, train_real, train_set,m,p_hat):
    word_absence_fake = {}
    word_absence_real = {}
    word_presence_fake = {}
    word_presence_real = {}    
    pfake = float(train_fake)/float(train_fake + train_real)
    preal = float(train_real)/float(train_fake + train_real)
    for headline in train_set:
        for word in headline:
            if word not in word_absence_fake:
                if word not in dict_fake_appear:
                    prob_noword_fake = 1 - ((float(m* p_hat)/float(train_fake + m)))
                    prob_word_fake = ((float(m* p_hat)/float(train_fake + m)))
                else:
                    prob_noword_fake = 1 - (float(dict_fake_appear[word] + m* p_hat)/float(train_fake + m))
                    prob_word_fake = (float(dict_fake_appear[word] + m* p_hat)/float(train_fake + m))
                if word not in dict_real_appear:
                    prob_noword_real = 1 - (float(m* p_hat)/float(train_real + m))
                    prob_word_real = (float(m* p_hat)/float(train_real + m))
                else:
                    prob_noword_real = 1 - (float(dict_real_appear[word] + m* p_hat)/float(train_real + m))
                    prob_word_real = (float(dict_real_appear[word] + m* p_hat)/float(train_real + m))
                    
                prob_word = prob_word_fake*pfake + prob_word_real*preal
                prob_noword = 1 - prob_word
                
                prob_word_fake_ = float(prob_word_fake * pfake)/float(prob_word)
                prob_word_real_ = float(prob_word_real * preal)/float(prob_word)
                prob_noword_fake_ = float(prob_noword_fake * pfake)/float(prob_noword)
                prob_noword_real_ = float(prob_noword_real * preal)/float(prob_noword)

                word_presence_fake[word] = float(format(prob_word_fake_, '.5f'))
                word_presence_real[word] = float(format(prob_word_real_, '.5f'))
                word_absence_fake[word] = float(format(prob_noword_fake_, '.5f'))
                word_absence_real[word] = float(format(prob_noword_real_, '.5f'))
    most_presense_real = sorted(word_presence_real.items(), key=lambda x:x[1], reverse=True)
    most_presense_fake = sorted(word_presence_fake.items(), key=lambda x:x[1], reverse=True)
    most_absense_fake = sorted(word_absence_fake.items(), key=lambda x:x[1], reverse=True)
    most_absense_real = sorted(word_absence_real.items(), key=lambda x:x[1], reverse=True)
    most_presense_real_STOP = {}
    most_presense_fake_STOP = {}
    most_absense_real_STOP = {}
    most_absense_fake_STOP = {}
    
    for word in word_presence_fake:
        if word not in ENGLISH_STOP_WORDS:
            most_presense_fake_STOP[word] = word_presence_fake[word]
    for word in word_presence_real:
        if word not in ENGLISH_STOP_WORDS:
            most_presense_real_STOP[word] = word_presence_real[word]
    for word in word_absence_real:
        if word not in ENGLISH_STOP_WORDS:
            most_absense_real_STOP[word] = word_absence_real[word]
    for word in word_absence_fake:
        if word not in ENGLISH_STOP_WORDS:
            most_absense_fake_STOP[word] = word_absence_fake[word]
    
    most_presense_fake_STOP_ = sorted(most_presense_fake_STOP.items(),key=lambda x:x[1], reverse=True)
    most_presense_real_STOP_ = sorted(most_presense_real_STOP.items(),key=lambda x:x[1], reverse=True)
    most_absense_fake_STOP_ = sorted(most_absense_fake_STOP.items(),key=lambda x:x[1], reverse=True)
    most_absense_real_STOP_ = sorted(most_absense_real_STOP.items(),key=lambda x:x[1], reverse=True)  

    print "most_presense_real_words: " + str(most_presense_real[0:10])
    print "\n"
    print "most_presense_fake_words: " + str(most_presense_fake[0:10])
    print "\n"
    print "most_absense_real_words: " + str(most_absense_real[0:10])
    print "\n"
    print "most_absense_fake_words: " + str(most_absense_fake[0:10]) 
    print "\n"
    print("Part3 b : 10 words whose presence/absense most strongly \npredicts that the news is real/fake\n") 
    print "most_presense_real_words_without_STOP: " + str(most_presense_real_STOP_[0:10])
    print "\n"
    print "most_presense_fake_words_without_STOP: " + str(most_presense_fake_STOP_[0:10])
    print "\n"
    print "most_absense_real_words_without_STOP: " + str(most_absense_real_STOP_[0:10])
    print "\n"
    print "most_absense_fake_words_without_STOP: " + str(most_absense_fake_STOP_[0:10])    
    print "\n"
    

def get_total_words_with_STOP(train_set, valid_set, test_set):
    total_words = []
    for headline in train_set + valid_set + test_set:
        for word in headline:
            if word not in total_words:
                total_words.append(word)
    
    return total_words


def build_np_set(set_t,total_words):
    np_set = np.zeros((len(set_t), len(total_words)))
    for line in np_set:
        line[0] = 1
    i = 0
    for headline in set_t:
        for word in headline:
            if word in total_words:
                np_set[i][total_words.index(word)] = 1
        i += 1
    return np_set
##



real_set, fake_set = [], []
training_x, validation_x, testing_x = [], [], []
training_y, validation_y, testing_y = [], [], []
FAKE_SET_SIZE = 1298
REAL_SET_SIZE = 1968
word_index_map = {}

def build_sets_p4():
    
    f = open("clean_fake.txt")
    fake_titles = f.readlines()

    for title in fake_titles:
        fake_set.append(str.split(title))

    f = open("clean_real.txt")
    real_titles = f.readlines()

    for title in real_titles:
        real_set.append(str.split(title))

    np.random.seed(10)
    np.random.shuffle(fake_set)
    np.random.shuffle(real_set)

    
    for i in range(FAKE_SET_SIZE):
        if i < FAKE_SET_SIZE * 0.7:
            training_y.append(0)
            training_x.append(fake_set[i])
        elif i < FAKE_SET_SIZE * 0.85:
            validation_y.append(0)
            validation_x.append(fake_set[i])
        else:
            testing_y.append(0)
            testing_x.append(fake_set[i])

    for i in range(REAL_SET_SIZE):
        if i < REAL_SET_SIZE * 0.7:
            training_y.append(1)
            training_x.append(real_set[i])
        elif i < REAL_SET_SIZE * 0.85:
            validation_y.append(1)
            validation_x.append(real_set[i])
        else:
            testing_y.append(1)
            testing_x.append(real_set[i])
            
    complete_set = [] 
    index = 0
    for i in range(FAKE_SET_SIZE):
        complete_set.append(fake_set[i])
    for j in range(REAL_SET_SIZE):
        complete_set.append(real_set[j])

    for k in range(len(complete_set)):
        for word in complete_set[k]:
            if word not in word_index_map: 
                word_index_map[word] = index
                index = index + 1
                
build_sets_p4()

TRAIN_SET_SIZE = len(training_x)
VALID_SET_SIZE = len(validation_x)
TEST_SET_SIZE = len(testing_x)

WORDS_INDEX = word_index_map
TOTAL_WORDS_LEN = len(WORDS_INDEX)
    

def set_converter_x(target_set, word_index_dict):
    set_np = np.zeros((0, TOTAL_WORDS_LEN))
    for title in target_set:
        vector = np.zeros(TOTAL_WORDS_LEN)

        for word in title:
            vector[word_index_dict[word]] = 1
        vector = np.reshape(vector, [1, TOTAL_WORDS_LEN])
        set_np = np.vstack((set_np, vector))
        
    return set_np
        

def set_converter_y(y):
    label_np = np.asarray(y).transpose()
    label_np_complement = 1 - label_np
    label_np = np.vstack((label_np, label_np_complement)).transpose()
    
    return label_np
    
    
    
    
def create_set():

    train_set = np.zeros((0, TOTAL_WORDS_LEN))
    valid_set = np.zeros((0, TOTAL_WORDS_LEN))
    testing_set = np.zeros((0, TOTAL_WORDS_LEN))

    train_set = set_converter_x(training_x, WORDS_INDEX)
    train_label = set_converter_y(training_y)
    valid_set = set_converter_x(validation_x, WORDS_INDEX)
    valid_label = set_converter_y(validation_y)
    testing_set = set_converter_x(testing_x, WORDS_INDEX)
    testing_label = set_converter_y(testing_y)

    return train_set, valid_set, testing_set, train_label, valid_label, testing_label




class logistic_regression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out



def part4_model():
    training_set, validation_set, test_set, training_label, validation_label, test_label = create_set()

    num_epochs = 1000
    learning_rate = 6e-4

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    
    model = logistic_regression(TOTAL_WORDS_LEN, 2)

    LossFunction = nn.CrossEntropyLoss()  
    Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    X = Variable(torch.from_numpy(training_set), requires_grad=False).type(dtype_float)
    Y = Variable(torch.from_numpy(np.argmax(training_label, 1)), requires_grad=False).type(dtype_long)

    #reg_lambda = 0.01
    lambdas = [0.01]#[0.1, 0.05, 0.01, 0.005, 0.001]
    lambda_performance = []
    for reg_lambda in lambdas:
        
        tr, va, te =  [],[],[]
        for epoch in range(num_epochs):
    
            Optimizer.zero_grad()
            outputs = model(X)
            l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
    
            for W in model.parameters():
                l2_reg = l2_reg + W.norm(2)
                    
            l = LossFunction(outputs, Y) + reg_lambda * l2_reg
            l.backward()
            Optimizer.step()
            
            
            if epoch % 50 == 0:
                print("epoch:" + str(epoch))
    
                TRAIN_X = Variable(torch.from_numpy(training_set), requires_grad=False).type(dtype_float)
                pred = model(TRAIN_X).data.numpy()
                performance = (np.mean(np.argmax(pred, 1) == np.argmax(training_label, 1)))
                print("train performance:" + str(100 * performance))
                tr.append(performance)      
    
                VALID_X = Variable(torch.from_numpy(validation_set), requires_grad=False).type(dtype_float)
                pred = model(VALID_X).data.numpy()
                performance = (np.mean(np.argmax(pred, 1) == np.argmax(validation_label, 1)))
                print("validation performance:" + str(100 * performance))
                va.append(performance)
                
                TEST_X = Variable(torch.from_numpy(test_set), requires_grad=False).type(dtype_float)
                pred = model(TEST_X).data.numpy()
                performance = (np.mean(np.argmax(pred, 1) == np.argmax(test_label, 1)))
                print("test performance:" + str(100 * performance) + "\n")
                te.append(performance)
        lambda_performance.append(va)
    epoch_num = []
    for i in range(0,20):
        epoch_num.append(50*i)
        
    plt.switch_backend('agg')
    cx = epoch_num
    plt.plot(cx, tr, 'r', cx, va, 'y', cx, te, 'g')
    plt.savefig("part4.jpg")
    
    train_plot, = plt.plot(cx, tr, color='r')
    valid_plot, = plt.plot(cx, va, color='y')
    test_plot, = plt.plot(cx, te, color='g')
    
    plt.legend([train_plot, valid_plot, test_plot], ['train', 'validation', 'test'])
    plt.title('Part 4 Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.savefig("part4.jpg")
    
    #print(str(len(lambda_performance[0])))
    # plot1,= plt.plot(cx, lambda_performance[0], color='r')
    # plot2,= plt.plot(cx, lambda_performance[1], color='g')
    # plot3,= plt.plot(cx, lambda_performance[2], color='b')
    # plot4,= plt.plot(cx, lambda_performance[3], color='y')
    # plot5,= plt.plot(cx, lambda_performance[4], color='c')
    # 
    # plt.legend([plot1, plot2, plot3,plot4,plot5], ['0.1', '0.05', '0.01', '0.005', '0.001'])
    # plt.title('Part 4 Lambda')
    # plt.xlabel('Iterations')
    # plt.ylabel('Performance')
    # plt.savefig("part4_lambda.jpg")
    
    return model


def Part4():
    model_p4 = part4_model()
    #torch.save(model_p4, "model_p4.pkl")
    np.savetxt("model_p4.txt", model_p4.linear.weight.data.numpy())


def select_top_words(max, stopwords, max_theta, min_theta, max_theta_nonstop, min_theta_nonstop):
    if max == True:
        if stopwords == True:
            theta_used = max_theta
        else:
            theta_used = max_theta_nonstop
    else:
        if stopwords == True:
            theta_used = min_theta
        else:
            theta_used = min_theta_nonstop
            
    if stopwords == True:
        count = 0
        for theta in theta_used:
            for W, index in word_index_map.items():
                if index == theta:
                    print("{0} : {1}".format(str(1 + count), W))
            count += 1
    else:
        count = 0
        for theta in theta_used:
            for W, index in word_index_map.items():
                if index == theta:
                    break
            
            if W in ENGLISH_STOP_WORDS:
                continue
            
            print("{0} : {1}".format(str(1 + count), W))
            count += 1
            if count == 10: break
        
    
def Part6():
    p4_weights = np.loadtxt("model_p4.txt")
    W = p4_weights
    W[1] = -W[1]
    W = 2 * np.mean(W, axis=0)
    
    W2 = W.copy()
    theta_max, theta_min = [], []
    theta_max_nonstop, theta_min_nonstop = [], []
    
    for i in range(10):
        theta_max.append(np.argmax(W))
        theta_min.append(np.argmin(W))
        W[theta_max[-1]] = 0
        W[theta_min[-1]] = 0
        
    for i in range(10 + len(ENGLISH_STOP_WORDS)):
        theta_max_nonstop.append(np.argmax(W2))
        theta_min_nonstop.append(np.argmin(W2))
        W2[theta_max_nonstop[-1]] = 0
        W2[theta_min_nonstop[-1]] = 0
    
    print("\nmax thetas include stopwords")
    select_top_words(True, True, theta_max, theta_min, theta_max_nonstop, theta_min_nonstop)
    
    print("\nmin thetas include stopwords")
    select_top_words(False, True, theta_max, theta_min, theta_max_nonstop, theta_min_nonstop)
    
    print("\nmax thetas exclude stopwords")
    select_top_words(True, False, theta_max, theta_min, theta_max_nonstop, theta_min_nonstop)
    
    print("\nmin thetas exclude stopwords")
    select_top_words(False, False, theta_max, theta_min, theta_max_nonstop, theta_min_nonstop)

##
def plot_curve(valid_rate,max_depth):
    pic_name = 'part7_max_depth_curve'
    plt.plot(max_depth, valid_rate, 'b', label = 'validation_set_rate')
    plt.legend(loc = 'lower right')
    title('part7_max_depth_curve')
    plt.savefig(pic_name)
    show()
    plt.close()

def part7(train_set,valid_set,test_set,train_target,valid_target,test_target,input_max_depth):
    # Training the decision tree do not need exclude the STOP WORDS.
    total_words = get_total_words_with_STOP(train_set,valid_set,test_set)
    total_words.insert(0,"bias")
    np_train_set = build_np_set(train_set,total_words)
    np_valid_set = build_np_set(valid_set,total_words)
    np_test_set  = build_np_set(test_set, total_words)
    np_train_target = np.array(train_target)
    np_valid_target = np.array(valid_target)
    np_test_target  = np.array(test_target)
    np_test_target  =np_test_target[:,np.newaxis]
    
    max_depth_choice = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
    valid_rate = []
    train_rate = []
    test_rate = []
    for depth in max_depth_choice:

        # model_graph = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth, min_samples_leaf = 4)
        model_graph = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        model_graph = model_graph.fit(np_train_set, np_train_target)
        score_graph = model_graph.score(np_valid_set,np_valid_target)
        train_graph = model_graph.score(np_train_set,np_train_target)
        test_graph  = model_graph.score(np_test_set ,np_test_target)
        valid_rate.append(score_graph)
        train_rate.append(train_graph)
        test_rate.append(test_graph)
        print "depth is "+ str(depth) + " , trainning: " + str(train_graph)
        print "depth is "+ str(depth) + " , validation: " + str(score_graph)
        print "depth is "+ str(depth) + " , test: " + str(test_graph)
        if depth == 120:
            visual = tree.export_graphviz(model_graph, out_file = 'model_120.dot', max_depth= 2, filled= True, class_names = ['fake', 'real'],feature_names=total_words )


        
    # plot_curve(valid_rate, max_depth_choice)
    #plot_curve(valid_rate,train_rate,test_rate,max_depth_choice)
    
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=input_max_depth, min_samples_leaf=4, random_state = 10)
    model = model.fit(np_train_set, np_train_target) 
    para = model.get_params(1)
    decision = model.decision_path(np_train_set)
    feature = model.feature_importances_
    x2 = 0
    y2 = 0
    
    for i in range(len(np_train_set)):
        sample = np_train_set[i]
        pre = model.predict(sample[np.newaxis,:])
        if pre  == [0]:
            y2 += 1
        else:
            x2 += 1
    
    train_rate = model.score(np_train_set,np_train_target)
    valid_rate = model.score(np_valid_set,np_valid_target)
    test_rate  = model.score(np_test_set, np_test_target)
    
    visual = tree.export_graphviz(model, out_file = 'model_max.dot', max_depth= 2, filled= True, class_names = ['fake', 'real'],feature_names=total_words )


    return 0

def part8(x1,y1,x2,y2,x3,y3):

    prob_x1_x1y1 = float(x1)/float(x1+y1)
    prob_y1_x1y1 = float(y1)/float(x1+y1)
    h_y = (-prob_x1_x1y1*math.log(prob_x1_x1y1,2)) - (prob_y1_x1y1*math.log(prob_y1_x1y1,2))
    
    prob_real = float(x2+y2)/float(x2+y2+x3+y3)
    prob_fake = float(x3+y3)/float(x2+y2+x3+y3)
    
    
    prob_2_real = float(x2)/float(x2+y2)
    prob_3_real = float(x3)/float(x3+y3)
    
    prob_2_fake = float(y2)/float(x2+y2)
    prob_3_fake = float(y3)/float(x3+y3)
    
    if x2 == 0 or y2 == 0:
        h_y_real = 0
    else:
        h_y_real = (-prob_2_real*math.log(prob_2_real,2)) - (prob_2_fake*math.log(prob_2_fake,2))
    
    if x3 == 0 or y3 == 0:
        h_y_fake = 0
    else:
        h_y_fake = (-prob_3_real*math.log(prob_3_real,2)) - (prob_3_fake*math.log(prob_3_fake,2))
    
    mut_info = h_y - (prob_real*h_y_real + prob_fake* h_y_fake)
    
    return mut_info

def get_split_result(word, train_set,target_set):
    apper_fake = 0
    apper_real = 0
    disapper_fake = 0
    disapper_real = 0
    i = 0
    for headline in train_set:
        if word in headline:
            if target_set[i] == 0:
                apper_fake += 1
            else:
                apper_real += 1
        else:
            if target_set[i] == 0:
                disapper_fake += 1
            else:
                disapper_real += 1
        i += 1
    return apper_fake, apper_real, disapper_fake, disapper_real
            

def part2(train_set, valid_set, test_set, train_target, valid_target, test_target, train_fake, train_real,m,p_hat):
    
    pfake = float(train_fake)/(float(train_fake+train_real))
    preal = float(train_real)/(float(train_fake+train_real))
    dict_fake_appear, dict_real_appear = words_appearance(train_set,train_target)
    total_words = get_total_words_with_STOP(train_set,valid_set,test_set)

    accurate_valid = part2_rate(train_set,train_target, valid_set,valid_target,m,p_hat,pfake,preal,train_fake,train_real)
    accurate_train = part2_rate(train_set,train_target, train_set,train_target,m,p_hat,pfake,preal,train_fake,train_real)
    accurate_test  = part2_rate(train_set,train_target, test_set, test_target, m,p_hat,pfake,preal,train_fake,train_real)
    print "Part2 : Naive Bayes performane: "
    print "Train performance: " + str(accurate_train)
    print "Valid performance: " + str(accurate_valid)
    print "Test performance : " + str(accurate_test)
    print "\n"
    
    return 0
    
def part3_output(train_set,train_target,train_fake,train_real,):
    dict_fake_appear, dict_real_appear = words_appearance(train_set,train_target)
    print("Part3 a : 10 words whose presence/absense most strongly \npredicts that the news is real/fake\n") 
    part3_absence(dict_real_appear, dict_fake_appear,train_fake, train_real, train_set,2,0.4)

def part8_output(train_set,train_target,new_word):
    part8a = part8(1377,908,789,748,588,160)
    apper_fake, apper_real, disapper_fake, disapper_real = get_split_result(new_word,train_set,train_target)
    part8b = part8(1377,908,apper_real,apper_fake,disapper_real,disapper_fake)
    print "Part8a: the mutual information is : " + str(part8a)
    print "Part8b: the mutual information for " + new_word + " is : " + str(part8b)

train_set, valid_set, test_set, train_target, valid_target, test_target, train_fake, train_real = build_sets()
part2(train_set, valid_set, test_set, train_target, valid_target, test_target, train_fake, train_real,2,0.4)
part3_output(train_set, train_target,train_fake,train_real)
Part4()
Part6()
part7(train_set,valid_set,test_set,train_target, valid_target, test_target,120)
part8_output(train_set,train_target,"trump")
















    