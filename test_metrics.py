import matplotlib.pyplot as plt
import numpy as np
import math


def metrics(y_score, y_target, threshold=0.4):
    TP = TN = FP = FN = 0

    for i in range(len(y_score)):
        score = y_score[i]
        target = y_target[i]

        if y_score==-1 or y_score==-3:
            continue

        else:
            if score < threshold and target == 1 :
                TP = TP + 1
            elif score > threshold and target == 1 :
                FN = FN + 1
            elif score < threshold and target == 0 :
                FP = FP + 1
            elif score > threshold and target == 0 :
                TN = TN + 1

    return TP, TN, FP, FN

def fscore(y_score, y_target, threshold=0.4):
    TP, TN, FP, FN = metrics(y_score, y_target, threshold)
    if TP + FP == 0:
        return math.inf
    else:
        fsc = TP / (TP+0.5*(TP + FP))
        return fsc   
    return 


def precision(y_score, y_target, threshold=0.4):
    TP, TN, FP, FN = metrics(y_score, y_target, threshold)
    if TP + FP == 0:
        return math.inf
    else:
        prec = TP / (TP + FP)
        return prec


def recall(y_score, y_target, threshold=0.4) :
    TP, TN, FP, FN = metrics(y_score, y_target, threshold)
    if TP + FN == 0:
        return math.inf
    else:
        rcll = TP / (TP + FN)
        return rcll


def accuracy(y_score, y_target, threshold=0.4) :
    TP, TN, FP, FN = metrics(y_score, y_target, threshold)
    acc = (TP + TN)/(TP + TN + FP + FN)
    return acc

def FPR(y_score, y_target, threshold=0.4) :
    TP, TN, FP, FN = metrics(y_score, y_target, threshold)
    fpr = (FP)/(TN + FP)
    return fpr

def error_rate(y_score, y_target, threshold=0.4) :
    TP, TN, FP, FN = metrics(y_score, y_target, threshold)
    error_rate = (FP+FN)/(TP + TN + FP + FN)
    return error_rate

def fallout(y_score, y_target, threshold=0.4) :
    TP, TN, FP, FN = metrics(y_score, y_target, threshold)
    fallout = (TN)/(TN + FP)
    return fallout


def precision_recall_curve(y_score, y_target) :

    prcsn = []
    rcll = []
    steplist = []
    step = 0.01
    thresh = np.arange(0, 1.0, step)
    for i in thresh:
        pr = precision(y_score, y_target, threshold=i)
        rc = recall(y_score, y_target, threshold=i)
        if math.isinf(pr) or math.isinf(rc):
            continue
        else:
            prcsn.append(pr)
            rcll.append(rc)
            steplist.append(i)

    print(prcsn[42])
    print(rcll[42])
    print(steplist[42])

    plt.plot(rcll, prcsn, 'go--',color ='cyan', linewidth=2, markersize=3)
    
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Dlib Precision vs Recall')     
    i=0
    for x,y in zip(rcll,prcsn):
    
        label = "{:.2f}".format(steplist[i])
    
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(15,15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center       
        i+=1
    plt.show()
    return prcsn.index(max(prcsn))*step

#Specially written function to produce the PRC for OpenCV

def precision_recall_curve_cv(y_score, y_target, maxi) :

    prcsn = []
    rcll = []
    step = 0.1
    steplist = []
    thresh = np.arange(0,int(maxi + 1), step)
    for i in thresh:
        pr = precision(y_score, y_target, threshold=i)
        rc = recall(y_score, y_target, threshold=i)
        if math.isinf(pr) or math.isinf(rc):
            continue
        else:
            prcsn.append(pr)
            rcll.append(rc)
            steplist.append(i)
    
    plt.plot(rcll, prcsn, 'go--',color ='black', linewidth=2, markersize=3)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('OpenCV Precision vs Recall')     
    i=0
    for x,y in zip(rcll,prcsn):
    
        label = "{:.2f}".format(steplist[i])
    
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(15,15), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center       
        i+=1      
    plt.show()

    return prcsn.index(max(prcsn))*step