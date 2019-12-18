# -*- coding: utf-8 -*-
"""
Last edited on Jun 27 2019

"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

#some pre-prossesing with the text data
print("Importing training set and performing pre-prossess...")
validation_on = True
path = r"E:\Programming\_Program\_Example,Portfolio\CS484\Hw1"

text = []
labels = []

X_train = [] 
X_Val = []
y_train = []
y_Val = []
X_test = []

#importing data
print('Importing text')  
Data = pd.read_fwf(path+'/train.dat',names=['Labels','Text'],widths=[2,5000], header=None)

X_train = Data['Text'].tolist()
text = Data['Text'].tolist()
labels = Data['Labels'].tolist()

if(validation_on):
    X_train, X_test, y_train, y_true = split(X_train, labels, test_size=0.3)
else:
    X_test = pd.read_fwf(path+'/test.dat',names=['Text'], header=None)
print('Done importing text')    

print('Building BOW')  
#Build bag of words with vectorizer
vectorizer = TfidfVectorizer(binary=True,
                             max_df=0.3,min_df=int(3),
                             stop_words ='english')

train_tfidf = vectorizer.fit_transform(X_train)
test_tfidf = vectorizer.transform(X_test) 

#create cosine similarity matrix
Sim = cosine_similarity(test_tfidf,train_tfidf)

k = int(0.0075 * len(X_train)) #best k value based on X_train size

#used to test different values ,d: dimension reduction, k: k neighbors
test_d = [0]
test_k = [k]

print('Done building BOW')  

for d in test_d:               
    if(d > 0):
        SVD = TruncatedSVD(d)
        train_svd = SVD.fit_transform(train_tfidf)
        test_svd = SVD.transform(test_tfidf)
        sim = cosine_similarity(test_svd,train_svd)
    else:
        sim = Sim.copy()
    try:  
        f = open("output_k_"+str(k)+"_d_"+str(d)+".txt", "w") #saving output
        
        y_pred =[]
        
        print("Classifying with d:%d k:%d" %(d,k))    
        for item in sim:                         
            #Sort indices to get largest similarity
            Sorted_indices = (-item).argsort(axis=0)
 
            for k in test_k:        
                predict = 0
                for i in range(0,k):
                    predict += y_train[Sorted_indices[i]] * item[Sorted_indices[i]]
            
                if predict> 0:
                    y_pred.append(1)
                else:
                    y_pred.append(-1)
                    
        score = accuracy_score(y_true, y_pred)   
        print("Score with k:%d d:%d is:%f " %(k,d,score))
    finally:
        for i in y_true:
            if (i > 0):
                f.write('+1\n')    
            else:
                f.write('-1\n')
        f.close()
    