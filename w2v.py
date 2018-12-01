# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 10:49:53 2018

@author: user
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim.models
from gensim.models import word2vec 
import collections 
from collections import Counter
import matplotlib.pyplot  as plt
import random


df = pd.read_csv('777_QTR_after_data_preprocessing.csv')
FL =  list(map(str, df[:214429]['FLIGHTLEG_ID']))
date = list(df[:214429]['PERIOD_ENDDATE'])
mmsgs = list(df[:214429]['MESSAGE_CODE'])
fde =  list(map(str,df[:214429]['FDE_CODE']))

"""
corpus= []
for i in range(1,214429):
    corpus.append(mmsgs[i])
    if(fde[i]=='nan'):
        continue
    corpus.append(fde[i][:-2])
print(" ")


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(corpus)
print(" ")
print(tokenized_corpus)
print(len(tokenized_corpus))

num_features = 300    # Word vector dimensionality                      
min_word_count = 3    # 50% of the corpus                    
num_workers = 4       # Number of CPUs
window_size = 500
negative_sampling=5


model = word2vec.Word2Vec(tokenized_corpus, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = window_size, sample = negative_sampling)

print(" ")


print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
print(len(words))
#print(model[])
# save model
model.save('model.bin')
# load model

#finding the cosine similarity for each
i=0
pairs = []
length_w=len(words)
while i < length_w:
        pairs.append([words[i-1],words[i+1],words[i]])
        length_w -= 1
        i+=1
    
print(pairs)  
print(" ")   
 
count=0 
for i in range(len(pairs)):
    b = pairs[i][2]
    
    if int(pairs[i][0][:2]) == int(b[:2]):
        a = pairs[i][0]
        x = pairs[i][1]
    else:
        a = pairs[i][1]
        x = pairs[i][0]
    predicted = model.most_similar([x, b], [a])[0][0]
    if(x[2] == "-" and predicted[2] != "-" or x[2] != "-" and predicted[2] == "-" and a[2] == "-" and b[2] != "-" or  a[2] != "-" and  b[2] == "-"):
        print(" {} is to  {} as {} is to {} ".format(a, b, x, predicted))
    if int(x[:2]) == int(predicted[:2]):
        count+=1
       
#to find the success rate
success_rate = (count/len(pairs))*100
print("The success rate is {0} %".format(success_rate))        

print("********************************************************************************")
print("********************************************************************************")
"""
"""
#SW-MERIT SYSTEM
print("SW-MERIT")

mmsgs = list(df[:214429]['MESSAGE_CODE'])


fde =  list(map(str,df[:214429]['FDE_CODE']))
fdes = []
for i in range (0,len(fde)):
    if(fde[i]== "nan"):
        continue
    fdes.append(fde[i][:-2])

print(fdes)

#LIST OF MMSGS
def tokenize_corpus(mmsgs):
    tokens = [x.split() for x in mmsgs]
    return tokens

tokenized_corpus1 = tokenize_corpus(mmsgs)
M = [] #vocabulary of unique mmsgs and fdes
for sentence in tokenized_corpus1:
    for token in sentence:
        if token not in M:
            M.append(token)

print(M)
print(" ")


#LIST OF FDES
def tokenize_corpus(fdes):
    tokens = [x.split() for x in fdes]
    return tokens


tokenized_corpus2 = tokenize_corpus(fdes)
F = [] #vocabulary of unique mmsgs and fdes
for sentence in tokenized_corpus2:
    for token in sentence:
        if token not in F:
            F.append(token)
print(F)
print(" ")

corpus= [] #building sentence from each target fde
i = 1
j = 1
f_len = len(F)
m_len = len(M)
for i in range (0,f_len):
   for j in range (0,m_len):
        corpus.append([F[i] , M[j]])

print(corpus)


pos = []
neg = []
for i in range (0,len(F)):
    for j in range (0,len(M)):
        if(F[i][:2] == M[j][:2]):
            pos.append([F[i],M[j]])
        else:
            neg.append([F[i],M[j]])

print (pos)
from keras.preprocessing.sequence import skipgrams

skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid in wids]




num_features = 300    # Word vector dimensionality                      
min_word_count = 3    # 50% of the corpus                    
num_workers = 4       # Number of CPUs
window_size = 500
negative_sampling=5

print(" ")

model = word2vec.Word2Vec(corpus, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = window_size, sample = negative_sampling)

print(" ")


# summarize the loaded model
print(model)
# summarize vocabulary
words = list([(model.wv.vocab),(model.wv.vocab)])
print(words)
#print(model[])
# save model
model.save('model.bin')
# load model



#vector representation of AR1


print(" ")
print("vector representation of AR1")
skip_pairs = []
i = 1
word_length = len(words)
while i < word_length:
    for j in range(i+1, word_length):
        if (int(words[j][:2]) == int(words[i-1][:2])  or int(words[j][:2]) == int(words[i][:2])) and words[j][2] == '-' and words[i-1][2] != '-' and words[i][2] != '-':
            skip_pairs.append(([words[i-1],words[i]],words[j]))
            del words[j]
            word_length -= 1
            break
    i+=1
        
print("Prediction of mmsgs from fdes")  
print("[fi+fj-mj=mi]")      
print(skip_pairs)

count=0
for i in range(len(skip_pairs)):
    b = skip_pairs[i][2]
    
    if int(skip_pairs[i][0][:2]) == int(b[:2]):
        a = skip_pairs[i][0]
        x = skip_pairs[i][1]
    else:
        a = skip_pairs[i][1]
        x = skip_pairs[i][0]
    predicted = model.most_similar([x, b], [a])[0][0] #x + b - a
    if(predicted[2] == "-"):
        print(" {} is to  {} as {} is to {} ".format(a, b, x, predicted))
    if int(x[:2]) == int(predicted[:2]):
         count+=1
"""
"""       
#to find the success rate
success_rate = (count/len(skip_pairs))*100
print("The success rate is {0} %".format(success_rate))        

#finding the success rate for each chapter list
print(" ")
print(" SUCCESS RATES FOR CHAPTER LISTS")
labels = []
success=[]
counts = []
for k in range (20,80):
    count=0
    success_rate=0
    for i in range (len(skip_pairs)):
        b = skip_pairs[i][2]
    
        if int(skip_pairs[i][0][:2]) == int(b[:2]):
            a = skip_pairs[i][0]
            x = skip_pairs[i][1]
        else:
            a = skip_pairs[i][1]
            x = skip_pairs[i][0]
            predicted = model.most_similar([x, b], [a])[0][0]
            
        if(int(a[:2]) == k or int(x[:2]) == int(predicted[:2]) == k):
            count+=1
            success_rate = (count/len(skip_pairs))*100
            
        else:
           continue
    if(success_rate == 0):
        continue
    else:
        print("The success rate for CHAPTER-LIST {0}---->{1}%".format(k,success_rate))
        labels.append(k)
        success.append(success_rate)
print(labels)        
print(success)



i = 1
length = len(skip_pairs)
fde = []
mmsg = []

for i in range (len(skip_pairs)):
    b = skip_pairs[i][2]
    
    if int(skip_pairs[i][0][:2]) == int(b[:2]):
        a = skip_pairs[i][0]
        x = skip_pairs[i][1]
    else:
        a = skip_pairs[i][1]
        x = skip_pairs[i][0]
        predicted = model.most_similar([x, b], [a])[0][0]
                
    fde.append(a +" , "+ x)
    mmsg.append(b +" , " +predicted)
    
print(mmsg)
print(" ")
print(fde)


print("CO-OCCURENCE MATRIX")
labels = np.array(labels)
# # Initialise co-occurrence matrix
co_occurrence_matrix = np.zeros((len(fde), len(mmsg)))
for k in range (20,80):
    count=0
    for i in range (len(mmsg)):
        for j in range (len(fde)):
            if(int(fde[j][:2])==int(mmsg[i][:2])==k):
                count+=1
                co_occurrence_matrix[j][i] = count
        
co_occurrence_matrix = np.matrix(co_occurrence_matrix)

print(co_occurrence_matrix)

#Creation of a heatmap






print("________________________________________________________________________")

print("")
print("vector representation of AR2")
cbow_pairs = []
i = 1
word_length = len(words)
while i < word_length:
    for j in range(i+1, word_length):
        if (int(words[j][:2]) == int(words[i-1][:2]) or int(words[j][:2]) == int(words[i][:2])) and words[j][2] != '-'and words[i][2] == '-' and words[i-1][2] ==  '-':
            cbow_pairs.append([words[i-1],words[i],words[j]])
            del words[j]
            word_length -= 1
            break
    i+=1
        
print("Prediction of fde from mmsgs")        
print(cbow_pairs)

count=0
for i in range(len(cbow_pairs)):
    b = cbow_pairs[i][2]
    if int(cbow_pairs[i][0][:2]) == int(b[:2]):
        a = cbow_pairs[i][0]
        x = cbow_pairs[i][1]
    else:
        a = cbow_pairs[i][1]
        x = cbow_pairs[i][0]
    predicted = model.most_similar([x, b], [a])[0][0]
    if(predicted[2]!= "-"):
        print(" {} is to  {} as {} is to {} ".format(a, b, x, predicted))
    if int(x[:2]) == int(predicted[:2]):
         count+=1
 #to find the success rate   
success_rate = (count/len(cbow_pairs))*100

print("The success rate is {0} %".format(success_rate))           


#finding success rate for all chapter lists
#finding the success rate for each chapter list
print(" ")
print(" SUCCESS RATES FOR CHAPTER LISTS")
labels=[]
success=[]
for k in range (20,80):
    count=0
    success_rate=0
    for i in range (len(cbow_pairs)):
        b = cbow_pairs[i][2]
    
        if int(cbow_pairs[i][0][:2]) == int(b[:2]):
            a = cbow_pairs[i][0]
            x = cbow_pairs[i][1]
        else:
            a = cbow_pairs[i][1]
            x = cbow_pairs[i][0]
            predicted = model.most_similar([x, b], [a])[0][0]
            
        if(int(a[:2]) == k or int(x[:2]) == int(predicted[:2]) == k):
            count+=1
            success_rate = (count/len(cbow_pairs))*100
        else:
           continue
    if(success_rate == 0):
        continue
    else:
        print("The success rate for CHAPTER-LIST {0}---->{1}%".format(k,success_rate))
        labels.append(k)
        success.append(str(success_rate))  
print(labels)       
print(success)
        
print("_________________________________________________________________________")

i = 1
length = len(cbow_pairs)
fde1 = []
mmsg1 = []

for i in range (len(cbow_pairs)):
    b = cbow_pairs[i][2]
    
    if int(cbow_pairs[i][0][:2]) == int(b[:2]):
        a = cbow_pairs[i][0]
        x = cbow_pairs[i][1]
    else:
        a = cbow_pairs[i][1]
        x = cbow_pairs[i][0]
        predicted = model.most_similar([x, b], [a])[0][0]
                
    mmsg1.append(a +" , "+ x)
    fde1.append(b +" , " +predicted)
    



print("CO-OCCURENCE MATRIX")
labels = np.array(labels)
# # Initialise co-occurrence matrix
co_occurrence_matrix1 = np.zeros((len(fde1), len(mmsg1)))
for k in range (20,80):
    count=0
    for i in range (len(mmsg1)):
        for j in range (len(fde1)):
            if(int(fde1[j][:2])==int(mmsg1[i][:2])==k):
                count+=1
                co_occurrence_matrix1[j][i] = count
        
co_occurrence_matrix1 = np.matrix(co_occurrence_matrix1)

print(co_occurrence_matrix1)

print("________________________________________________________________________________")

print("TOP 5 PREDICTIONS FOR THE MMSGS AND FDES")
print("ALGEBRAIC REPRESENTATION")

"""




