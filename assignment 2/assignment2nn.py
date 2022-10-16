# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:13:56 2022

@author: rjohn
"""


import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlroseh
import numpy as np
import random as rand
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()

file = r"wdbc.csv"

cancer = pd.read_csv(file, delimiter=',')

y = cancer.pop('id').values
y = cancer.pop('diagnosis').values
X = cancer.iloc[:,:].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size = 0.2, random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
















iterationHill = []
iterationHillTime = []
iterationHillTrain = []
iterationHillTest  = []

####Hill Climb#####
for i in range(1,5001,200):
    # Initialize neural network object and fit object
    if i != 1:
        i = i - 1
        
    nn_model1 = mlroseh.NeuralNetwork(hidden_nodes = [10,10,10], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = i, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     early_stopping = True, clip_max = 5, max_attempts = 200, \
                                     random_state = 3, curve=True, restarts = 3)
    
    start = time.time()
    nn_model1.fit(X_train_scaled, y_train_hot)
    stop = time.time()
    
    
    fitness_curve = nn_model1.fitness_curve
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)
    
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    
    
    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)
    
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    
    print("RHC Iteration: ", i, y_train_accuracy, y_test_accuracy)
    
    iterationHill.append(i)
    iterationHillTime.append(stop-start)
    iterationHillTrain.append(y_train_accuracy)
    iterationHillTest.append(y_test_accuracy)

plt.close()   
plt.title('RHC Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Accuracy') 
plt.plot(iterationHill,iterationHillTrain,label="Train")
plt.plot(iterationHill,iterationHillTest,label="Test")
plt.legend()
plt.show()
plt.close()

plt.close()   
plt.title('RHC Learning Time')
plt.xlabel('Iterations')
plt.ylabel('Training Time in Seconds') 
plt.plot(iterationHill,iterationHillTime,label="No Restarts")
plt.plot(iterationHill,iterationHillTime,label="3 Restarts")
plt.legend()
plt.show()
plt.close()


for i in range(0,10,3):
    restartsHill = []
    restartsHillTime = []
    restartsHillTrain = []
    restartsHillTest  = []
    
    for it in range(1,5001,200):
        if it != 1:
            it = it - 1
        nn_model1 = mlroseh.NeuralNetwork(hidden_nodes = [10,10,10], activation = 'relu', \
                                             algorithm = 'random_hill_climb', max_iters = it, \
                                             bias = True, is_classifier = True, learning_rate = 0.1, \
                                             early_stopping = True, clip_max = 5, max_attempts = 200, \
                                             random_state = 3, curve=True, restarts=i)
            
        start = time.time()
        nn_model1.fit(X_train_scaled, y_train_hot)
        stop = time.time()
        
        
        fitness_curve = nn_model1.fitness_curve
        
        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model1.predict(X_train_scaled)
        
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        
        
        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model1.predict(X_test_scaled)
        
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        
        print("RHC Restarts", i, it, y_train_accuracy, y_test_accuracy)
        
        restartsHill.append(it)
        restartsHillTime.append(stop-start)
        restartsHillTrain.append(y_train_accuracy)
        restartsHillTest.append(y_test_accuracy)
     
    plt.plot(restartsHill,restartsHillTrain,label=i)

plt.title('RHC Learning Curve: By # of Restarts')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()
plt.close()









iterationAnneal = []
iterationAnnealTime = []
iterationAnnealTrain = []
iterationAnnealTest  = []
####Annealing#####
# Initialize neural network object and fit object
for i in range(1,10001,500):
    nn_model1 = mlroseh.NeuralNetwork(hidden_nodes = [10,10,10], activation = 'relu', \
                                     algorithm = 'simulated_annealing', max_iters = i, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     early_stopping = True, clip_max = 5, max_attempts = 200, \
                                     random_state = 3, curve=True, schedule = mlroseh.GeomDecay())
    
    start = time.time()
    nn_model1.fit(X_train_scaled, y_train_hot)
    stop = time.time()
    
    
    fitness_curve = nn_model1.fitness_curve
    
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)
    
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    
    
    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)
    
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    
    print("SA Iteration: ", i, y_train_accuracy, y_test_accuracy)
    
    iterationAnneal.append(i)
    iterationAnnealTime.append(stop-start)
    iterationAnnealTrain.append(y_train_accuracy)
    iterationAnnealTest.append(y_test_accuracy)


plt.close()   
plt.title('SA Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Accuracy') 
plt.plot(iterationAnneal,iterationAnnealTrain,label="Train")
plt.plot(iterationAnneal,iterationAnnealTest,label="Test")
plt.legend()
plt.show()
plt.close()

plt.close()   
plt.title('Learning Time')
plt.xlabel('Iterations')
plt.ylabel('Training Time in Seconds') 
plt.plot(iterationAnneal,iterationAnnealTime,label="Train")
plt.legend()
plt.show()
plt.close()
    

temperatureAnneal = []
temperatureAnnealTime = []
temperatureAnnealTrain = []
temperatureAnnealTest  = []
####Annealing#####
# Initialize neural network object and fit object

temp = [.01,1,3]
for i in temp:
    
    temperatureAnneal = []
    temperatureAnnealTime = []
    temperatureAnnealTrain = []
    temperatureAnnealTest  = []
    
    for it in range(1,10001,500):
        if it != 1:
            it = it - 1
        nn_model1 = mlroseh.NeuralNetwork(hidden_nodes = [10,10,10], activation = 'relu', \
                                         algorithm = 'simulated_annealing', max_iters = it, \
                                         bias = True, is_classifier = True, learning_rate = 0.1, \
                                         early_stopping = True, clip_max = 5, max_attempts = 200, \
                                         curve=True, schedule = mlroseh.GeomDecay(init_temp=i, min_temp = .00099))
        
        start = time.time()
        nn_model1.fit(X_train_scaled, y_train_hot)
        stop = time.time()
        
        
        fitness_curve = nn_model1.fitness_curve
        
        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model1.predict(X_train_scaled)
        
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        
        
        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model1.predict(X_test_scaled)
        
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        
        print("SA Temperature: ", i, it, y_train_accuracy, y_test_accuracy)
        
        temperatureAnneal.append(it)
        temperatureAnnealTime.append(stop-start)
        temperatureAnnealTrain.append(y_train_accuracy)
        temperatureAnnealTest.append(y_test_accuracy)
    plt.plot(temperatureAnneal,temperatureAnnealTime,label=i)

plt.title('SA Learning Time')
plt.xlabel('Iterations')
plt.ylabel('Time in Seconds')
plt.legend()
plt.show()
plt.close()







iterationGA = []
iterationGATime = []
iterationGATrain = []
iterationGATest  = []

iterationGATr  = [0] * 20
iterationGATe  = [0] * 20
count = 0
countIter = True
####Genetic Algorithm#####
# Initialize neural network object and fit object
for z in range(0,5):
    iterationGA = []
    iterationGATime = []
    iterationGATrain = []
    iterationGATest  = []
    print("GA Iteration: ", z, y_train_accuracy, y_test_accuracy)
    for i in range(1,501,25):
        if i != 1:
            i -= 1
        nn_model1 = mlroseh.NeuralNetwork(hidden_nodes = [10,10,10], activation = 'relu', \
                                         algorithm = 'genetic_alg', max_iters = i, \
                                         bias = True, is_classifier = True, learning_rate = 0.1, \
                                         early_stopping = False, clip_max = 5, max_attempts = 200, \
                                         curve=True, pop_size = 100, mutation_prob=.01)
        
        start = time.time()
        nn_model1.fit(X_train_scaled, y_train_hot)
        stop = time.time()
        
        
        fitness_curve = nn_model1.fitness_curve
        
        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model1.predict(X_train_scaled)
        
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        
        
        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model1.predict(X_test_scaled)
        
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        
        
        if countIter:
            iterationGA.append(i)
        iterationGATime.append(stop-start)
        iterationGATrain.append(y_train_accuracy)
        iterationGATest.append(y_test_accuracy)
    
    countIter = False
    count += 1
    for x in range(0,20):
        if len(iterationGATrain) < x:
            iterationGATr[x]+=iterationGATrain[-1]
            iterationGATe[x]+=iterationGATest[-1]
        else:
            iterationGATr[x]+=iterationGATrain[x]
            iterationGATe[x]+=iterationGATest[x]

iterationGA = []        
for i in range(0,20):
    iterationGATr[i] = (iterationGATr[i]/count)
    iterationGATe[i] = (iterationGATe[i]/count)

for i in range(1,501,25):
    if i != 1:
        iterationGA.append(i-1)
    else:
        iterationGA.append(i-1)
    

    
    
plt.close()   
plt.title('GA Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Accuracy') 
plt.plot(iterationGA[2:],iterationGATr[2:],label="Train")
plt.plot(iterationGA[2:],iterationGATe[2:],label="Test")
plt.legend()
plt.show()
plt.close()

plt.close()   
plt.title('GA Learning Time')
plt.xlabel('Iterations')
plt.ylabel('Training Time in Seconds') 
plt.plot(iterationGA,iterationGATime,label="Train")
plt.legend()
plt.show()
plt.close()
    
    

####Genetic Algorithm#####
# Initialize neural network object and fit object
listOP = []
mut = [.001,.01,.1,.25]#,500,1000]
pop = [35, 100, 200, 500]
for i in pop:
    populationGA = []
    populationGATime = []
    populationGATrain = []
    populationGATest  = []
    for it in range(1,500,25):
        if it != 1:
            it = it - 1
        nn_model1 = mlroseh.NeuralNetwork(hidden_nodes = [10,10,10], activation = 'relu', \
                                         algorithm = 'genetic_alg', max_iters = it, \
                                         bias = True, is_classifier = True, learning_rate = 0.1, \
                                         early_stopping = True, clip_max = 5, max_attempts = it, \
                                         curve=True, pop_size = i, mutation_prob=.01)
        
        start = time.time()
        nn_model1.fit(X_train_scaled, y_train_hot)
        stop = time.time()
        
        
        fitness_curve = nn_model1.fitness_curve
        
        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model1.predict(X_train_scaled)
        
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        
        
        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model1.predict(X_test_scaled)
        
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        
        print("GA Population: ", i, it, y_train_accuracy, y_test_accuracy)
        
        populationGA.append(it)
        populationGATime.append(stop-start)
        populationGATrain.append(y_train_accuracy)
        populationGATest.append(y_test_accuracy)
    
    listOP.append((populationGA,populationGATrain,i))
    plt.plot(populationGA,populationGATime,label=i)

plt.title('GA Learning Curve: By Mutation Rate')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.legend(loc='upper left')
plt.show()
plt.close()

"""
for run in listOP:
    plt.plot(run[0],run[1],label=run[2])

plt.title('GA Learning Curve: By Population')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.legend(loc='upper left')
plt.show()
plt.close()

for run in listOT:
    plt.plot(run[0],run[1],label=run[2])

plt.title('GA Learning Time: By Mutation Rate')
plt.xlabel('Iterations')
plt.ylabel('Time in Seconds')
plt.legend(loc='upper left')
plt.show()
plt.close()


for run in listO:
    plt.plot(run[0],run[1],label=run[2])

plt.title('GA Learning Curve: By Mutation Rate')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.legend(loc='upper left')
plt.show()
plt.close()
"""


















