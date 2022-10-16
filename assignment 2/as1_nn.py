import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


#cancer basic
performance = []
file = r"wdbc.csv"
cancer = pd.read_csv(file, delimiter=',')

file = r"winequality-red.csv"
wine = pd.read_csv(file, delimiter=';')

wine['highQuality'] = np.where(wine['quality'] >= 7, 1, 0)
y = wine.pop('highQuality').values
X = wine.iloc[:,:10].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

mlp = MLPClassifier(activation='relu', solver='adam', max_iter=50, random_state=42, hidden_layer_sizes=(7,11,2,))
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)

defaultLayer = mlp.get_params()['hidden_layer_sizes']

confusion_matrix(y_test, predict_test)
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Bad Wine', 'Good Wine']
title='Wine NN Base Performance'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

#performance.append(['wine',.001,.9,50,trainAcc,testAcc])


layerSize = []
"""
for i in range(2,15):
    for i2 in range(2,15):
        for i3 in range(2,15):
            layerSize.append((i,i2,i3,))
"""

learnMomentum = []

for l in [.01,.1,.3,.5]:
    for m in [0,.5,1]:
        learnMomentum.append((l, m))

netSize = []
learnID = []
acc_train = []
acc_test = []
count = 0

for lm in learnMomentum:
    count += 1
    
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=50, hidden_layer_sizes = (7,11,2), random_state=42, learning_rate_init=lm[0], momentum=lm[1])
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    acc_train.append(trainAcc)
    acc_test.append(testAcc)
    
    performance.append(['wine',lm[0],lm[1],50,trainAcc,testAcc])


#take max alpha
ind = acc_test.index(max(acc_test))
learning, momentum = learnMomentum[ind]

#why are some layer sizes so bad?


mlp = MLPClassifier(activation='relu', solver='adam', max_iter=50, hidden_layer_sizes = (7,11,2), random_state=42, learning_rate_init=learning, momentum = momentum)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

"""
plt.scatter(layerID,acc_train)
plt.scatter(layerID,acc_test)
plt.plot(layerID,acc_train,label='train_accuracy',drawstyle="steps-post")
plt.plot(layerID,acc_test,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy by Network Size')
plt.xlabel('Number of Perceptrons')
plt.ylabel('Average Accuracy')
plt.show()
"""


nnIterations = []
acc_train = []
acc_test = []

for i in range(1,200):
    print(i)
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (7,11,2))
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    nnIterations.append(i)
    
    acc_train.append(trainAcc)
    acc_test.append(testAcc)
    
#plt.scatter(nnIterations,acc_train)
#plt.scatter(nnIterations,acc_test)
plt.plot(nnIterations,acc_train,label='train_accuracy',drawstyle="steps-post")
plt.plot(nnIterations,acc_test,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Accuracy by Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.show()
            
#take max alpha
ind = acc_test.index(max(acc_test))
iterations = nnIterations[ind]

mlp = MLPClassifier(activation='relu', solver='adam', max_iter=iterations, hidden_layer_sizes = (7,11,2), random_state=42)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)


title='Wine NN Performance Epochs: ' + str(iterations)
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['wine',.001,.9,iterations,trainAcc,testAcc])








#use case 2
#cancer basic
#cancer basic
performance = []
file = r"C:\Users\rjohn\OneDrive\Documents\GitHub\ML\wdbc.csv"
cancer = pd.read_csv(file, delimiter=',')

cancer['diagnosis'] = np.where(cancer['diagnosis'] == 'M',1,0)

y = cancer.pop('id').values
y = cancer.pop('diagnosis').values
X = cancer.iloc[:,:].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


mlp = MLPClassifier(activation='relu', solver='adam', max_iter=50, random_state=42, hidden_layer_sizes=(7,11,2,))
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)


confusion_matrix(y_test, predict_test)
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Bad Wine', 'Good Wine']
title='Wine NN Base Performance'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

#performance.append(['wine',.001,.9,50,trainAcc,testAcc])


layerSize = []
"""
for i in range(2,15):
    for i2 in range(2,15):
        for i3 in range(2,15):
            layerSize.append((i,i2,i3,))
"""

learnMomentum = []

for l in [.01,.1,.3,.5]:
    for m in [0,.5,1]:
        learnMomentum.append((l, m))

netSize = []
learnID = []
acc_train = []
acc_test = []
count = 0

for lm in learnMomentum:
    count += 1
    
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=50, hidden_layer_sizes = (7,11,2), random_state=42, learning_rate_init=lm[0], momentum=lm[1])
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    acc_train.append(trainAcc)
    acc_test.append(testAcc)
    
    performance.append(['cancer',lm[0],lm[1],50,trainAcc,testAcc])


#take max alpha
ind = acc_test.index(max(acc_test))
learning, momentum = learnMomentum[ind]

#why are some layer sizes so bad?


mlp = MLPClassifier(activation='relu', solver='adam', max_iter=50, hidden_layer_sizes = (7,11,2), random_state=42, learning_rate_init=learning, momentum = momentum)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

"""
plt.scatter(layerID,acc_train)
plt.scatter(layerID,acc_test)
plt.plot(layerID,acc_train,label='train_accuracy',drawstyle="steps-post")
plt.plot(layerID,acc_test,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy by Network Size')
plt.xlabel('Number of Perceptrons')
plt.ylabel('Average Accuracy')
plt.show()
"""


nnIterations = []
acc_train = []
acc_test = []

for i in range(1,500):
    print(i)
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    nnIterations.append(i)
    
    acc_train.append(trainAcc)
    acc_test.append(testAcc)
    
#plt.scatter(nnIterations,acc_train)
#plt.scatter(nnIterations,acc_test)
plt.plot(nnIterations,acc_train,label='train_accuracy',drawstyle="steps-post")
plt.plot(nnIterations,acc_test,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Accuracy by Iteration: Backpropagation')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
            
#take max alpha
ind = acc_test.index(max(acc_test))
iterations = nnIterations[ind]

import time
mlp = MLPClassifier(activation='relu', solver='adam', max_iter=iterations, hidden_layer_sizes = (10,10,10), random_state=42)
start = time.time()
mlp.fit(X_train,y_train)
stop = time.time()

mlpClock = stop - start

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)


title='Cancer NN Performance Iterations: ' + str(iterations)
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['cancer',.001,.9,iterations,trainAcc,testAcc])

performanceDF = pd.DataFrame(performance, columns = ['dataset','learningRate','momentum','epochs','trainAcc','testAcc'])









