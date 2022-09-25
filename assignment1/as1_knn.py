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
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


#why is knn worse than decision tree for our dataset with few examples? Well, there 
#are simply *more* neighbors that are labelled incorectly
performance = []
#cancer basic
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

knn = KNeighborsClassifier(n_neighbors=8,weights="distance")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Bad Wine', 'Good Wine']
title='Wine Distance KNN Performance K = 8'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['wine',5, 'uniform',trainAcc,testAcc])

knn = KNeighborsClassifier(n_neighbors=5,weights="distance")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Bad Wine', 'Good Wine']
title='Wine Base KNN Performance'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['wine',5, 'weighted',trainAcc,testAcc])

kList = []
test_accu  = []
train_acc = []
for k in range(1,150):
    knn = KNeighborsClassifier(n_neighbors=k,weights="uniform")
    knn.fit(X_train,y_train)
    
    predict_train = knn.predict(X_train)
    predict_test = knn.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    test_accu.append(testAcc)
    train_acc.append(trainAcc)
    kList.append(k)
    

plt.plot(kList,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(kList,test_accu,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Accuracy vs Uniform K')
plt.xlabel('Uniform K')
plt.ylabel('Accuracy')
plt.show()

plt.plot(kList[:25],train_acc[:25],label='train_accuracy',drawstyle="steps-post")
plt.plot(kList[:25],test_accu[:25],label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Accuracy vs Uniform K')
plt.xlabel('Uniform K')
plt.ylabel('Accuracy')
plt.show()

ind = test_accu.index(max(test_accu))
k = kList[ind]
print(k)

knn = KNeighborsClassifier(n_neighbors=k,weights="uniform")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)

confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['wine',k, 'uniform',trainAcc,testAcc])


kList = []
test_accd  = []
train_acc = []
for k in range(1,150):
    knn = KNeighborsClassifier(n_neighbors=k,weights="distance")
    knn.fit(X_train,y_train)
    
    predict_train = knn.predict(X_train)
    predict_test = knn.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    test_accd.append(testAcc)
    train_acc.append(trainAcc)
    kList.append(k)

plt.plot(kList,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(kList,test_accd,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Accuracy vs Distance Weighted K')
plt.xlabel('Weighted K')
plt.ylabel('Accuracy')
plt.show()

plt.plot(kList[:50],train_acc[:50],label='train_accuracy',drawstyle="steps-post")
plt.plot(kList[:50],test_accd[:50],label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Accuracy vs Distance Weighted K')
plt.show()

plt.plot(kList,test_accu,label='uniform_accuracy',drawstyle="steps-post")
plt.plot(kList,test_accd,label='distance_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Uniform vs Distance Weighted Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()



ind = test_accd.index(max(test_accd))
k = kList[ind]
print(k)

knn = KNeighborsClassifier(n_neighbors=k,weights="distance")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

performance.append(['wine',k, 'distance',trainAcc,testAcc])

print(trainAcc, testAcc)

confusion_matrix(y_test, predict_test)
title='Wine Tuned KNN Performance K: ' + str(k) + ' Weights: Uniform'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

knn.get_params()










cancer['diagnosis'] = np.where(cancer['diagnosis'] == 'M',1,0)

y = cancer.pop('id').values
y = cancer.pop('diagnosis').values
X = cancer.iloc[:,:].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5,weights="uniform")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Benign', 'Malignant']
title='Cancer Base KNN Performance'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['cancer',5, 'uniform',trainAcc,testAcc])


knn = KNeighborsClassifier(n_neighbors=5,weights="uniform")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Benign', 'Malignant']
title='Cancer Base KNN Performance'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['cancer',5, 'weighted',trainAcc,testAcc])


kList = []
test_accu  = []
train_acc = []
for k in range(1,150):
    knn = KNeighborsClassifier(n_neighbors=k,weights="uniform")
    knn.fit(X_train,y_train)
    
    predict_train = knn.predict(X_train)
    predict_test = knn.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    test_accu.append(testAcc)
    train_acc.append(trainAcc)
    kList.append(k)
    

plt.plot(kList,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(kList,test_accu,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Accuracy vs Uniform K')
plt.xlabel('Uniform K')
plt.ylabel('Accuracy')
plt.show()

plt.plot(kList[:25],train_acc[:25],label='train_accuracy',drawstyle="steps-post")
plt.plot(kList[:25],test_accu[:25],label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Accuracy vs Uniform K')
plt.xlabel('Uniform K')
plt.ylabel('Accuracy')
plt.show()

ind = test_accu.index(max(test_accu))
k = kList[ind]
print(k)

knn = KNeighborsClassifier(n_neighbors=k,weights="uniform")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(trainAcc, testAcc)

confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

performance.append(['cancer',k, 'uniform',trainAcc,testAcc])


kList = []
test_accd  = []
train_acc = []
for k in range(1,150):
    knn = KNeighborsClassifier(n_neighbors=k,weights="distance")
    knn.fit(X_train,y_train)
    
    predict_train = knn.predict(X_train)
    predict_test = knn.predict(X_test)

    trainAcc  = accuracy_score(y_train,predict_train)
    testAcc   = accuracy_score(y_test,predict_test)
    
    test_accd.append(testAcc)
    train_acc.append(trainAcc)
    kList.append(k)

plt.plot(kList,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(kList,test_accd,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Accuracy vs Distance Weighted K')
plt.xlabel('Weighted K')
plt.ylabel('Accuracy')
plt.show()

plt.plot(kList[:50],train_acc[:50],label='train_accuracy',drawstyle="steps-post")
plt.plot(kList[:50],test_accd[:50],label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Accuracy vs Distance Weighted K')
plt.xlabel('Weighted K')
plt.ylabel('Accuracy')
plt.show()

plt.plot(kList,test_accu,label='uniform_accuracy',drawstyle="steps-post")
plt.plot(kList,test_accd,label='distance_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Uniform vs Distance Weighted K')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

ind = test_accd.index(max(test_accd))
k = kList[ind]
print(k)

knn = KNeighborsClassifier(n_neighbors=k,weights="distance")
knn.fit(X_train,y_train)

predict_train = knn.predict(X_train)
predict_test = knn.predict(X_test)

trainAcc  = accuracy_score(y_train,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

performance.append(['cancer',k, 'weighted',trainAcc,testAcc])

print(trainAcc, testAcc)

confusion_matrix(y_test, predict_test)
title='Cancer Tuned KNN Performance K: ' + str(k) + ' Weights: Uniform'
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()




performanceDF = pd.DataFrame(performance, columns = ['dataset','k','weighting','trainAcc','testAcc'])



