import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier 


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



file = r"winequality-red.csv"
wineRed = pd.read_csv(file, delimiter=';')

file = r"winequality-white.csv"
wineWhite = pd.read_csv(file, delimiter=';')


performance = []
#wine basic
#wineRed['color']   = 1
#wineWhite['color'] = 0

wine = wineRed#pd.concat([wineRed,wineWhite])

sns.histplot(data=wine, x="quality").set(title='Distribution of Wine Quality')

wine['highQuality'] = np.where(wine['quality'] >= 7, 1, 0)

#wine['qualityClass'] = np.where(wine['quality'] >= 4, 2, wine['qualityClass'])
#wine['qualityClass'] = np.where(wine['quality'] >= 8, 3, wine['qualityClass'])

#y = wine.pop('color').values
y = wine.pop('highQuality').values
X = wine.iloc[:,:10].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

dt = tree.DecisionTreeClassifier(random_state = 42,criterion='entropy')
start = time.time()
dt.fit(X_train, y_train)
stop = time.time()


y_train_pred = dt.predict(X_train)
start1 = time.time()
y_test_pred  = dt.predict(X_test)
stop1 = time.time()

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

print(trainAcc, testAcc)#classification_report(y_test,y_test_pred))

performance.append(['wine','base','entropy',dt.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])

plt.figure(figsize=(25,20))
plot_tree(dt, filled=True,feature_names=wine.columns[:10],class_names=['poor','high'])
plt.title("Wine Decision Tree")
plt.show()

dt = tree.DecisionTreeClassifier(random_state = 42,criterion='gini')
start = time.time()
dt.fit(X_train, y_train)
stop = time.time()


y_train_pred = dt.predict(X_train)

start1 = time.time()
y_test_pred  = dt.predict(X_test)
stop1 = time.time()


trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

print(trainAcc, testAcc)#classification_report(y_test,y_test_pred))

performance.append(['wine','base','gini',dt.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])


labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Bad Wine', 'Good Wine']
title = 'Wine Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

#pruning
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.title('Wine CCP: Number of Nodes vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Number of Nodes')
plt.show()

train_acc = []
test_acc = []
train_f1 = []
test_f1 = []

for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))
    train_f1.append(f1_score(y_train_pred,y_train))
    test_f1.append(f1_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine CCP: Accuracy vs alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()

plt.scatter(ccp_alphas,train_f1)
plt.scatter(ccp_alphas,test_f1)
plt.plot(ccp_alphas,train_f1,label='train_f1',drawstyle="steps-post")
plt.plot(ccp_alphas,test_f1,label='test_f1',drawstyle="steps-post")
plt.legend()
plt.title('Wine GINI F1 vs alpha')
plt.xlabel('Alpha')
plt.ylabel('F1 Score')
plt.show()

#take max alpha
ind = test_acc.index(max(test_acc))
alpha = ccp_alphas[ind]

dtGini = tree.DecisionTreeClassifier(ccp_alpha=alpha,random_state=0)
boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
start = time.time()
dtGini.fit(X_train, y_train)
stop = time.time()
y_train_pred = dtGini.predict(X_train)
start1 = time.time()
y_test_pred = dtGini.predict(X_test)
stop1 = time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

title = 'Wine Pruned GINI Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

plt.figure(figsize=(25,20))
plot_tree(dtGini, filled=True,feature_names=wine.columns[:10],class_names=['poor','high'])
plt.title("Wine Pruned GINI Decision Tree")
plt.show()

performance.append(['wine','pruned'+str(round(alpha,3)),'gini',dtGini.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])





#pruning for entropy

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.title('Wine Entropy Number of Nodes vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Number of Nodes')
plt.show()

train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Entropy Accuracy vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()

#take max alpha
ind = test_acc.index(max(test_acc))
alpha = ccp_alphas[ind]

dtEnt = clfs[ind]

dtEnt = tree.DecisionTreeClassifier(ccp_alpha=alpha, criterion='entropy',random_state=0)
start = time.time()
dtEnt.fit(X_train, y_train)
stop = time.time()
y_train_pred = dtEnt.predict(X_train)
start1 = time.time()
y_test_pred = dtEnt.predict(X_test)
stop1 = time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

plt.figure(figsize=(25,20))
plot_tree(dtEnt, filled=True,feature_names=wine.columns[:10],class_names=['poor','high'])
plt.title("Wine Decision Tree after Pruning")
plt.show()

title = 'Wine Pruned Entropy Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()


performance.append(['wine','pruned'+str(round(alpha,3)),'ent',dtEnt.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])




#boosting
perfBoost = []
for p in performance:
    if p[0] == 'wine':
        perfBoost.append([p[0],'base',p[1],'',p[4],p[5],p[6],p[7]])
clfs = []

for ccp_alpha in ccp_alphas:
    clf1 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha) 
    clf = AdaBoostClassifier(base_estimator=clf1, n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    

train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Boosted DT Accuracy vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()

#take max alpha
ind = test_acc.index(max(test_acc))
alpha = ccp_alphas[ind]

boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=10, random_state=0)

start = time.time()
boost.fit(X_train,y_train)
stop=time.time()
y_train_pred = boost.predict(X_train)
start1=time.time()
y_test_pred = boost.predict(X_test)
stop1=time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

confusion_matrix(y_test, y_test_pred)

title = 'Wine Boosted Pruned Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

perfBoost.append(['wine','boosted','pruned'+str(round(alpha,3)),10,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])



boost_acc_train = []
boost_acc_test = []
nestimators = []
for i in range(1,100):
    boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
    boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=i, random_state=0)

    boost.fit(X_train,y_train)
    y_train_pred = boost.predict(X_train)
    y_test_pred = boost.predict(X_test)
    
    boost_acc_train.append(accuracy_score(y_train_pred,y_train))
    boost_acc_test.append(accuracy_score(y_test_pred,y_test))
    nestimators .append(i)

plt.plot(nestimators,boost_acc_train,label='train_accuracy',drawstyle="steps-post")
plt.plot(nestimators,boost_acc_test,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Wine Boosted Gain by Estimator Iteration')
plt.xlabel('Boost Iteration')
plt.ylabel('Accuracy')
plt.show()

ind = boost_acc_test.index(max(boost_acc_test))
est = nestimators[ind]

boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=est, random_state=0)

start = time.time()
boost.fit(X_train,y_train)
stop=time.time()
y_train_pred = boost.predict(X_train)
start1=time.time()
y_test_pred = boost.predict(X_test)
stop1=time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

confusion_matrix(y_test, y_test_pred)

title = 'Wine Boosted Pruned and Iterated Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()


perfBoost.append(['wine','boosted','pruned'+str(round(alpha,3)),est,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])


#why does performance decay with more iterations instead of flatlining like the theory presented in class
#compare confusion matrices



#analysis, difficulty of predicting things that are somewhat rare
#process of pruning to eliminate overfitting and allow model to generalize somewhat
#gini vs entropy https://quantdare.com/decision-trees-gini-vs-entropy/

#need to get training time per 

#need to try CV folds
k = 5
kf = KFold(n_splits=k, random_state=None)
 
acc_score = []
k_mods = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    dtGini.fit(X_train,y_train)
    pred_values = dtGini.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
    k_mods.append(dtGini)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

#need to get some confusion matrices

#need to plot learning curves










#cancer basic
file = r"wdbc.csv"
cancer = pd.read_csv(file, delimiter=',')

cancer['diagnosis'] = np.where(cancer['diagnosis'] == 'M',1,0)

y = cancer.pop('id').values
y = cancer.pop('diagnosis').values
X = cancer.iloc[:,:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


dt = tree.DecisionTreeClassifier(random_state = 42,criterion='entropy')
start = time.time()
dt.fit(X_train, y_train)
stop = time.time()

y_train_pred = dt.predict(X_train)
start1 = time.time()
y_test_pred  = dt.predict(X_test)
stop1 = time.time()

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

print(trainAcc, testAcc)#classification_report(y_test,y_test_pred))

performance.append(['cancer','base','entropy',dt.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])

print(trainAcc, testAcc)#classification_report(y_test,y_test_pred))

plt.figure(figsize=(25,20))
plot_tree(dt, filled=True,feature_names=cancer.columns[:],class_names=['poor','high'])
plt.title("Cancer Decision Tree")
plt.show()

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Benign', 'Malignant']
title = 'Cancer Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()


dt = tree.DecisionTreeClassifier(random_state = 42,criterion='gini')
start = time.time()
dt.fit(X_train, y_train)
stop = time.time()


y_train_pred = dt.predict(X_train)

start1 = time.time()
y_test_pred  = dt.predict(X_test)
stop1 = time.time()


trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

print(trainAcc, testAcc)#classification_report(y_test,y_test_pred))

performance.append(['cancer','base','gini',dt.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])

#pruning
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.title('Cancer CCP: Number of Nodes vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Number of Nodes')
plt.show()

train_acc = []
test_acc = []
train_f1 = []
test_f1 = []

for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))
    train_f1.append(f1_score(y_train_pred,y_train))
    test_f1.append(f1_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer CCP: Accuracy vs alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()

plt.scatter(ccp_alphas,train_f1)
plt.scatter(ccp_alphas,test_f1)
plt.plot(ccp_alphas,train_f1,label='train_f1',drawstyle="steps-post")
plt.plot(ccp_alphas,test_f1,label='test_f1',drawstyle="steps-post")
plt.legend()
plt.title('Cancer GINI F1 vs alpha')
plt.show()

#take max alpha
ind = test_acc.index(max(test_acc))
alpha = ccp_alphas[ind]

dtGini = tree.DecisionTreeClassifier(ccp_alpha=alpha)
boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
start = time.time()
dtGini.fit(X_train, y_train)
stop = time.time()
y_train_pred = dtGini.predict(X_train)
start1 = time.time()
y_test_pred = dtGini.predict(X_test)
stop1 = time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

title = 'Cancer Pruned Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()

plt.figure(figsize=(25,20))
plot_tree(dtGini, filled=True,feature_names=cancer.columns[:],class_names=['poor','high'])
plt.title("Cancer Pruned Decision Tree")
plt.show()


performance.append(['cancer','pruned'+str(round(alpha,3)),'gini',dtGini.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])



#pruning for entropy

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Entropy Number of Nodes vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Number of Nodes')
plt.show()

train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Entropy Accuracy vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()

#take max alpha
ind = test_acc.index(max(test_acc))
alpha = ccp_alphas[ind]

dtEnt = tree.DecisionTreeClassifier(ccp_alpha=alpha, criterion='entropy')
start = time.time()
dtEnt.fit(X_train, y_train)
stop = time.time()
y_train_pred = dtEnt.predict(X_train)
start1 = time.time()
y_test_pred = dtEnt.predict(X_test)
stop1 = time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

plt.figure(figsize=(25,20))
plot_tree(dtEnt, filled=True,feature_names=cancer.columns[:],class_names=['poor','high'])
plt.title("Wine Decision Tree after Pruning")
plt.show()

title = 'Cancer Pruned Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()



performance.append(['cancer','pruned'+str(round(alpha,3)),'ent',dtGini.tree_.node_count,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])

performanceDF = pd.DataFrame(performance, columns = ['dataset','model','splitting','treeSize','trainacc','testacc','trainTime','predictTime'])





#boosting
for p in performance:
    if p[0] == 'cancer':
        perfBoost.append([p[0],'base',p[1],'',p[4],p[5],p[6],p[7]])
        
clfs = []

for ccp_alpha in ccp_alphas:
    clf1 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha) 
    clf = AdaBoostClassifier(base_estimator=clf1, n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    

train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Boosted DT Accuracy vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()

#take max alpha
ind = test_acc.index(max(test_acc))
alpha = ccp_alphas[ind]

boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=10, random_state=0)

start = time.time()
boost.fit(X_train,y_train)
stop=time.time()
y_train_pred = boost.predict(X_train)
start1=time.time()
y_test_pred = boost.predict(X_test)
stop1=time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

confusion_matrix(y_test, y_test_pred)

title = 'Cancer Boosted Pruned Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
#sns.heatmap(cm, annot=True)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()


perfBoost.append(['cancer','boosted','pruned'+str(round(alpha,3)),10,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])



boost_acc_train = []
boost_acc_test = []
nestimators = []
for i in range(1,100):
    boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
    boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=i, random_state=0)

    boost.fit(X_train,y_train)
    y_train_pred = boost.predict(X_train)
    y_test_pred = boost.predict(X_test)
    
    boost_acc_train.append(accuracy_score(y_train_pred,y_train))
    boost_acc_test.append(accuracy_score(y_test_pred,y_test))
    nestimators .append(i)

plt.plot(nestimators,boost_acc_train,label='train_accuracy',drawstyle="steps-post")
plt.plot(nestimators,boost_acc_test,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Cancer Boosted Gain by Estimator Iteration')
plt.xlabel('Boost Iteration')
plt.ylabel('Accuracy')
plt.show()

ind = boost_acc_test.index(max(boost_acc_test))
est = nestimators[ind]

boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=est, random_state=0)

start = time.time()
boost.fit(X_train,y_train)
stop=time.time()
y_train_pred = boost.predict(X_train)
start1=time.time()
y_test_pred = boost.predict(X_test)
stop1=time.time()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

trainAcc  = accuracy_score(y_train,y_train_pred)
testAcc   = accuracy_score(y_test,y_test_pred)

confusion_matrix(y_test, y_test_pred)

title = 'Cancer Boosted Pruned and Iterated Decision Tree Confusion Matrix'
cm = confusion_matrix(y_test, y_test_pred)
make_confusion_matrix(cm,group_names=labels,categories=categories, title = title)
plt.show()
plt.close()



perfBoost.append(['cancer','boosted','pruned'+str(round(alpha,3)),est,round(trainAcc,3),round(testAcc,3),round(stop-start,3),round(stop1-start1,3)])


perfBoostDF = pd.DataFrame(perfBoost, columns = ['dataset','model','pruneAlpha','nEstimators','trainAcc','testAcc','trainTime','predictTime'])















