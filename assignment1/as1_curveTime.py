import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



#why is knn worse than decision tree for our dataset with few examples? Well, there 
#are simply *more* neighbors that are labelled incorectly

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

trainingSize = []

svmTime = []
svmTimeP = []
svmLearnCurve_train = []
svmLearnCurve_test  = []

knnTime = []
knnTimeP = []
knnLearnCurve_train = []
knnLearnCurve_test  = []

boostTime = []
boostTimeP = []
boostLearnCurve_train = []
boostLearnCurve_test  = []

dtTime = []
dtTimeP = []
dtLearnCurve_train = []
dtLearnCurve_test  = []

nnTime = []
nnTimeP = []
nnLearnCurve_train = []
nnLearnCurve_test  = []

for s in range(10, len(X_train)):
    print("On TrainSize: ", s)
    
    X_trainCurve = X_train[:s,:]
    y_trainCurve = y_train[:s]
    
    if np.mean(y_trainCurve) > 0:
        trainingSize.append(s)
        
        #svm
        svm = SVC(kernel = 'rbf', gamma = 4)
        start = time.time()
        svm.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = svm.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = svm.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        svmTime.append(stop - start)
        svmTimeP.append(stop1 - start1)
        svmLearnCurve_train.append(trainAcc)
        svmLearnCurve_test.append(testAcc)
        
        
        
        #knn
        knn = KNeighborsClassifier(n_neighbors=8,weights="distance",algorithm='brute')
        start = time.time()
        knn.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = knn.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = knn.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        knnTime.append(stop - start)
        knnTimeP.append(stop1 - start1)
        knnLearnCurve_train.append(trainAcc)
        knnLearnCurve_test.append(testAcc)
        
        
        
        #boosting
        alpha = 0.0015387655246150347
        nest = 98
        boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
        boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=nest, random_state=0)
        start = time.time()
        boost.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = boost.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = boost.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        boostTime.append(stop - start)
        boostTimeP.append(stop1 - start1)
        boostLearnCurve_train.append(trainAcc)
        boostLearnCurve_test.append(testAcc)
        
        
        
        #dt
        alpha = .005
        dt = tree.DecisionTreeClassifier(ccp_alpha=alpha, criterion='entropy')
        start = time.time()
        dt.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = dt.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = dt.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        dtTime.append(stop - start)
        dtTimeP.append(stop1 - start1)
        dtLearnCurve_train.append(trainAcc)
        dtLearnCurve_test.append(testAcc)
        
        
        #nn
        alpha = .0035387655246150347
        nn = MLPClassifier(activation='relu', solver='adam', max_iter=150, random_state=42)
        start = time.time()
        nn.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = nn.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = nn.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        nnTime.append(stop - start)
        nnTimeP.append(stop1 - start1)
        nnLearnCurve_train.append(trainAcc)
        nnLearnCurve_test.append(testAcc)
        


#Learning Curves Train
plt.plot(trainingSize,nnLearnCurve_train,label='Neural Net Train Curve')
plt.plot(trainingSize,dtLearnCurve_train,label='Decision Tree Train Curve')
plt.plot(trainingSize,boostLearnCurve_train,label='Boost Train Curve')
plt.plot(trainingSize,knnLearnCurve_train,label='KNN Train Curve')
plt.plot(trainingSize,svmLearnCurve_train,label='SVM Train Curve')
plt.legend()
plt.title('Wine Training Learning Curves')
plt.xlabel('Number of Training Records')
plt.ylabel('Accuracy')
plt.show()

#Learning Curves Test
plt.plot(trainingSize,nnLearnCurve_test,label='Neural Net Test Curve')
plt.plot(trainingSize,dtLearnCurve_test,label='Decision Tree Test Curve')
plt.plot(trainingSize,boostLearnCurve_test,label='Boost Test Curve')
plt.plot(trainingSize,knnLearnCurve_test,label='KNN Test Curve')
plt.plot(trainingSize,svmLearnCurve_test,label='SVM Test Curve')
plt.legend()
plt.title('Wine Testing Learning Curves')
plt.xlabel('Number of Training Records')
plt.ylabel('Accuracy')
plt.show()

#Learning Times
plt.plot(trainingSize,nnTime,label='Neural Net Train Time')
plt.plot(trainingSize,dtTime,label='Decision Tree Train Time')
plt.plot(trainingSize,boostTime,label='Boost Train Time')
plt.plot(trainingSize,knnTime,label='KNN Train Time')
plt.plot(trainingSize,svmTime,label='SVM Train Time')
plt.legend()
plt.title('Wine Training Learning Times')
plt.xlabel('Number of Training Records')
plt.ylabel('Time')
plt.show()

#Learning Times
plt.plot(trainingSize,dtTime,label='Decision Tree Train Time')
plt.plot(trainingSize,knnTime,label='KNN Train Time')
plt.plot(trainingSize,svmTime,label='SVM Train Time')
plt.legend()
plt.title('Wine Training Learning Times')
plt.xlabel('Number of Training Records')
plt.ylabel('Time')
plt.show()

#Learning Times
plt.plot(trainingSize,nnTimeP,label='Neural Net Train Time')
plt.plot(trainingSize,dtTimeP,label='Decision Tree Train Time')
plt.plot(trainingSize,boostTimeP,label='Boost Train Time')
plt.plot(trainingSize,knnTimeP,label='KNN Train Time')
plt.plot(trainingSize,svmTimeP,label='SVM Train Time')
plt.legend()
plt.title('Wine Training Query Times')
plt.xlabel('Number of Training Records')
plt.ylabel('Time')
plt.show()


"""
#why boosting jump?
#boosting
#boosting jump corresponds with accuracy jump.
#Likely there are samples in this segment of the data that alter the information gain
#This gives the model more to learn, but also more to process as well
X_trainCurve = X_train[:250,:]
y_trainCurve = y_train[:250]

alpha = 0.0035387655246150347
nest = 89
boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=nest, random_state=0)
start = time.time()
boost.fit(X_trainCurve,y_trainCurve)
stop = time.time()

start1 = time.time()
predict_train = boost.predict(X_trainCurve)
stop1 = time.time()
predict_test = boost.predict(X_test)

trainAcc  = accuracy_score(y_trainCurve,predict_train)
testAcc   = accuracy_score(y_test,predict_test)

print(f'Train score {trainAcc}')
print(f'Test score {testAcc}')

confusion_matrix(y_test, predict_test)

boostTime.append(stop - start)
boostTimeP.append(stop1 - start1)
boostLearnCurve_train.append(trainAcc)
boostLearnCurve_test.append(testAcc)

"""







cancer['diagnosis'] = np.where(cancer['diagnosis'] == 'M',1,0)

y = cancer.pop('id').values
y = cancer.pop('diagnosis').values
X = cancer.iloc[:,:].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

trainingSize = []

svmTime = []
svmTimeP = []
svmLearnCurve_train = []
svmLearnCurve_test  = []

knnTime = []
knnTimeP = []
knnLearnCurve_train = []
knnLearnCurve_test  = []

boostTime = []
boostTimeP = []
boostLearnCurve_train = []
boostLearnCurve_test  = []

dtTime = []
dtTimeP = []
dtLearnCurve_train = []
dtLearnCurve_test  = []

nnTime = []
nnTimeP = []
nnLearnCurve_train = []
nnLearnCurve_test  = []

for s in range(10, len(X_train)):
    print("On TrainSize: ", s)
    
    X_trainCurve = X_train[:s,:]
    y_trainCurve = y_train[:s]
    
    if np.mean(y_trainCurve) > 0:
        trainingSize.append(s)
        
        #svm
        svm = SVC(kernel = 'poly',degree=1)
        start = time.time()
        svm.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = svm.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = svm.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        svmTime.append(stop - start)
        svmTimeP.append(stop1 - start1)
        svmLearnCurve_train.append(trainAcc)
        svmLearnCurve_test.append(testAcc)
        
        
        
        #knn
        knn = KNeighborsClassifier(n_neighbors=10,weights="distance",algorithm='brute')
        start = time.time()
        knn.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = knn.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = knn.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        knnTime.append(stop - start)
        knnTimeP.append(stop1 - start1)
        knnLearnCurve_train.append(trainAcc)
        knnLearnCurve_test.append(testAcc)
        
        
        
        #boosting
        alpha = 0.005
        nest = 94
        boostDt = tree.DecisionTreeClassifier(ccp_alpha=alpha) 
        boost = AdaBoostClassifier(base_estimator=boostDt, n_estimators=nest, random_state=0)
        start = time.time()
        boost.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = boost.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = boost.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        boostTime.append(stop - start)
        boostTimeP.append(stop1 - start1)
        boostLearnCurve_train.append(trainAcc)
        boostLearnCurve_test.append(testAcc)
        
        
        
        #dt
        alpha = .011
        dt = tree.DecisionTreeClassifier(ccp_alpha=alpha)
        start = time.time()
        dt.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = dt.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = dt.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        dtTime.append(stop - start)
        dtTimeP.append(stop1 - start1)
        dtLearnCurve_train.append(trainAcc)
        dtLearnCurve_test.append(testAcc)
        
        
        #nn
        nn = MLPClassifier(activation='relu', solver='adam', max_iter=150, random_state=42)
        start = time.time()
        nn.fit(X_trainCurve,y_trainCurve)
        stop = time.time()
    
        start1 = time.time()
        predict_train = nn.predict(X_trainCurve)
        stop1 = time.time()
        predict_test = nn.predict(X_test)
        
        trainAcc  = accuracy_score(y_trainCurve,predict_train)
        testAcc   = accuracy_score(y_test,predict_test)
        
        nnTime.append(stop - start)
        nnTimeP.append(stop1 - start1)
        nnLearnCurve_train.append(trainAcc)
        nnLearnCurve_test.append(testAcc)
        


#Learning Curves Train
plt.plot(trainingSize,nnLearnCurve_train,label='Neural Net Train Curve')
plt.plot(trainingSize,dtLearnCurve_train,label='Decision Tree Train Curve')
plt.plot(trainingSize,boostLearnCurve_train,label='Boost Train Curve')
plt.plot(trainingSize,knnLearnCurve_train,label='KNN Train Curve')
plt.plot(trainingSize,svmLearnCurve_train,label='SVM Train Curve')
plt.legend()
plt.title('Cancer Training Learning Curves')
plt.xlabel('Number of Training Records')
plt.ylabel('Accuracy')
plt.show()

#Learning Curves Test
plt.plot(trainingSize,nnLearnCurve_test,label='Neural Net Test Curve')
plt.plot(trainingSize,dtLearnCurve_test,label='Decision Tree Test Curve')
plt.plot(trainingSize,boostLearnCurve_test,label='Boost Test Curve')
plt.plot(trainingSize,knnLearnCurve_test,label='KNN Test Curve')
plt.plot(trainingSize,svmLearnCurve_test,label='SVM Test Curve')
plt.legend()
plt.title('Cancer Testing Learning Curves')
plt.xlabel('Number of Training Records')
plt.ylabel('Accuracy')
plt.show()

#Learning Times
plt.plot(trainingSize,nnTime,label='Neural Net Train Time')
plt.plot(trainingSize,dtTime,label='Decision Tree Train Time')
plt.plot(trainingSize,boostTime,label='Boost Train Time')
plt.plot(trainingSize,knnTime,label='KNN Train Time')
plt.plot(trainingSize,svmTime,label='SVM Train Time')
plt.legend()
plt.title('Cancer Training Learning Times')
plt.xlabel('Number of Training Records')
plt.ylabel('Time')
plt.show()

#Learning Times
plt.plot(trainingSize,dtTime,label='Decision Tree Train Time')
plt.plot(trainingSize,knnTime,label='KNN Train Time')
plt.plot(trainingSize,svmTime,label='SVM Train Time')
plt.legend()
plt.title('Cancer Training Learning Times')
plt.xlabel('Number of Training Records')
plt.ylabel('Time')
plt.show()

#Learning Times
plt.plot(trainingSize,nnTimeP,label='Neural Net Train Time')
plt.plot(trainingSize,dtTimeP,label='Decision Tree Train Time')
plt.plot(trainingSize,boostTimeP,label='Boost Train Time')
plt.plot(trainingSize,knnTimeP,label='KNN Train Time')
plt.plot(trainingSize,svmTimeP,label='SVM Train Time')
plt.legend()
plt.title('Cancer Training Query Times')
plt.xlabel('Number of Training Records')
plt.ylabel('Time')
plt.show()














