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

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from scipy.stats import norm, kurtosis
import time

file = r"wdbc.csv"
cancer = pd.read_csv(file, delimiter=',')

cancer['diagnosis'] = np.where(cancer['diagnosis'] == 'M',1,0)

y = cancer.pop('id').values
y = cancer.pop('diagnosis').values
X = cancer.iloc[:,:].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


model = KMeans(n_clusters = 2)
knn_train = np.c_[X_train, model.fit_predict(X_train)]
knn_test = np.c_[X_test,model.fit_predict(X_test)]

model = gmm = GaussianMixture(n_components = 2)
em_train = np.c_[X_train, model.fit_predict(X_train)]
em_test = np.c_[X_test,model.fit_predict(X_test)]


nnIterations = []
control_acc = []
control_wallClock = []

knn_acc = []
knn_wallClock = []

em_acc = []
em_wallClock = []

for i in range(1,200):
    print(i)
    nnIterations.append(i)
    
    #control
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    start = time.time()
    mlp.fit(X_train,y_train)
    stop = time.time()

    predict_test = mlp.predict(X_test)
    testAcc      = accuracy_score(y_test,predict_test)
    
    control_acc.append(testAcc)
    control_wallClock.append(stop-start)
    
    
    #K Nearest
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    start = time.time()
    mlp.fit(knn_train,y_train)
    stop = time.time()

    predict_test = mlp.predict(knn_test)
    testAcc      = accuracy_score(y_test,predict_test)
    
    knn_acc.append(testAcc)
    knn_wallClock.append(stop-start)
    
    
    #EM
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    start = time.time()
    mlp.fit(em_train,y_train)
    stop = time.time()

    predict_test = mlp.predict(em_test)
    testAcc      = accuracy_score(y_test,predict_test)
    
    em_acc.append(testAcc)
    em_wallClock.append(stop-start)
    
    
    
#plt.scatter(nnIterations,acc_train)
#plt.scatter(nnIterations,acc_test)
plt.plot(nnIterations,control_acc,label='Control',drawstyle="steps-post")
plt.plot(nnIterations,knn_acc,label='K Nearest',drawstyle="steps-post")
plt.plot(nnIterations,em_acc,label='EM',drawstyle="steps-post")
plt.legend()
plt.title('Cross Val Accuracy by Epochs and Clustering Method')
plt.xlabel('Number of Epochs')
plt.ylabel('Cross Val Accuracy')
plt.show()

#plt.scatter(nnIterations,acc_train)
#plt.scatter(nnIterations,acc_test)
plt.plot(nnIterations,control_wallClock,label='Control',drawstyle="steps-post")
plt.plot(nnIterations,knn_wallClock,label='K nearest',drawstyle="steps-post")
plt.plot(nnIterations,em_wallClock,label='EM',drawstyle="steps-post")
plt.legend()
plt.title('Wall Clock Runtime by Epochs and Clustering Method')
plt.xlabel('Number of Epochs')
plt.ylabel('Wall Clock Time in Seconds')
plt.show()

























