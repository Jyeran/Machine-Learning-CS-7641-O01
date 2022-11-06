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


pca = PCA(n_components=30)
pca_train = pca.fit_transform(X_train)[:,:3]
pca_test  = pca.transform(X_test)[:,:3]
pca_test = pca_test[:,:3]


ica = FastICA(n_components=30)
ica_train = ica.fit_transform(X_train)[:,[ 6, 11, 26, 22,  2, 15, 24, 27, 21, 13,  0,  7, 19, 17, 20, 29]]
ica_test  = ica.transform(X_test)
ica_test = ica_test[:,[6, 11, 26, 22,  2, 15, 24, 27, 21, 13,  0,  7, 19, 17, 20, 29]]


rp1 = GaussianRandomProjection(n_components = 30, random_state = 25)
rca_train = rp1.fit_transform(X_train)[:,:17]
rca_test  = rp1.transform(X_test)
rca_test = rca_test[:,:17]


dComps = [7, 21, 22, 23]
dt_train = X_train[:,dComps]
dt_test = X_test[:,dComps]



nnIterations = []
control_acc = []
control_wallClock = []

pca_acc = []
pca_wallClock = []

ica_acc = []
ica_wallClock = []

rca_acc = []
rca_wallClock = []

dt_acc = []
dt_wallClock = []

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
    
    
    #PCA
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    start = time.time()
    mlp.fit(pca_train,y_train)
    stop = time.time()

    predict_test = mlp.predict(pca_test)
    testAcc      = accuracy_score(y_test,predict_test)
    
    pca_acc.append(testAcc)
    pca_wallClock.append(stop-start)
    
    
    #ICA
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    start = time.time()
    mlp.fit(ica_train,y_train)
    stop = time.time()

    predict_test = mlp.predict(ica_test)
    testAcc      = accuracy_score(y_test,predict_test)
    
    ica_acc.append(testAcc)
    ica_wallClock.append(stop-start)
    
    
    #RCA
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    start = time.time()
    mlp.fit(rca_train,y_train)
    stop = time.time()

    predict_test = mlp.predict(rca_test)
    testAcc      = accuracy_score(y_test,predict_test)
    
    rca_acc.append(testAcc)
    rca_wallClock.append(stop-start)
    
    
    #DT
    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=i, random_state=42, hidden_layer_sizes = (10,10,10))
    start = time.time()
    mlp.fit(dt_train,y_train)
    stop = time.time()

    predict_test = mlp.predict(dt_test)
    testAcc      = accuracy_score(y_test,predict_test)
    
    dt_acc.append(testAcc)
    dt_wallClock.append(stop-start)
    
#plt.scatter(nnIterations,acc_train)
#plt.scatter(nnIterations,acc_test)
plt.plot(nnIterations,control_acc,label='Control',drawstyle="steps-post")
plt.plot(nnIterations,pca_acc,label='PCA',drawstyle="steps-post")
plt.plot(nnIterations,ica_acc,label='ICA',drawstyle="steps-post")
plt.plot(nnIterations,rca_acc,label='RCA',drawstyle="steps-post")
plt.plot(nnIterations,dt_acc,label='DT',drawstyle="steps-post")
plt.legend()
plt.title('Cross Val Accuracy by Epochs and Dimensionality Reduction Method')
plt.xlabel('Number of Epochs')
plt.ylabel('Cross Val Accuracy')
plt.show()

#plt.scatter(nnIterations,acc_train)
#plt.scatter(nnIterations,acc_test)
plt.plot(nnIterations,control_wallClock,label='Control',drawstyle="steps-post")
plt.plot(nnIterations,pca_wallClock,label='PCA',drawstyle="steps-post")
plt.plot(nnIterations,ica_wallClock,label='ICA',drawstyle="steps-post")
plt.plot(nnIterations,rca_wallClock,label='RCA',drawstyle="steps-post")
plt.plot(nnIterations,dt_wallClock,label='DT',drawstyle="steps-post")
plt.legend()
plt.title('Wall Clock Runtime by Epochs and Dimensionality Reduction Method')
plt.xlabel('Number of Epochs')
plt.ylabel('Wall Clock Time in Seconds')
plt.show()

























