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

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

file = r"winequality-red.csv"
wine = pd.read_csv(file, delimiter=';')

wine['highQuality'] = np.where(wine['quality'] >= 7, 1, 0)
y = wine.pop('highQuality').values
q = wine.pop('quality').values
X = wine.iloc[:,:10].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


seeds = [1,10,25,50,100]
sSeries = []
for s in seeds:
    reconErr = []
    for k in range(1,11):
        rp1 = GaussianRandomProjection(n_components = k)
        
        rca_train = rp1.fit_transform(X_train)
        
        rca_inverse = rp1.inverse_transform(rca_train)
        
        tmp = np.sum(np.square(X_train - rca_inverse)/rca_inverse.size)
        rce = np.sqrt(tmp)
        
        reconErr.append(rce)
        
    sSeries.append(reconErr)

pcaErr = []
for k in range(1,11):
    rp1 = PCA(n_components = k)
    
    rca_train = rp1.fit_transform(X_train)
    
    rca_inverse = rp1.inverse_transform(rca_train)
    
    tmp = np.sum(np.square(X_train - rca_inverse)/rca_inverse.size)
    rce = np.sqrt(tmp)
    
    pcaErr.append(rce)

    
crange = range(1,11)
plt.plot(crange, pcaErr, label = "PCA") 
count = 0
for s in sSeries:
    plt.plot(crange,s, label = "Seed: " + str(seeds[count]))
    count += 1 
plt.title('WINE: Reconstruction Error across Random Projections', color = 'purple')
plt.xlabel('Components')
plt.ylabel('RMSE')  
plt.legend()  
plt.show()
plt.close()


rp1 = GaussianRandomProjection(n_components = 10)
rca_train = rp1.fit_transform(X_train)
rca_test  = rp1.transform(X_test)
    

seeds = [1,10,25,50,100]
sSeries = []
for s in seeds:
    rp1 = GaussianRandomProjection(n_components = 10, random_state = s)
    rca_train = rp1.fit_transform(X_train)
    dtAcc = []
    for i in range(1,10):
        maxD = 0
        bestD = 0
        for d in range(1,6):
            dt = tree.DecisionTreeClassifier(random_state = 42,criterion='entropy',max_depth=d)
            dt.fit(rca_train[:,:i], y_train)
            
            y_test_pred  = dt.predict(rca_test[:,:i])
            testAcc   = accuracy_score(y_test,y_test_pred)
            if testAcc > maxD:
                maxD = testAcc
                bestD = d
        
        dt = tree.DecisionTreeClassifier(random_state = 42,criterion='entropy',max_depth=bestD)
        dt.fit(rca_train[:,:i], y_train)
        y_test_pred  = dt.predict(rca_test[:,:i])
        testAcc   = accuracy_score(y_test,y_test_pred)
        dtAcc.append(testAcc)
    sSeries.append(dtAcc)

crange = range(0,9)
count = 0
for s in sSeries:
    plt.plot(crange,s,label='Seed: ' + str(seeds[count]))
    count += 1
plt.title('WINE: Avg Decision Tree Accuracy by # Random Components', color = 'purple')
plt.xlabel('Components')
plt.ylabel('DT Cross Val Accuracy')
plt.legend()
plt.show()
plt.close()  

#####################################################################

krange = range(1,15)
sum_squared_errors = []
sum_squared_errorsPCA = []
sil = [0]
i = 6
for k in krange:
  model = KMeans(n_clusters = k)
  model.fit_predict(rca_train[:,:i])
  sum_squared_errorsPCA.append(model.inertia_)
  
  model = KMeans(n_clusters = k)
  model.fit_predict(X_train)
  sum_squared_errors.append(model.inertia_)
          
  if k >= 2:
      kmeans = KMeans(n_clusters = k).fit(rca_train[:,:i])
      labels = kmeans.labels_
      sil.append(silhouette_score(rca_train[:,:i], labels, metric = 'euclidean'))
  
plt.plot(krange,sum_squared_errors, label = 'Kmeans Original')
plt.plot(krange,sum_squared_errorsPCA, label = 'Kmeans RCA')
plt.legend()
plt.title('WINE: K Clusters by Sum of Squared Errors', color = 'purple')
plt.xlabel('K-Value')
plt.ylabel('Sum of Squared Errors')
plt.show()
plt.close()


fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, sum_squared_errorsPCA, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Sum of Squared Errors", fontsize=14)
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14)
ax2.tick_params(axis="y")

fig.suptitle("WINE: K Clusters by SSE and Silhouette", fontsize=20, color = 'purple')
plt.show()
plt.close()


logLpca = []
logL = []
sil = [0]
for k in krange:
    gmm = GaussianMixture(n_components = k)
    gmm.fit(rca_train[:,:i])
    labels = gmm.predict(rca_train[:,:i])
    logLpca.append(gmm.score(rca_train[:,:i]))
    
    gmm = GaussianMixture(n_components = k)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    logL.append(gmm.score(X_train))
    
    if k >= 2:
        sil.append(silhouette_score(rca_train[:,:i], labels, metric = 'euclidean'))
        

plt.plot(krange,logL, label = 'EM Original')
plt.plot(krange,logLpca, label = 'EM PCA')
plt.legend()
plt.title('WINE: EM Clusters by Log Likelihood', color = 'purple')
plt.xlabel('K-Value')
plt.ylabel('Log Likelihood')
plt.show()
plt.close()

fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, logLpca, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Log Likelihood", fontsize=14, color = 'g')
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14, color = 'b')
ax2.tick_params(axis="y")

fig.suptitle("WINE: EM Clusters by SSE and Silhouette", fontsize=20, color = 'purple')
plt.show()
plt.close()
    
##############################################################

kmeans = KMeans(n_clusters = 2).fit(rca_train[:,:i])
Klabels = kmeans.labels_

gmm = GaussianMixture(n_components = 2)
gmm.fit(rca_train[:,:i])
Elabels = gmm.predict(rca_train[:,:i])

wineTrain = pd.DataFrame(X_train, columns = wine.iloc[:,:10].columns)
wineTrain['Kcluster'] = Klabels
wineTrain['Ecluster'] = Elabels
wineTrain['Quality']  = y_train
wineTrain['True Label'] = 1
wineTrain['High Quality'] = wineTrain['Quality']

cols = ['High Quality','Quality','Kcluster', 'Ecluster','True Label']
viz = wineTrain[cols]

kviz = viz.groupby('Kcluster').agg({'High Quality':'mean'})
kviz.plot(kind="bar",legend=False)
plt.title("WINE: Percentage of High Quality Wine by Kmeans Cluster: RCA", color = 'purple')
plt.xlabel("Cluster")
plt.ylabel("Percent of High Quality Wine")
plt.show()
plt.close()
                             
eviz = viz.groupby('Ecluster').agg({'High Quality':'mean'})
eviz.plot(kind="bar",legend=False)
plt.title("WINE: Percentage of High Quality Wine by EM Cluster: RCA", color = 'purple')
plt.xlabel("Cluster")
plt.ylabel("Percent of High Quality Wine")
plt.show()
plt.close()

#cluster chart kmeans
range_n_clusters = [2]
X = rca_train[:,:i]
y = y_train
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("WINE: Visualizing K Means Clusters", color = 'purple')
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()
plt.close()

range_n_clusters = [2]
X = rca_train[:,:i]
y = y_train
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components = n_clusters)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    """
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
        """
    ax2.set_title("WINE: Visualization of EM Clusters", color = 'purple')
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()
plt.close()













file = r"wdbc.csv"
cancer = pd.read_csv(file, delimiter=',')

cancer['diagnosis'] = np.where(cancer['diagnosis'] == 'M',1,0)

y = cancer.pop('id').values
y = cancer.pop('diagnosis').values
X = cancer.iloc[:,:].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
  
seeds = [1,10,25,50,100]
sSeries = []
for s in seeds:
    reconErr = []
    for k in range(1,31):
        rp1 = GaussianRandomProjection(n_components = k)
        
        rca_train = rp1.fit_transform(X_train)
        
        rca_inverse = rp1.inverse_transform(rca_train)
        
        tmp = np.sum(np.square(X_train - rca_inverse)/rca_inverse.size)
        rce = np.sqrt(tmp)
        
        reconErr.append(rce)
        
    sSeries.append(reconErr)

pcaErr = []
for k in range(1,31):
    rp1 = PCA(n_components = k)
    
    rca_train = rp1.fit_transform(X_train)
    
    rca_inverse = rp1.inverse_transform(rca_train)
    
    tmp = np.sum(np.square(X_train - rca_inverse)/rca_inverse.size)
    rce = np.sqrt(tmp)
    
    pcaErr.append(rce)

    
crange = range(1,31)
plt.plot(crange, pcaErr, label = "PCA") 
count = 0
for s in sSeries:
    plt.plot(crange,s, label = "Seed: " + str(seeds[count]))
    count += 1 
plt.title('CANCER: Reconstruction Error across Random Projections', color ='r')
plt.xlabel('Components')
plt.ylabel('RMSE')  
plt.legend()  
plt.show()
plt.close()


rp1 = GaussianRandomProjection(n_components = 30)
rca_train = rp1.fit_transform(X_train)
rca_test  = rp1.transform(X_test)


seeds = [1,10,25,50,100]
sSeries = []
for s in seeds:
    rp1 = GaussianRandomProjection(n_components = 30, random_state = s)
    rca_train = rp1.fit_transform(X_train)
    dtAcc = []
    for i in range(1,30):
        maxD = 0
        bestD = 0
        for d in range(1,6):
            dt = tree.DecisionTreeClassifier(random_state = 42,criterion='entropy',max_depth=d)
            dt.fit(rca_train[:,:i], y_train)
            
            y_test_pred  = dt.predict(rca_test[:,:i])
            testAcc   = accuracy_score(y_test,y_test_pred)
            if testAcc > maxD:
                maxD = testAcc
                bestD = d
        
        dt = tree.DecisionTreeClassifier(random_state = 42,criterion='entropy',max_depth=bestD)
        dt.fit(rca_train[:,:i], y_train)
        y_test_pred  = dt.predict(rca_test[:,:i])
        testAcc   = accuracy_score(y_test,y_test_pred)
        dtAcc.append(testAcc)
    sSeries.append(dtAcc)

crange = range(0,29)
count = 0
for s in sSeries:
    plt.plot(crange,s,label='Seed: ' + str(seeds[count]))
    count += 1
plt.title('CANCER: Avg Decision Tree Accuracy by # Random Components', color ='r')
plt.xlabel('Components')
plt.ylabel('DT Cross Val Accuracy')
plt.legend()
plt.show()
plt.close()  


##########################################################################
rp1 = GaussianRandomProjection(n_components = 30, random_state = 25)
rca_train = rp1.fit_transform(X_train)

krange = range(1,15)
sum_squared_errors = []
sum_squared_errorsPCA = []
sil = [0]
i = 17
for k in krange:
  model = KMeans(n_clusters = k)
  model.fit_predict(rca_train[:,:i])
  sum_squared_errorsPCA.append(model.inertia_)
  
  model = KMeans(n_clusters = k)
  model.fit_predict(X_train)
  sum_squared_errors.append(model.inertia_)
          
  if k >= 2:
      kmeans = KMeans(n_clusters = k).fit(rca_train[:,:i])
      labels = kmeans.labels_
      sil.append(silhouette_score(rca_train[:,:i], labels, metric = 'euclidean'))
  
plt.plot(krange,sum_squared_errors, label = 'Kmeans Original')
plt.plot(krange,sum_squared_errorsPCA, label = 'Kmeans RCA')
plt.legend()
plt.title('CANCER: K Clusters by Sum of Squared Errors', color ='r')
plt.xlabel('K-Value')
plt.ylabel('Sum of Squared Errors')
plt.show()
plt.close()


fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, sum_squared_errorsPCA, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Sum of Squared Errors", fontsize=14)
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14)
ax2.tick_params(axis="y")

fig.suptitle("CANCER: K Clusters by SSE and Silhouette", fontsize=20, color ='r')
plt.show()
plt.close()


logLpca = []
logL = []
sil = [0]
for k in krange:
    gmm = GaussianMixture(n_components = k)
    gmm.fit(rca_train[:,:i])
    labels = gmm.predict(rca_train[:,:i])
    logLpca.append(gmm.score(rca_train[:,:i]))
    
    gmm = GaussianMixture(n_components = k)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    logL.append(gmm.score(X_train))
    
    if k >= 2:
        sil.append(silhouette_score(rca_train[:,:i], labels, metric = 'euclidean'))
        

plt.plot(krange,logL, label = 'EM Original')
plt.plot(krange,logLpca, label = 'EM PCA')
plt.legend()
plt.title('CANCER: EM Clusters by Log Likelihood', color ='r')
plt.xlabel('K-Value')
plt.ylabel('Log Likelihood')
plt.show()
plt.close()

fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, logLpca, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Log Likelihood", fontsize=14, color = 'g')
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14, color = 'b')
ax2.tick_params(axis="y")

fig.suptitle("CANCER: EM Clusters by SSE and Silhouette", fontsize=20, color = 'r')
plt.show()
plt.close()    
   
################################################################

kmeans = KMeans(n_clusters = 3).fit(rca_train[:,:i])
Klabels = kmeans.labels_

gmm = GaussianMixture(n_components = 2)
gmm.fit(rca_train[:,:i])
Elabels = gmm.predict(rca_train[:,:i])

wineTrain = pd.DataFrame(X_train, columns = cancer.iloc[:,:30].columns)
wineTrain['Kcluster'] = Klabels
wineTrain['Ecluster'] = Elabels
wineTrain['Diagnosis']  = y_train

cols = ['Diagnosis','Kcluster', 'Ecluster']
viz = wineTrain[cols]

kviz = viz.groupby('Kcluster').agg({'Diagnosis':'mean'})
kviz.plot(kind="bar",legend=False)
plt.title("CANCER: Percentage of Positive Diagnosis by Kmeans Cluster: RCA", color = 'r')
plt.xlabel("Cluster")
plt.ylabel("Percent Malignant")
plt.show()
plt.close()
                             
eviz = viz.groupby('Ecluster').agg({'Diagnosis':'mean'})
eviz.plot(kind="bar",legend=False)
plt.title("CANCER: Percentage of Positive Diagnosis by EM Cluster: RCA", color = 'r')
plt.xlabel("Cluster")
plt.ylabel("Percent Malignant")
plt.show()
plt.close()

##################################################################

#cluster chart kmeans
range_n_clusters = [3]
X = rca_train[:,:i]
y = y_train
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("CANCER: Visualization of K Means Clusters", color = 'r')
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()
plt.close()

range_n_clusters = [2]
X = rca_train[:,:i]
y = y_train
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components = k)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    """
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
        """
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()
plt.close()

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    