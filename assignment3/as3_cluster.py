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

file = r"winequality-red.csv"
wine = pd.read_csv(file, delimiter=';')

wine['highQuality'] = np.where(wine['quality'] >= 7, 1, 0)
y = wine.pop('highQuality').values
#q = wine.pop('quality').values
X = wine.iloc[:,:10].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


krange = range(1,15)
sum_squared_errors = []
sum_squared_errors25 = []
sum_squared_errors50 = []
sum_squared_errors75 = []
sil = [0]

for k in krange:
  model = KMeans(n_clusters = k)
  model.fit_predict(X_train)
  sum_squared_errors.append(model.inertia_)
  
  for i in [25,50,75]:
      modelr = KMeans(n_clusters = k, random_state=i)
      modelr.fit_predict(X_train)
      
      if i == 25:
          sum_squared_errors25.append(modelr.inertia_)
      elif i == 50:
          sum_squared_errors50.append(modelr.inertia_)
      else:
          sum_squared_errors75.append(modelr.inertia_)
          
  if k >= 2:
      kmeans = KMeans(n_clusters = k).fit(X_train)
      labels = kmeans.labels_
      sil.append(silhouette_score(X_train, labels, metric = 'euclidean'))
  
plt.plot(krange,sum_squared_errors25, label='Seed: 25')
plt.plot(krange,sum_squared_errors50, label='Seed: 50')
plt.plot(krange,sum_squared_errors75, label='Seed: 75')
plt.title('K Clusters by Sum of Squared Errors')
plt.xlabel('K-Value')
plt.ylabel('Sum of Squared Errors')
plt.legend()
plt.show()
plt.close()


fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, sum_squared_errors, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Sum of Squared Errors", fontsize=14, color = 'b')
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14, color = 'g')
ax2.tick_params(axis="y")

fig.suptitle("K Clusters by SSE and Silhouette Score", fontsize=20)
plt.show()
plt.close()


logL = []
sil = [0]
for k in krange:
    gmm = GaussianMixture(n_components = k)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    logL.append(gmm.score(X_train))
    
    if k >= 2:
        sil.append(silhouette_score(X_train, labels, metric = 'euclidean'))

fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, logL, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Log Likelihood", fontsize=14, color = 'b')
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14, color = 'g')
ax2.tick_params(axis="y")

fig.suptitle("EM Clusters by Log Likelihood and Silhouette Score", fontsize=20)
plt.show()
plt.close()


####################################################################

kmeans = KMeans(n_clusters = 2).fit(X_train)
Klabels = kmeans.labels_

gmm = GaussianMixture(n_components = 2)
gmm.fit(X_train)
Elabels = gmm.predict(X_train)

wineTrain = pd.DataFrame(X_train, columns = wine.iloc[:,:10].columns)
wineTrain['Kcluster'] = Klabels
wineTrain['Ecluster'] = Elabels
wineTrain['Quality']  = y_train
wineTrain['True Label'] = 1
wineTrain['Wine Quality'] = np.where(wineTrain['Quality'] >= 7, "High Quality", "Low Quality")
wineTrain['High Quality'] = np.where(wineTrain['Quality'] >= 7, 1, 0)

cols = ['High Quality','Quality','Kcluster', 'Ecluster','True Label']
viz = wineTrain[cols]

kviz = viz.groupby('Kcluster').agg({'Quality':'mean'})
kviz.plot(kind="bar",legend=False)
plt.title("Percentage of High Quality Wine by Kmeans Cluster")
plt.xlabel("Cluster")
plt.ylabel("Percent of High Quality Wine")
plt.show()
plt.close()
                        
eviz = viz.groupby('Ecluster').agg({'Quality':'mean'})
eviz.plot(kind="bar",legend=False)
plt.title("Percentage of High Quality Wine by EM Cluster")
plt.xlabel("Cluster")
plt.ylabel("Percent of High Quality Wine")
plt.show()
plt.close()

#cluster chart kmeans
range_n_clusters = [2]
X = X_train
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

    ax2.set_title("Visualization of K Means Clusters")
    ax2.set_xlabel("Feature space for Fixed Acidity")
    ax2.set_ylabel("Feature space for Volatile Acidity")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()
plt.close()

range_n_clusters = [2]
X = X_train
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
    ax2.set_title("The visualization of EM Clusters")
    ax2.set_xlabel("Feature space for Fixed Acidity")
    ax2.set_ylabel("Feature space for Volatile Acidity")

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


krange = range(1,15)
sum_squared_errors = []
sum_squared_errors25 = []
sum_squared_errors50 = []
sum_squared_errors75 = []
sil = [0]

for k in krange:
  model = KMeans(n_clusters = k)
  model.fit_predict(X_train)
  sum_squared_errors.append(model.inertia_)
  
  for i in [25,50,75]:
      modelr = KMeans(n_clusters = k, random_state=i)
      modelr.fit_predict(X_train)
      
      if i == 25:
          sum_squared_errors25.append(modelr.inertia_)
      elif i == 50:
          sum_squared_errors50.append(modelr.inertia_)
      else:
          sum_squared_errors75.append(modelr.inertia_)
          
  if k >= 2:
      kmeans = KMeans(n_clusters = k).fit(X_train)
      labels = kmeans.labels_
      sil.append(silhouette_score(X_train, labels, metric = 'euclidean'))
  
plt.plot(krange,sum_squared_errors25, label='Seed: 25')
plt.plot(krange,sum_squared_errors50, label='Seed: 50')
plt.plot(krange,sum_squared_errors75, label='Seed: 75')
plt.title('K Clusters by Sum of Squared Errors')
plt.xlabel('K-Value')
plt.ylabel('Sum of Squared Errors')
plt.legend()
plt.show()
plt.close()


fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, sum_squared_errors, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Sum of Squared Errors", fontsize=14, color = 'b')
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14, color = 'g')
ax2.tick_params(axis="y")

fig.suptitle("K Clusters by SSE and Silhouette Score", fontsize=20)
plt.show()
plt.close()


logL = []
sil = [0]
for k in krange:
    gmm = GaussianMixture(n_components = k)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    logL.append(gmm.score(X_train))
    
    if k >= 2:
        sil.append(silhouette_score(X_train, labels, metric = 'euclidean'))

fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(krange, logL, color = 'b')
ax2.plot(krange, sil, color = 'g')

ax1.set_xlabel("K")
ax1.set_ylabel("Log Likelihood", fontsize=14, color = 'b')
ax1.tick_params(axis="y")

ax2.set_ylabel("Silhouette Score", fontsize=14, color = 'g')
ax2.tick_params(axis="y")

fig.suptitle("K Clusters by Log Likelihood and Silhoouette", fontsize=20)
plt.show()
plt.close()


####################################################################

kmeans = KMeans(n_clusters = 3).fit(X_train)
Klabels = kmeans.labels_

gmm = GaussianMixture(n_components = 2)
gmm.fit(X_train)
Elabels = gmm.predict(X_train)

wineTrain = pd.DataFrame(X_train, columns = cancer.iloc[:,:30].columns)
wineTrain['Kcluster'] = Klabels
wineTrain['Ecluster'] = Elabels
wineTrain['Diagnosis']  = y_train

cols = ['Diagnosis','Kcluster', 'Ecluster']
viz = wineTrain[cols]

kviz = viz.groupby('Kcluster').agg({'Diagnosis':'mean'})
kviz.plot(kind="bar",legend=False)
plt.title("Percentage of Positive Diagnosis by Kmeans Cluster")
plt.xlabel("Cluster")
plt.ylabel("Percent Malignant")
plt.show()
plt.close()
                             
eviz = viz.groupby('Ecluster').agg({'Diagnosis':'mean'})
eviz.plot(kind="bar",legend=False)
plt.title("Percentage of Positive Diagnosis by EM Cluster")
plt.xlabel("Cluster")
plt.ylabel("Percent Malignant")
plt.show()
plt.close()

#cluster chart kmeans
range_n_clusters = [2]
X = X_train
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

range_n_clusters = [2]
X = X_train
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
    ax2.set_title("The visualization of EM Clusters")
    ax2.set_xlabel("Feature space for Tumor Radius")
    ax2.set_ylabel("Feature space for Tumor Texture")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()
plt.close()







































