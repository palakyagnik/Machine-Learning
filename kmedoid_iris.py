import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

# Load Iris Dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Perform K-Means Clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
kmeans.fit(x)

# Get Cluster Labels
labels = kmeans.labels_

# Evaluate Clustering Performance
print("Confusion Matrix:")
print(confusion_matrix(y, labels))

print("\nClassification Report:")
print(classification_report(y, labels))

# Visualize Clusters
plt.scatter(x[:,0], x[:,1], c=labels, s=50, cmap='viridis')
plt.title("K-Means Clustering of Iris Dataset")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()
