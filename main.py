from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

data = load_iris()
inputs = data.data
results = data.target

x,x_test,y,y_test = train_test_split(inputs, results, test_size = 0.1)

def find_closest_centroids(x,centriods):
    K = centriods.shape[0]

    idx = np.zeros(x.shape[0], dtype=int)

    for i in range(x.shape[0]):
        distance = []
        for j in range(K):
            norm_ij = np.linalg.norm(x[i] - centriods[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)

    return idx

def compute_centroids(x,idx,K):
    m,n = x.shape

    centriods = np.zeros((K,n))

    for k in range(K):
        points = x[idx==k]
        centriods[k] = np.mean(points, axis = 0)

    return centriods

def kMean_init_centroids(x,K):
    randinx = np.random.permutation(x.shape[0])
    centriods = x[randinx[:K]]
    return centriods

def compute_kmeans(x,initial_centroids,max_iters=10):
    m,n = x.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids.copy()
    idx = np.zeros(m)

    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(x,centroids)
        centriods = compute_centroids(x,idx,K)

    return centriods, idx

i_centroids = kMean_init_centroids(x,3)
max_iters = 20
centriods,idx = compute_kmeans(x,i_centroids,max_iters)

def predict_cluster(x_new, centroids):
    distances = np.linalg.norm(centroids - x_new, axis=1)
    return np.argmin(distances)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, accuracy_score

# Reduce dimensions to 2D for plotting
pca = PCA(n_components=2)
x_2d = pca.fit_transform(x_test)

# Predict cluster for each point using final centroids
predicted_clusters = np.array([predict_cluster(xi, centriods) for xi in x_test])

conf_matrix = confusion_matrix(y_test, predicted_clusters)

# Use Hungarian algorithm to best map predicted clusters to actual classes
row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # maximize accuracy
mapping = dict(zip(col_ind, row_ind))

# Remap predicted clusters
mapped_preds = np.array([mapping[c] for c in predicted_clusters])

# Compute accuracy
acc = accuracy_score(y_test, mapped_preds)
print(f"Clustering Accuracy: {acc:.2f}")
