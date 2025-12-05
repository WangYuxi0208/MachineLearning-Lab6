import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def run_kmeans(X, k, save_left, save_right):
    km = KMeans(n_clusters=k, random_state=0)
    y_hat = km.fit_predict(X)

    # 图 1：含质心
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y_hat)
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
                marker='x', s=200)
    plt.title(f'K={k} 聚类结果（含质心）')
    plt.savefig(save_left)
    plt.close()

    # 图 2：不含质心
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y_hat)
    plt.title(f'K={k} 聚类结果（不含质心）')
    plt.savefig(save_right)
    plt.close()
