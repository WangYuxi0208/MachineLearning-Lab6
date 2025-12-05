import numpy as np
import matplotlib.pyplot as plt
from kmeans import run_kmeans

# 生成二维数据
np.random.seed(0)
X = np.vstack([
    np.random.randn(100,2) + np.array([0,0]),
    np.random.randn(100,2) + np.array([5,5]),
    np.random.randn(100,2) + np.array([0,5])
])

# 图 1：原始散点图
plt.scatter(X[:,0], X[:,1])
plt.title("原始散点图")
plt.savefig("kmeans_original.png")
plt.close()

# k=2
run_kmeans(X, 2, "kmeans_k2_left.png", "kmeans_k2_right.png")

# k=4
run_kmeans(X, 4, "kmeans_k4_left.png", "kmeans_k4_right.png")

print("K-Means 图像生成完毕，共 5 张")
