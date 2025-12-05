import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from knn import KNN
from metrics import plot_confusion

# 加载数据
X, y = load_digits(return_X_y=True)

# 图 6：前 32 张图
plt.figure(figsize=(8,4))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.imshow(X[i].reshape(8,8), cmap="gray")
    plt.axis("off")
plt.savefig("knn_digits.png")
plt.close()

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练 KNN
knn = KNN(k=6, X_train=X_train, y_train=y_train)
y_pred = knn.predict(X_test)

# 图 7：蓝色混淆矩阵
plot_confusion(y_test, y_pred, "confusion_matrix.png")

print("KNN 图像生成完毕，共 2 张")
