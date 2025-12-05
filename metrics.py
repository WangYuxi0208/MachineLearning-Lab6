import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=range(10), columns=range(10))

    plt.figure(figsize=(6,5))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")  # 蓝色
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()
