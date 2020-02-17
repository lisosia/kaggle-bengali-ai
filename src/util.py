import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_cmx(y_true, y_pred, outpath, figsize=(20,20)):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(df_cmx, annot=True)
    plt.savefig(outpath)

