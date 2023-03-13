from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def createConfusionMatrix(y_true, y_pred):
    # Classes from 0 to num_classes-1
    num_classes = np.max(y_true) + 1
    classes = [i for i in range(num_classes)]

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix /= np.sum(cf_matrix)
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=[i for i in classes],
    #                      columns=[i for i in classes])
    plt.figure(figsize=(2*num_classes, 2*num_classes))    
    return sn.heatmap(cf_matrix, annot=True).get_figure()