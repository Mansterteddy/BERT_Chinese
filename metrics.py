import numpy as np
from sklearn import metrics

def calAUC(out, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, out, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    return auc_score

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs

if __name__ == "__main__":
    labels = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.4, 0.3, 0.8])
    auc_score = calAUC(scores, labels)
    print("auc_score: ", auc_score)