import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

def plot_roc(true_labels, target_scores):
    """ 
        Function plots ROC curve and calculates AUC 
        for given true_labels and target_scores. 
    """
    fpr, tpr, threshold = roc_curve(true_labels, target_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, threshold, roc_auc

def compute_eer(fpr, tpr, threshold):
    """ 
        Returns equal error rate (EER) and
        the corresponding threshold. 
    """
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, threshold)(eer)
    return eer


def report(eval_asvspoof):
    """
        Report results for model
    """
    # divide data
    train_data = [eval_asvspoof["MFCC"][0], eval_asvspoof["CQCC"][0], eval_asvspoof["LPS"][0]]
    test_data = [eval_asvspoof["MFCC"][1][:-2], eval_asvspoof["CQCC"][1][:-2], eval_asvspoof["LPS"][1][:-2]]
    test_data_21 = [eval_asvspoof["MFCC"][2][:-2], eval_asvspoof["CQCC"][2][:-2], eval_asvspoof["LPS"][2][:-2]]
    
    # Column and index names
    columns = ['Loss', 'Balanced Accuracy', 'Precision', 'Recall']
    index = ['MFCC', 'CQCC', 'LPS']
    
    # Create DataFrame
    results_train = pd.DataFrame(train_data, columns=columns, index=index)
    results_test = pd.DataFrame(test_data, columns=columns, index=index)
    results_test_21 = pd.DataFrame(test_data_21, columns=columns, index=index)

    print(results_train)
    print(results_test)
    print(results_test_21)

    # colors
    colors = ['darkorange', 'green', 'blue']
    labels = ['ASVspoof19', 'ASVspoof21']
    
    # plot ROC and calculate AUC
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    lw = 2
    colors = ['darkorange', 'green', 'blue']
    
    for i in range(1, 3):
        for j, (key, vals) in enumerate(eval_asvspoof.items()):
            fpr, tpr, threshold, roc_auc = plot_roc(eval_asvspoof[key][i][5], eval_asvspoof[key][i][4])
            eer = compute_eer(fpr, tpr, threshold) * 100
            ax[i - 1].plot(fpr, tpr, color=colors[j], lw=lw, label=f'{key}_ROC curve (area = {roc_auc:.3f}, EER = {int(eer)} %)')
        ax[i - 1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax[i - 1].set_xlim([0.0, 1.0])
        ax[i - 1].set_ylim([0.0, 1.05])
        ax[i - 1].set_xlabel('False Positive Rate')
        ax[i - 1].set_ylabel('True Positive Rate')
        ax[i - 1].set_title(f'ROC Curve for Class 1, {labels[i - 1]}')
        ax[i - 1].legend(loc="lower right")