import os
import json
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, classification_report, multilabel_confusion_matrix

from .constants import map_to_field


def plot_roc_curve(labels, preds_probs, ax):
    """Plots the ROC curve for multi-label classification predictions and returns the J-Stat maximizing threshold values of each class

    Parameters
    ----------
    labels : Array of true label values of size of # of categories

    preds_probs : Array of probabilities of size of # of categories

    ax : A optional plt axis
    """
    roc_auc = dict()
    tpr = dict()
    fpr = dict()
    threshold = dict()
    best_j_stat = dict()
    best_threshold = dict()

    if not ax:
        ax = plt

    counter = 0
    for (idx, class_labels) in labels.iteritems():
        # Get class probabilites for the current class
        class_probs = preds_probs[:, counter]
        label = map_to_field().get(idx, "Relevant")
        color = map_to_field('color').get(idx, "black")
        linestyle = map_to_field('linestyle').get(idx, "-")

        # calculate the roc curve
        fpr[idx], tpr[idx], threshold[idx] = roc_curve(
            class_labels, class_probs)

        # calc the score
        roc_auc[idx] = roc_auc_score(class_labels, class_probs)

        # Plot the class curve
        ax.plot(fpr[idx], tpr[idx], color, linestyle=linestyle, lw=2,
                label=f'{label} ROC AuC: {roc_auc[idx]:.3f}')

        # Get the max Youden's J stat,
        # Since J = Sensititivty + Specificity - 1 --> Sens + (1- FPR) - 1
        J_stat = tpr[idx] - fpr[idx]

        max_threshold_idx = np.argmax(J_stat)
        best_threshold[idx] = threshold[idx][max_threshold_idx]
        best_j_stat[idx] = J_stat[max_threshold_idx]

        # Add a dot for the F1 optimizing threshold value
        ax.scatter(fpr[idx][max_threshold_idx], tpr[idx]
                   [max_threshold_idx], marker='o', color=color, label=f'J-Stat: {best_j_stat[idx]:.3f} | Threshold: {best_threshold[idx]:.3f}')

        counter += 1

    # Add baseline/no skill line
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Recall (TPR)")
    ax.legend(loc="best")
    ax.set_title("ROC curve")
    return best_threshold


def plot_pr_curve(labels, preds_probs, ax):
    """Plots the Precision-Recall curve for multi-label classification predictions and returns the F1-Score maximizing threshold values of each class

    Parameters
    ----------
    labels : Array of true label values of size of # of categories

    preds_probs : Array of probabilities of size of # of categories

    ax : A optional plt axis
    """
    pr_auc = dict()
    precision = dict()
    recall = dict()
    threshold = dict()
    best_f1_score = dict()
    best_threshold = dict()

    if not ax:
        ax = plt

    counter = 0
    for (idx, class_labels) in labels.iteritems():
        # Get class probabilites for the current class
        class_probs = preds_probs[:, counter]

        label = map_to_field().get(idx, "Relevant")
        color = map_to_field('color').get(idx, "black")
        linestyle = map_to_field('linestyle').get(idx, "-")

        # Average precision is the same as P-R AuC...
        pr_auc[idx] = average_precision_score(class_labels, class_probs)

        # calculate the P-R curve
        precision[idx], recall[idx], threshold[idx] = precision_recall_curve(
            class_labels, class_probs)

        # Plot the class curve
        ax.plot(recall[idx], precision[idx], color, linestyle=linestyle, lw=2,
                label=f'{idx} P-R AuC: {pr_auc[idx]:.3f}')

        # Get the F1-maximizing threshold
        f1_scores = (2 * precision[idx] * recall[idx]
                     ) / (precision[idx] + recall[idx])

        max_threshold_idx = np.argmax(f1_scores)
        best_threshold[idx] = threshold[idx][max_threshold_idx]
        best_f1_score[idx] = f1_scores[max_threshold_idx]

        # Add a dot for the F1 optimizing threshold value
        ax.scatter(recall[idx][max_threshold_idx], precision[idx]
                   [max_threshold_idx], marker='o', color=color, label=f'F1: {best_f1_score[idx]:.3f} | Threshold: {best_threshold[idx]:.3f}')

        counter += 1

    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.set_title("Precision-Recall curve")
    return best_threshold


def threshold_moving_report(labels, preds_probs, averaging="macro", export_path=None):
    try:
        is_binary = False
        np.shape(labels)[1]
    except IndexError:
        is_binary = True
        preds_probs = preds_probs[:, 1]
        labels = labels.to_frame(name="Relevant")

    # 1) Calculate AuC evaluation metrics
    roc_auc = roc_auc_score(labels, preds_probs, average=averaging)
    pr_auc = average_precision_score(labels, preds_probs, average=averaging)

    # 2) Plot curves
    sns.set_theme(style="ticks", rc={'text.usetex': True})
    sns.set_context("paper")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    corr_probs = np.reshape(preds_probs, (-1, 1)) if is_binary else preds_probs
    best_roc_threshold = plot_roc_curve(labels, corr_probs, axes[0])
    best_pr_threshold = plot_pr_curve(labels, corr_probs, axes[1])

    if export_path:
        fig.savefig(export_path)

    scores = pd.DataFrame(data={"ROC AuC": [roc_auc], "PR AuC": [pr_auc]})

    return scores, best_roc_threshold, best_pr_threshold


def test_evaluation_report(labels, preds_probs, thresholds, averaging="macro", export_path=None):

    try:
        is_binary = False
        np.shape(labels)[1]
    except IndexError:
        is_binary = True
        preds_probs = preds_probs[:, 1]
        labels = labels.to_frame(name="Relevant")

    # 1) Calculate AuC evaluation metrics
    roc_auc = roc_auc_score(labels, preds_probs, average=averaging)
    pr_auc = average_precision_score(labels, preds_probs, average=averaging)

    # Set labels based on provided threshold values
    preds_bool = preds_probs > np.array([i for i in thresholds])

    # Print the classification evaluation metrics
    print(f"Using threshold values {thresholds}")
    print(classification_report(labels, preds_bool,
                                target_names=labels.columns if not is_binary else None))
    cls_report = classification_report(
        labels, preds_bool, labels=[1] if is_binary else None, target_names=labels.columns if not is_binary else None, output_dict=True)

    # Plot confusion matrix
    mcm = multilabel_confusion_matrix(labels, preds_bool)
    if is_binary:
        mcm = mcm[1:]
    # The cm output of scikit-learn is flipped...
    flipped_mcm = []
    for i in mcm:
        flipped_mcm.append(np.flip(np.rot90(np.fliplr(i))).tolist())

    sns.set_theme(style="ticks", rc={'text.usetex': True})
    sns.set_context("paper")

    fig = plot_cm_grid(flipped_mcm, labels.columns)

    if export_path:
        fig.savefig(export_path)

    scores = pd.DataFrame(data={"ROC AuC": [roc_auc], "PR AuC": [
                          pr_auc], "F1": [cls_report["macro avg"]["f1-score"]], "Report": [json.dumps(cls_report)], "CMS": [json.dumps(flipped_mcm)]})

    return scores


def plot_cm_grid(mcm, class_labels, ncols=2):

    figure, axes = plt.subplots(ncols=ncols, nrows=(
        math.ceil(len(mcm) / ncols)), sharex=True, sharey=False)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(mcm):
            figure.delaxes(ax)
            continue
        cm = mcm[idx]
        subplot = sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"],
                              ax=ax, vmin=0, cbar=False)
        subplot.set(
            title=f"{class_labels[idx]}", xlabel="Actual (j)", ylabel="Predicted (i)")
    # plt.tight_layout()
    plt.show()
    return figure


class Results:
    def __init__(self, path, params):
        self.path = path
        self.params = params
        self.id = params["scenario"] + "_" + params["model_name"]
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
            self.df = self.df.set_index("id")
        else:
            self.df = pd.DataFrame(data={"id": [self.id]})
            self.df = self.df.set_index("id")
            self.save()

    def save(self):
        # TODO: Do we need to reset the index before saving?
        self.df.to_csv(self.path, index_label="id")

    def log_experiment(self, data, prefix=None):
        self.load()
        # If not a dict, assume its a dataframe and convert the first row
        if type(data) is not dict:
            data = data.to_dict('records')[0]

        keys = [k if not prefix else prefix + "_" + k for k in data.keys()]
        self.df.loc[self.id, keys] = data.values()
        self.save()

    def export_report(self):
        pass
