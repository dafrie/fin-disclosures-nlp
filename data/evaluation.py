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
        label = map_to_field()[idx]
        color = map_to_field('color')[idx]
        linestyle = map_to_field('linestyle')[idx]

        # calculate the P-R curve
        fpr[idx], tpr[idx], threshold[idx] = roc_curve(
            class_labels, class_probs)

        # AP is the same as P-R AuC...
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
        color = map_to_field('color')[idx]
        linestyle = map_to_field('linestyle')[idx]

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


def threshold_moving_report(labels, preds_probs, averaging="macro"):
    # 1) Calculate AuC evaluation metrics
    roc_auc = roc_auc_score(labels, preds_probs, average=averaging)
    pr_auc = average_precision_score(labels, preds_probs, average=averaging)

    # 2) Plot curves
    sns.set_theme(style="ticks", rc={'text.usetex': True})
    sns.set_context("paper")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    best_roc_threshold = plot_roc_curve(labels, preds_probs, axes[0])
    best_pr_threshold = plot_pr_curve(labels, preds_probs, axes[1])

    scores = pd.DataFrame(data={"ROC AuC": [roc_auc], "PR AuC": [pr_auc]})

    return scores, best_roc_threshold, best_pr_threshold


def test_evaluation_report(labels, preds_probs, thresholds, averaging="macro"):
    # 1) Calculate AuC evaluation metrics
    roc_auc = roc_auc_score(labels, preds_probs, average=averaging)
    pr_auc = average_precision_score(labels, preds_probs, average=averaging)
    scores = pd.DataFrame(data={"ROC AuC": [roc_auc], "PR AuC": [pr_auc]})

    # Set labels based on provided threshold values
    preds_bool = preds_probs > np.array([i for i in thresholds])

    # Print the classification evaluation metrics
    print(f"Using threshold values {thresholds}")
    print(classification_report(labels, preds_bool, target_names=labels.columns))

    # Plot confusion matrix
    cms = multilabel_confusion_matrix(labels, preds_bool)
    print(cms)

    return scores


def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix
