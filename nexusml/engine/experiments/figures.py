from typing import Dict, List

import matplotlib

# Note: This pylint directive is disabled because `matplotlib.use('Agg')`
# must be executed before importing `matplotlib.pyplot`.
# pylint: disable-next=wrong-import-position,
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

from nexusml.constants import ENGINE_MAX_NUM_CLASSES
from nexusml.engine.exceptions import DataError
from nexusml.enums import MLProblemType


def get_sorted_target_prediction_plot(target: np.ndarray, prediction: np.ndarray, title: str = None) -> plt.Figure:
    """
    Function that returns a plot with sorted the target and plot a line with target and prediction or each label
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, ) with the predictions
        title (str): optional title for the graph

    Returns:
        plt.Figure object with the generated plot
    """

    if target.ndim == 2:
        if target.shape[1] != 1:
            raise DataError('More than one element given on the target array')
        target = target[:, 0]

    if prediction.ndim == 2:
        if prediction.shape[1] != 1:
            raise DataError('More than one element given on the prediction array')
        prediction = prediction[:, 0]

    # Assert that we have 2 dimensions and both arrays have the same shape
    if target.shape[0] != prediction.shape[0]:
        raise DataError('Different number of examples for target and prediction arrays')

    # Sort values by target
    sort_indices = np.argsort(target)

    # Plot
    figure = plt.figure()
    plt.plot(np.arange(target.shape[0]), prediction[sort_indices], label='prediction')
    plt.plot(np.arange(target.shape[0]), target[sort_indices], label='target')
    plt.legend()
    # Add title if it is None
    if title is not None:
        plt.title(title)

    return figure


def plot_r2(target: np.ndarray, prediction: np.ndarray, title: str = None) -> plt.Figure:
    """
    Function that returns a r2 (r squared) plot for a given target and prediction
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, ) with the predictions
        title (str): optional title for the graph

    Returns:
        Figure with the r2 (r squared) plot

    """
    if target.ndim == 2:
        if target.shape[1] != 1:
            raise DataError('More than one element given on the target array')
        target = target[:, 0]

    if prediction.ndim == 2:
        if prediction.shape[1] != 1:
            raise DataError('More than one element given on the prediction array')
        prediction = prediction[:, 0]

    # Assert that we have 2 dimensions and both arrays have the same shape
    if target.shape[0] != prediction.shape[0]:
        raise DataError('Different number of examples for target and prediction arrays')

    # Get r2 score
    r2_score = metrics.r2_score(y_true=target, y_pred=prediction)

    # Plot
    figure = plt.figure()
    # Plot target and prediction
    plt.scatter(x=target, y=prediction)
    # Fit a line on target and prediction
    poly = np.polyfit(x=target, y=prediction, deg=1)
    # Fit the with the line
    line_pred = np.poly1d(poly)(np.unique(target))
    # Plot the line
    plt.plot(np.unique(target), line_pred)
    # Add R2 score
    plt.text(0.6, 0.5, f'R-squared = {r2_score}:.4f')
    # Add title if it is None
    if title is not None:
        plt.title(title)
    # Return figure
    return figure


def compute_roc_curve(target: np.ndarray,
                      prediction: np.ndarray,
                      title: str = None,
                      threshold: float = 0.5) -> plt.Figure:
    """
    Returns the ROC curve
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, ) with the predictions (scores) for the positive class
        title (str): optional title for the graph
        threshold (float): decision threshold to be shown in red
    Returns:
        plt.Figure: ROC curve
    """
    plt.rcParams.update({'font.size': 16})

    # Get TPR and FPR for the given threshold
    cm = metrics.confusion_matrix(y_true=target, y_pred=prediction >= threshold)
    tn, fn, tp, fp = cm[0, 0], cm[1, 0], cm[1, 1], cm[0, 1]
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = 1 - tnr

    # ROC and AUC
    auc = metrics.roc_auc_score(y_true=target, y_score=prediction)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=target, y_score=prediction)
    fig_roc = plt.figure(figsize=(20, 15))
    plt.plot(fprs, tprs, color='aqua', lw=2, label='ROC curve')

    # Draw threshold point
    plt.scatter(x=fpr, y=tpr, s=28, color='red', zorder=10, label='Decision threshold')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title is None:
        plt.title(f'ROC curve (AUC = {auc:.4f})')
    else:
        plt.title(title)
    plt.legend(loc='lower right')
    plt.close()
    return fig_roc


def compute_pr_curve(target: np.ndarray,
                     prediction: np.ndarray,
                     title: str = None,
                     threshold: float = 0.5) -> plt.Figure:
    """
    Returns the PR curve
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, ) with the predictions (scores) for the positive class
        title (str): optional title for the graph
        threshold (float): decision threshold to be shown in red
    Returns:
        plt.Figure: PR curve
    """
    plt.rcParams.update({'font.size': 16})

    # Get precision and recall for the given threshold
    p = metrics.precision_score(y_true=target, y_pred=prediction >= threshold)
    r = metrics.recall_score(y_true=target, y_pred=prediction >= threshold)

    # Precision-Recall curve
    avg_prec = metrics.average_precision_score(y_true=target, y_score=prediction)
    fig_pr = plt.figure(figsize=(20, 15))
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true=target, probas_pred=prediction)
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    plt.scatter(x=r, y=p, s=28, color='red', zorder=10, label='Decision threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    if title is None:
        plt.title(f'Precision-Recall curve (average precision score = {avg_prec:.4f})')
    else:
        plt.title(title)
    plt.close()
    return fig_pr


def compute_confusion_matrix_figure(target: np.ndarray,
                                    prediction: np.ndarray,
                                    title: str = None,
                                    class_names: List[str] = None) -> plt.Figure:
    """
    Returns the PR curve
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, ) with the predictions
        title (str): optional title for the graph
        class_names (List[str]): class names for showing them in the confusion matrix figure
    Returns:
        plt.Figure: ConfusionMatrix as figure
    """
    plt.rcParams.update({'font.size': 16})

    cm_display = ConfusionMatrixDisplay.from_predictions(y_true=target, y_pred=prediction, labels=class_names)
    if title is None:
        plt.title('Confusion matrix')
    else:
        plt.title(title)
    return cm_display.figure_


def get_regression_figures(target: np.ndarray, prediction: np.ndarray, title: str = None) -> Dict[str, plt.Figure]:
    """
    Function that computes all regression figures (R2 and sorted predictions) given the target and prediction
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, ) with the predictions
        title (str): optional title for the graph

    Returns:
        Dict[str, Figure] with the "sorted_plot" and "r2_plot" figures
    """
    return dict(sorted_plot=get_sorted_target_prediction_plot(target=target, prediction=prediction, title=title),
                r2_plot=plot_r2(target=target, prediction=prediction, title=title))


def get_binary_classification_figures(target: np.ndarray,
                                      prediction: np.ndarray,
                                      class_names: List[str],
                                      title: str = None,
                                      threshold: float = 0.5,
                                      pos_class_index: int = 1) -> Dict[str, plt.Figure]:
    """
    Function that computes all binary classification figures (ROC, PR and Confusion Matrix)
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, 2) with the score for each class
        title (str): optional title for the graph
        threshold (float): to determine when a prediction belongs to class 1 (predict score >= threshold ==> class 1)
        pos_class_index (int): the index (0 or 1) of the positive class to compute the ROC and PR curves
        class_names (List[str]): class names for showing them in the confusion matrix figure

    Returns:
        Dict[str, Figure] with the "roc", "pr" and "cm" figures
    """
    if target.shape[0] != prediction.shape[0]:
        raise DataError('Different number of examples for target and prediction arrays')

    if target.ndim != 1:
        raise DataError('Expected target array with one single dimension')

    # Prediction of shape Nx2
    if prediction.ndim != 2 or prediction.shape[1] != 2:
        raise DataError('Expected a prediction array of two dimensions with two columns')

    # Assert target only two values
    if np.unique(target).shape[0] > 2:
        raise DataError('Expected only two unique values on target')

    # Positive class must be 0 or 1
    if pos_class_index not in [0, 1]:
        raise Exception('Positive class index must be 0 or 1')

    # There are two class names
    if len(class_names) != 2:
        raise Exception('Expected only two classes')

    predicted_labels = (prediction[:, pos_class_index] >= threshold).astype(np.int32)
    predicted_labels = np.array([class_names[1 - pos_class_index], class_names[pos_class_index]])[predicted_labels]
    figures_dict = dict(cm=compute_confusion_matrix_figure(
        target=target, prediction=predicted_labels, title=title, class_names=class_names))
    try:
        figures_dict['roc'] = compute_roc_curve(target=(target == class_names[pos_class_index]).astype(np.int32),
                                                prediction=prediction[:, pos_class_index],
                                                title=title,
                                                threshold=threshold)
        figures_dict['pr'] = compute_pr_curve(target=(target == class_names[pos_class_index]).astype(np.int32),
                                              prediction=prediction[:, pos_class_index],
                                              title=title,
                                              threshold=threshold)
    except Exception:
        pass
    return figures_dict


def get_multiclass_classification_figures(target: np.ndarray,
                                          prediction: np.ndarray,
                                          class_names: List[str],
                                          title: str = None) -> Dict[str, plt.Figure]:
    """
    Function that computes all multiclass classification figures (Confusion Matrix)
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, n_classes) with the score for each class
        title (str): optional title for the graph
        class_names (List[str]): class names for naming the scores (for example, true_class0_pred_class1).


    Returns:
        Dict[str, Figure] with the "cm" figure
    """
    # Same number of instances
    if target.shape[0] != prediction.shape[0]:
        raise DataError('Different number of examples for target and prediction arrays')
    # Target only one dim
    if target.ndim != 1:
        raise DataError('Expected target array with one single dimension')
    # Prediction two dimensions
    if prediction.ndim != 2:
        raise DataError('Expected a prediction array of two dimensions')
    # The length of the list must be the same as the shape of predictions
    if len(class_names) != prediction.shape[1]:
        raise DataError('Different number of classes and predicted values')
    # Only create confusion matrix if there are ENGINE_MAX_NUM_CLASSES classes or fewer
    # It makes no sense to draw a big CM where we cannot see anything
    if len(class_names) <= ENGINE_MAX_NUM_CLASSES:
        predicted_labels = np.array(class_names)[prediction.argmax(axis=1)]
        return dict(cm=compute_confusion_matrix_figure(target=target, prediction=predicted_labels, title=title))
    else:
        return dict()


def get_figures(problem_type: MLProblemType, target_name: str, **kwargs) -> Dict[str, plt.Figure]:
    """
    Function that given the problem type computes its figures
    Args:
        problem_type (MLProblemType): enum that indicates the problem type
        target_name (str): the target feature name for renaming the figure names. For example, rename "r2" to "price_r2"
        **kwargs (dict): all needed arguments for each different function that computes figures

    Returns:
        Dict[str, Figure] with figures of the given problem type
    """
    if problem_type == MLProblemType.REGRESSION:
        figures = get_regression_figures(**kwargs)
    elif problem_type == MLProblemType.BINARY_CLASSIFICATION:
        figures = get_binary_classification_figures(**kwargs)
    elif problem_type == MLProblemType.MULTI_CLASS_CLASSIFICATION:
        figures = get_multiclass_classification_figures(**kwargs)
    else:
        raise Exception(f'Unknown problem type {problem_type}')

    figures = dict(list(map(lambda x: (f'{target_name}_{x[0]}', x[1]), figures.items())))
    return figures
