from typing import Dict, List

import cv2
import numpy as np
from sklearn import metrics

from nexusml.constants import ENGINE_MAX_NUM_CLASSES
from nexusml.constants import ENGINE_ODS_MIN_OVERLAP
from nexusml.engine.exceptions import DataError
from nexusml.enums import MLProblemType


def get_regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """
    Function that computes the regression metrics (MSE, MAE and R2)
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, ) with the predictions

    Returns:
        Dict[str, float] with the regression metrics ("mse", "mae" and "r2")
    """
    # Same number of instances
    if target.shape[0] != prediction.shape[0]:
        raise DataError('Different number of examples for target and prediction arrays')
    # Target and prediction only one dim or if it has two dim, the second one is of shape of 1
    if not (target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1)):
        raise DataError('Target array must be single dimension or have a single column')
    if not (prediction.ndim == 1 or (prediction.ndim == 2 and prediction.shape[1] == 1)):
        raise DataError('Predicted array must be single dimension or have a single column')
    # If arrays have two dimensions, remove the second one
    if target.ndim == 2:
        target = target[:, 0]
    if prediction.ndim == 2:
        prediction = prediction[:, 0]
    # Get MSE
    mse = metrics.mean_squared_error(y_true=target, y_pred=prediction)
    # Get MAE
    mae = metrics.mean_absolute_error(y_true=target, y_pred=prediction)
    # Get R2
    r2 = metrics.r2_score(y_true=target, y_pred=prediction)
    # Return as dict
    return dict(mse=mse, mae=mae, r2=r2)


def get_binary_classification_metrics(target: np.ndarray,
                                      prediction: np.ndarray,
                                      class_names: List[str],
                                      threshold: float = 0.5,
                                      pos_class_index: int = 1) -> Dict[str, float]:
    """
    Function that computes the binary classification metrics
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, 2) with the score for each class
        threshold (float): to determine when a prediction belongs to class 1 (predict score >= threshold ==> class 1)
        pos_class_index (int): the index (0 or 1) of the positive class to compute the ROC and PR curves
        class_names (List[str]): class names for naming the scores (for example, true_class0_pred_class1).

    Returns:
        Dict[str, float] with the binary classification metrics ("tp", "tn", "fp", "fn", "tpr", "tnr", "fnr", "fpr",
                "acc", "p", "f1", "auc" and "ap")
    """
    # Same number of instances
    if target.shape[0] != prediction.shape[0]:
        raise DataError('Different number of examples for target and prediction arrays')

    if target.ndim != 1:
        raise DataError('Expected target array with one single dimension')

    # Prediction of shape Nx2
    if prediction.ndim != 2 or prediction.shape[1] != 2:
        raise DataError('Expected a prediction array of two dimensions with two columns')

    # Assert target two values or fewer
    if np.unique(target).shape[0] > 2:
        raise DataError('Expected only two unique values on target')

    # Positive class must be 0 or 1
    if pos_class_index not in [0, 1]:
        raise Exception('Positive class index must be 0 or 1')

    # There are two class names
    if len(class_names) != 2:
        raise Exception('Expected only two classes')

    # Get predicted labels
    predicted_labels = (prediction[:, pos_class_index] >= threshold).astype(np.int32)
    predicted_labels = np.array([class_names[1 - pos_class_index], class_names[pos_class_index]])[predicted_labels]
    # Get confusion matrix
    cm = metrics.confusion_matrix(y_true=target, y_pred=predicted_labels, labels=class_names)
    # Get TP, FP, TN and FN
    tp = cm[pos_class_index, pos_class_index]
    tn = cm[1 - pos_class_index, 1 - pos_class_index]
    fp = cm[1 - pos_class_index, pos_class_index]
    fn = cm[pos_class_index, 1 - pos_class_index]
    # Get TPR (recall), TNR, FNR and FPR
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    fnr = 1 - tpr
    fpr = 1 - tnr
    # Get accuracy
    acc = metrics.accuracy_score(y_true=target, y_pred=predicted_labels)
    # Get precision and F1
    p = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    # Get AUC and AP
    # If there is a single class, the metrics cannot be calculated and a ValueError is thrown
    try:
        auc = metrics.roc_auc_score(y_true=target, y_score=prediction[:, pos_class_index])
        ap = metrics.average_precision_score(y_true=target,
                                             y_score=prediction[:, pos_class_index],
                                             pos_label=class_names[pos_class_index])
    except ValueError:
        auc = 0.0
        ap = 0.0
    # Return metrics as dict
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, tpr=tpr, tnr=tnr, fnr=fnr, fpr=fpr, acc=acc, p=p, f1=f1, auc=auc, ap=ap)


def get_multiclass_classification_metrics(target: np.ndarray, prediction: np.ndarray,
                                          class_names: List[str]) -> Dict[str, float]:
    """
    Function that computes the multiclass classification metrics
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, 2) with the score for each class
        class_names (List[str]): class names for naming the scores (for example, true_class0_pred_class1).

    Returns:
        Dict[str, float] with the multiclass classification metrics ("tp" for each class, "tpr" for each class,
                        miss-classifications (true_class{i}_pred_class{j}) and "acc")
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
    predicted_labels = np.array(class_names)[prediction.argmax(axis=1)]
    # Get confusion matrix
    cm = metrics.confusion_matrix(y_true=target, y_pred=predicted_labels)
    # Get TP by class only if there are ENGINE_MAX_NUM_CLASSES classes or fewer
    if cm.shape[0] <= ENGINE_MAX_NUM_CLASSES:
        tps = {f'tp_{class_names[i]}': cm[i, i] for i in range(cm.shape[0])}
    else:
        tps = {}
    # Get TPR by class (always)
    tprs = {
        f'tpr_{class_names[i]}': cm[i, i] / cm[i, :].sum() if cm[i, :].sum() != 0 else 0.0 for i in range(cm.shape[0])
    }
    # Get miss-classifications only when we have ENGINE_MAX_NUM_CLASSES classes or fewer
    miss_class = {}
    if cm.shape[0] <= ENGINE_MAX_NUM_CLASSES:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[0]):
                if i != j:
                    miss_class[f'true_{class_names[i]}_pred_{class_names[j]}'] = cm[i, j]

    # Get accuracy
    acc = metrics.accuracy_score(y_true=target, y_pred=predicted_labels)
    # Join all metrics in one dict
    metrics_dict = {}
    for k, v in tps.items():
        assert k not in metrics_dict
        metrics_dict[k] = v

    for k, v in tprs.items():
        assert k not in metrics_dict
        metrics_dict[k] = v

    for k, v in miss_class.items():
        assert k not in metrics_dict
        metrics_dict[k] = v

    # Add accuracy and return
    metrics_dict['acc'] = acc
    return metrics_dict


def get_object_detection_metrics(target: List, prediction: np.ndarray) -> Dict[str, float]:
    """
    Function that computes mean Average Precision for object detection
    Args:
        target (List): list of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, 2) with the score for each class

    Returns:
        Dict[str, float] with the object detection metrics ("mAP" , mean Average Precision)
    """

    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for tgt in target:
        # create ground-truth dictionary
        already_seen_classes = []
        for annotation in tgt:
            class_name = annotation['outputs'][0]['value']
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)
            annotation['used'] = False

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    min_overlap = ENGINE_ODS_MIN_OVERLAP

    count_true_positives = {}

    sum_AP = 0.0
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0

        # Examples containing at list one detection of the given class
        contains_class = []
        for example in prediction:
            if class_name in [pred_bbox['outputs'][0]['value']['category'] for pred_bbox in example]:
                contains_class.append(True)
            else:
                contains_class.append(False)

        idx_detections_class = np.where(list(contains_class))[0]

        tp = []  # creates an array of zeros of size nd
        fp = []

        for example_id in idx_detections_class:
            tgt = target[example_id]

            # For each detected bbox in the example
            for i in range(len(prediction[example_id])):
                # Check if the bbox is predicted as the current class
                if prediction[example_id][i]['outputs'][0]['value']['category'] == class_name:
                    overlap_max = -1
                    gt_match = -1
                    # Search matching real annotation
                    for annotation in tgt:
                        # Check if the real annotation is from the current class
                        if annotation['outputs'][0]['value'] == class_name:
                            bb = prediction[example_id][i]['polygon']
                            bb_x = [coord['x'] for coord in bb]
                            bb_y = [coord['y'] for coord in bb]
                            bbgt = annotation['polygon']
                            bbgt_x = [coord['x'] for coord in bbgt]
                            bbgt_y = [coord['y'] for coord in bbgt]
                            bi = [
                                max(min(bb_x), min(bbgt_x)),
                                max(min(bb_y), min(bbgt_y)),
                                min(max(bb_x), max(bbgt_x)),
                                min(max(bb_y), max(bbgt_y))
                            ]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (max(bb_x) - min(bb_x) + 1) * (max(bb_y) - min(bb_y) + 1) + (
                                    max(bbgt_x) - min(bbgt_x) + 1) * (max(bbgt_y) - min(bbgt_y) + 1) - iw * ih
                                overlap = iw * ih / ua

                                # Get real target with the higher overlap
                                if overlap > overlap_max:
                                    overlap_max = overlap
                                    gt_match = annotation

                    if overlap_max >= min_overlap:
                        if not gt_match['used']:
                            tp.append(1)
                            fp.append(0)
                            gt_match['used'] = True
                            count_true_positives[class_name] += 1
                        else:
                            tp.append(0)
                            fp.append(1)
                    else:
                        tp.append(0)
                        fp.append(1)
                        if overlap_max > 0:
                            status = 'INSUFFICIENT OVERLAP'

        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap

    mAP = sum_AP / n_classes
    # print("mAP = {0:.2f}%".format(mAP * 100))

    metrics_dict = {}

    # Add accuracy and return
    metrics_dict['mAP'] = mAP
    return metrics_dict


def get_object_segmentation_metrics(target: List, prediction: np.ndarray) -> Dict[str, float]:
    """
    Function that computes mean Average Precision for object detection
    Args:
        target (np.ndarray): array of (num_samples, ) with the target
        prediction (np.ndarray): array of (num_samples, 2) with the score for each class

    Returns:
        Dict[str, float] with the multiclass classification metrics ("tp" for each class, "tpr" for each class,
                        miss-classifications (true_class{i}_pred_class{j}) and "acc")
    """
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for tgt in target:
        # create ground-truth dictionary
        already_seen_classes = []
        for annotation in tgt:
            class_name = annotation['outputs'][0]['value']
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)
            annotation['used'] = False

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    min_overlap = ENGINE_ODS_MIN_OVERLAP

    prediction = np.array(prediction)
    count_true_positives = {}

    sum_AP = 0.0
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0

        # Examples containing at list one detection of the given class
        contains_class = []
        for example in prediction:
            if class_name in [pred_bbox['outputs'][0]['value']['category'] for pred_bbox in example]:
                contains_class.append(True)
            else:
                contains_class.append(False)

        idx_detections_class = np.where(list(contains_class))[0]

        tp = []  # creates an array of zeros of size nd
        fp = []

        for example_id in idx_detections_class:
            tgt = target[example_id]

            # For each detected bbox in the example
            for i in range(len(prediction[example_id])):
                # Check if the bbox is predicted as the current class
                if prediction[example_id][i]['outputs'][0]['value']['category'] == class_name:
                    overlap_max = -1
                    gt_match = -1
                    # Search matching real annotation
                    for annotation in tgt:
                        # Check if the real annotation is from the current class
                        if annotation['outputs'][0]['value'] == class_name:
                            bb = prediction[example_id][i]['polygon']
                            bbgt = annotation['polygon']

                            bb_x = [coord['x'] for coord in bb]
                            bb_y = [coord['y'] for coord in bb]
                            bbgt_x = [coord['x'] for coord in bbgt]
                            bbgt_y = [coord['y'] for coord in bbgt]

                            # Create pixel masks from the polygon coordinates
                            bb_mask = np.zeros((max(max(bb_y + bbgt_y)) + 1, max(max(bb_x + bbgt_x)) + 1),
                                               dtype=np.uint8)
                            bbgt_mask = np.zeros((max(max(bb_y + bbgt_y)) + 1, max(max(bb_x + bbgt_x)) + 1),
                                                 dtype=np.uint8)

                            # Get all pixels from the polygon vertices pixels
                            pts = np.array([[bb_x[i], bb_y[i]] for i in range(len(bb_x))], np.int32)
                            bb_mask = cv2.fillPoly(bb_mask, [pts], 1)

                            pts = np.array([[bbgt_x[i], bbgt_y[i]] for i in range(len(bbgt_x))], np.int32)
                            bbgt_mask = cv2.fillPoly(bbgt_mask, [pts], 1)

                            # compute overlap (IoU) = area of intersection / area of union
                            intersection = np.logical_and(bbgt_mask, bb_mask)
                            union = np.logical_or(bbgt_mask, bb_mask)
                            overlap = np.sum(intersection) / np.sum(union)

                            # Get real target with the higher overlap
                            if overlap > overlap_max:
                                overlap_max = overlap
                                gt_match = annotation

                    if overlap_max >= min_overlap:
                        if not gt_match['used']:
                            tp.append(1)
                            fp.append(0)
                            gt_match['used'] = True
                            count_true_positives[class_name] += 1
                        else:
                            tp.append(0)
                            fp.append(1)
                    else:
                        tp.append(0)
                        fp.append(1)
                        if overlap_max > 0:
                            status = 'INSUFFICIENT OVERLAP'

        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap

    mAP = sum_AP / n_classes
    # print("mAP = {0:.2f}%".format(mAP * 100))

    metrics_dict = {}

    # Add accuracy and return
    metrics_dict['mAP'] = mAP
    return metrics_dict


def voc_ap(rec: List[float], prec: List[float]):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    ##########################################################
    # This part makes the precision monotonically decreasing #
    #    (goes from the end to the beginning)                #
    #    matlab: for i=numel(mpre)-1:-1:1                    #
    #                mpre(i)=max(mpre(i),mpre(i+1));         #
    ##########################################################
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ################################################################
    # This part creates a list of indexes where the recall changes #
    #    matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;             #
    ################################################################
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    ##########################################################
    # The Average Precision (AP) is the area under the curve #
    #    (numerical integration)                             #
    #    matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));       #
    ##########################################################
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def get_metrics(problem_type: MLProblemType, target_name: str, **kwargs) -> Dict[str, float]:
    """
    Function that given the problem type computes its metrics
    Args:
        problem_type (MLProblemType): enum that indicates the problem type
        target_name (str): the target feature name for renaming the metric names. For example, rename "r2" to "price_r2"
        **kwargs (dict): all needed arguments for each different function that computes metrics

    Returns:
        Dict[str, float] with metrics for the given problem type
    """
    if problem_type == MLProblemType.REGRESSION:
        computed_metrics = get_regression_metrics(**kwargs)
    elif problem_type == MLProblemType.BINARY_CLASSIFICATION:
        computed_metrics = get_binary_classification_metrics(**kwargs)
    elif problem_type == MLProblemType.MULTI_CLASS_CLASSIFICATION:
        computed_metrics = get_multiclass_classification_metrics(**kwargs)
    else:
        raise Exception(f'Unknown problem type {problem_type}')

    computed_metrics = dict(list(map(lambda x: (f'{target_name}_{x[0]}', x[1]), computed_metrics.items())))
    return computed_metrics
