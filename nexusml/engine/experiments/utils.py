from operator import gt
from operator import lt
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.utils import get_shapes_from_targets
from nexusml.engine.exceptions import DataError
from nexusml.engine.experiments import figures
from nexusml.engine.experiments import metrics
from nexusml.enums import MLProblemType


def _update_dict(g_dict: Dict, l_dict: Dict):
    """
    Function that is used to update a given dictionary, including all elements of other dictionary
    Note: the keys of the second dictionary must not exist on the first dictionary
    Args:
        g_dict (Dict): dict where store all the values
        l_dict (Dict): dict from where get the information to copy to the other dict

    Returns:
        first dictionary updated with the information of the second dictionary
    """
    # For each item in the second dictionary
    for k, v in l_dict.items():
        # Assert that the key does not exist in the first dictionary
        if k in g_dict:
            raise DataError(f"Key '{k}' already exists on dict")
        # Set the item in the first dictionary
        g_dict[k] = v
    return g_dict


def get_metrics_and_figures(targets: Union[Dict, pd.DataFrame], predictions: Dict, outputs: List[Dict],
                            output_transforms: ElementTransforms, prefix: str) -> Tuple[Dict, Dict]:
    """
    Function that get all metrics and figures of all given outputs
    Args:
        targets (Union[Dict, pd.DataFrame]): dict or DataFrame where we can get the targets by their name
        predictions (Dict): dict where the key is the output id and the value is either a DataFrame or a numpy
                        array with the output predictions
        outputs (List[Dict]): list with the output elements information
        output_transforms (ElementTransforms): transformations that are applied on each output element
                                            so we can get the problem type
        prefix (str): prefix to use in the keys of the dictionaries that will be return (like 'train' or 'test')

    Returns:
        Tuple of:
            - Dict: dict with all the metrics for all outputs
            - Dict: dict with all the figures for all outputs
    """
    # Create empty dicts for metrics and figures
    metrics_dict = {}
    figures_dict = {}
    # For each output
    for o in outputs:
        # Don't get predictions if it is a non-required output
        # If it is not required, and it is not on predictions, go to next
        if not o['required'] and o['name'] not in predictions:
            continue
        # If this is reach, the element is required, so assert that it is on predictions
        # Could be also that the element is not required, but it is on predictions
        if o['name'] not in predictions:
            raise DataError(f"Element '{o['name']}' not in predictions")
        # Get target and predictions. If they are not numpy arrays, convert to them
        out_target = targets[o['name']]
        if not isinstance(out_target, np.ndarray):
            out_target = out_target.to_numpy()
        out_prediction = predictions[o['name']]
        if isinstance(out_prediction, pd.DataFrame):
            out_prediction = out_prediction.to_numpy()

        # Get the transformation output info to get the problem type
        tfm_out_info = output_transforms.get_transform(o['name']).get_transform_output_info()
        # Ensure that the problem type is set
        if tfm_out_info.output_problem_type is None:
            raise DataError('The output problem type is None')

        # Get metrics based on the problem type
        if tfm_out_info.output_problem_type == MLProblemType.BINARY_CLASSIFICATION:
            # Assert that we have two columns in the prediction
            if predictions[o['name']].shape[1] != 2:
                raise DataError('Expected two columns for BinaryClassification problem')
            # Get metrics and figures
            sub_m = metrics.get_binary_classification_metrics(target=out_target,
                                                              prediction=out_prediction,
                                                              class_names=predictions[o['name']].columns.tolist())
            sub_f = figures.get_binary_classification_figures(target=out_target,
                                                              prediction=out_prediction,
                                                              class_names=predictions[o['name']].columns.tolist())
        elif tfm_out_info.output_problem_type == MLProblemType.MULTI_CLASS_CLASSIFICATION:
            # Assert that we have more than two columns in the prediction
            if predictions[o['name']].shape[1] <= 2:
                raise DataError('Expected more than two columns for MultiClassClassification problem')
            # Get metrics
            sub_m = metrics.get_multiclass_classification_metrics(target=out_target,
                                                                  prediction=out_prediction,
                                                                  class_names=predictions[o['name']].columns.tolist())
            sub_f = figures.get_multiclass_classification_figures(out_target,
                                                                  out_prediction,
                                                                  class_names=predictions[o['name']].columns.tolist())
        elif tfm_out_info.output_problem_type == MLProblemType.REGRESSION:
            # Get metrics and figures
            sub_m = metrics.get_regression_metrics(target=out_target, prediction=out_prediction)
            sub_f = figures.get_regression_figures(target=out_target, prediction=out_prediction)
        elif tfm_out_info.output_problem_type == MLProblemType.OBJECT_DETECTION:
            out_shapes = get_shapes_from_targets(targets=out_target, shapes_column=targets['shapes'].to_numpy())
            sub_m = metrics.get_object_detection_metrics(target=out_shapes, prediction=out_prediction)
            sub_f = {}
        elif tfm_out_info.output_problem_type == MLProblemType.OBJECT_SEGMENTATION:
            out_shapes = get_shapes_from_targets(targets=out_target, shapes_column=targets['shapes'].to_numpy())
            sub_m = metrics.get_object_segmentation_metrics(target=out_shapes, prediction=out_prediction)
            sub_f = {}
        else:
            raise Exception(f'Output problem type "{tfm_out_info.output_problem_type}" not recognized')

        # Add the prefix to all dictionaries
        sub_m = dict(list(map(lambda x: (f"{prefix}_{o['name']}_{x[0]}", x[1]), sub_m.items())))
        sub_f = dict(list(map(lambda x: (f"{prefix}_{o['name']}_{x[0]}", x[1]), sub_f.items())))
        # Copy the information of the current output dictionaries to global dictionaries
        metrics_dict = _update_dict(g_dict=metrics_dict, l_dict=sub_m)
        figures_dict = _update_dict(g_dict=figures_dict, l_dict=sub_f)
    return metrics_dict, figures_dict


def flat_dict(d: Dict, sep: str = '.', prefix: str = None) -> Dict:
    """
    Function that flatten a dict, transforming all values that are dictionaries to a key/value pair
    For example, getting a dict like this:
    {'key': {'in1_key': {'in2_key': value}}} and a '.' as prefix, it will transform this dict to this one:
    {'key.in1_key.in2_key': value}, that is, to a single dict
    Args:
        d (Dict): dict to be flattened
        sep (str): which separator use between dict keys
        prefix (str): prefix used by recursive call with the "parent" dict key

    Returns:
        Dict after input dictionary is "flattened"
    """
    # Create a empty dict
    new_dict = dict()
    # For each item of the input dict
    for k, v in d.items():
        # If the type of the value is adict
        if isinstance(v, dict):
            # Call recursively to flat with the key as prefix and iterate over its elements
            for k2, v2 in flat_dict(d=v, prefix=k if prefix is None else prefix + sep + k).items():
                # Store the elements in the new dict
                new_dict[k2] = v2
        elif isinstance(v, list):
            # If the value is a list, flat each element of the list with its index value
            for idx, el in enumerate(v):
                # Iterate over the flattened dict and store elements in the new dict
                if isinstance(el, dict):
                    for k2, v2 in flat_dict(d=el,
                                            prefix=f'{k}_{idx+1}' if prefix is None else prefix + sep +
                                            f'{k}_{idx+1}').items():
                        new_dict[k2] = v2
                else:
                    new_dict[f'{k}_{idx+1}'] = el
        else:
            # If the value is other type, get the new key name and store in the new dict
            new_k = prefix + sep + k if prefix is not None else k
            new_dict[new_k] = v
    return new_dict


def join_data_and_predictions_df(data: pd.DataFrame,
                                 predictions: Dict,
                                 outputs: List[Dict],
                                 output_transforms: ElementTransforms,
                                 inplace: bool = False) -> pd.DataFrame:
    """
    Function that given data and the made predictions on that data, it joins them in one pd.DataFrame
    Args:
        data (pd.DataFrame): data used to make predictions on
        predictions (pd.DataFrame): predictions made on the given data
        outputs (List[Dict]): outputs information for getting the ID of each output
        output_transforms (ElementTransforms): transformations made to the outputs, so we can get the problem type
        inplace (bool): if True, de given "data" pd.DataFrame will be modified including predictions
                        If False (default), a copy of the data will be made to not modify it
                        We should set this to True when dealing with large amount of data to avoid
                        RAM memory errors

    Returns:
        pd.DataFrame after joining the data with the predictions
        For classification, we will have one new column for each class as 'pred_output_class_x' indicating
        the probability that the examples belongs to that class
        For regression, we will have one new column as 'pred_output' with the predicted value
    """
    # Create a copy of the data to no modify it
    if not inplace:
        data = data.copy()
    # For each output
    for o in outputs:

        # Don't get predictions if it is a non-required output
        # If it is not required, and it is not on predictions, go to next
        if not o['required'] and o['name'] not in predictions:
            continue
        # If this is reach, the element is required, so assert that it is on predictions
        # Could be also that the element is not required, but it is on predictions
        if o['name'] not in predictions:
            raise DataError(f"Element '{o['name']}' not in predictions")

        # Get the transformation output info to get the problem type
        tfm_out_info = output_transforms.get_transform(o['name']).get_transform_output_info()
        # Ensure that the problem type is set
        if tfm_out_info.output_problem_type is None:
            raise DataError('The output problem type is None')

        # If it is a classification problem, the predictions must be a pd.DataFrame
        # with one column per class indicating the probability of each class
        if tfm_out_info.output_problem_type in [
                MLProblemType.BINARY_CLASSIFICATION, MLProblemType.MULTI_CLASS_CLASSIFICATION
        ]:
            # Get predictions pd.DataFrame
            predicted_df = predictions[o['name']]
            if not isinstance(predicted_df, pd.DataFrame):
                raise DataError('The prediction for classification problems must be a pandas DataFrame')
            # Same length as data DataFrame
            if data.shape[0] != predicted_df.shape[0]:
                raise DataError('Different number of instances in data and prediction')
            # Assert that we have at least two classes
            if predicted_df.shape[1] < 2:
                raise DataError('Expected at least two columns on prediction')
            # Get classes
            classes = list(predicted_df.columns)
            # Create one column in data per class indicating the predicted probability
            for i in range(predicted_df.shape[1]):
                # Column name
                k = f'pred_{o["name"]}_class_{classes[i]}'
                # Assert that this column does not exist
                if k in data.columns:
                    raise DataError(f"Cannot create column named '{k}' because it already exists")
                # Set data
                data[k] = predicted_df.iloc[:, i].to_numpy()
        elif tfm_out_info.output_problem_type == MLProblemType.REGRESSION:
            # If it is regression, the prediction will be a numpy array of one dim
            # Or if it has two dims, with one column
            prediction_array = predictions[o['name']]
            if not isinstance(prediction_array, np.ndarray):
                raise DataError('Expected numpy array on predictions')
            # Assert size
            if prediction_array.shape[0] != data.shape[0]:
                raise DataError('Different number of instances in data and prediction')
            # Check dimensions
            if prediction_array.ndim != 1:
                if (prediction_array.ndim != 2) or (prediction_array.shape[1] != 1):
                    raise DataError('The predicted array must have a single column')
                # Remove 2nd dimension
                prediction_array = prediction_array[:, 0]
            # Create column name
            k = f'pred_{o["name"]}'
            # Assert that this column does not exist
            if k in data.columns:
                raise DataError(f"Cannot create column named '{k}' because it already exists")
            # Set data
            data[k] = prediction_array
        elif tfm_out_info.output_problem_type in [MLProblemType.OBJECT_DETECTION, MLProblemType.OBJECT_SEGMENTATION]:
            # Get predictions list
            predicted_list = predictions[o['name']]
            if not isinstance(predicted_list, list):
                raise DataError('The prediction for ObjectDetection and ObjectSegmentation must be a list')
            # Same length as data DataFrame
            if data.shape[0] != len(predicted_list):
                raise DataError('Different number of instances in data and prediction')
            # Create column name
            k = f'pred_{o["name"]}'
            # Assert that this column does not exist
            if k in data.columns:
                raise DataError(f"Cannot create column named '{k}' because it already exists")
            # Set data
            data[k] = predicted_list
        else:
            raise Exception(f'Output problem type "{tfm_out_info.output_problem_type}" not recognized')

    # Return data with predictions
    return data


def get_metric_name_and_comparator(problem_type: MLProblemType) -> Tuple[str, Callable]:
    """
    Function that given the problem type, returns the name of the metric that should be used
    to get the best model. This function also returns an "operator" object that can be used
    for comparing two metrics, returning True if the first metrics is best than the second one
    Args:
        problem_type (MLProblemType): MLProblemType instance indicating the problem type

    Returns:
        Two values:
            - str with the name of the metric to be used
            - Callable that is an operator that can be used for comparing two metrics,
            returning True if the first one is better than the second one

    """
    if problem_type == MLProblemType.BINARY_CLASSIFICATION:
        metric = 'auc'
        cmp = gt
    elif problem_type == MLProblemType.MULTI_CLASS_CLASSIFICATION:
        metric = 'acc'
        cmp = gt
    elif problem_type == MLProblemType.REGRESSION:
        metric = 'mse'
        cmp = lt
    elif problem_type == MLProblemType.OBJECT_DETECTION:
        metric = 'mAP'
        cmp = gt
    elif problem_type == MLProblemType.OBJECT_SEGMENTATION:
        metric = 'mAP'
        cmp = gt
    else:
        raise Exception(f'Output problem type "{problem_type}"')

    return metric, cmp
