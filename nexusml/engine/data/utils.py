import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from nexusml.constants import ENGINE_LIMIT_SCORE_CLASSES
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.exceptions import DataError
from nexusml.enums import MLProblemType


def np_column_cat(x1: Optional[np.ndarray], x2: np.ndarray) -> np.ndarray:
    """
    Helper function used for concatenating two numpy arrays by column.
    The first array (x1) can be None, so just the second array (x2) is returned
    Args:
        x1 (np.ndarray): The first array to where concatenate column. Can be None
        x2 (np.ndarray): The second array, data to concatenate to the first one

    Returns:
        np.ndarray after adding x2 column to x1 array
    """
    # If x1 is not None
    if x1 is not None:
        # Check that have the same number of elements
        if x1.shape[0] != x2.shape[0]:
            raise DataError('Error trying to join two numpy arrays of different shape')
        # If the arrays have a single dimension, add the second one
        if x1.ndim == 1:
            x1 = x1[:, np.newaxis]
        if x2.ndim == 1:
            x2 = x2[:, np.newaxis]
        # Concatenate both arrays by column
        return np.concatenate((x1, x2), axis=1)
    else:
        # If x1 is None, return just x2
        return x2


def json_example_to_series(json_ex: Dict) -> pd.Series:
    """
    Function that transforms a given example in JSON format to a pandas Series object
    Check the API for more info about the JSON format.

    Args:
        json_ex (Dict): dict with the example data and metadata

    Returns:
        pd.Series object with all the content of the input dict
    """
    # To store data and column names
    data = []
    column_names = []
    # For each element of the dict
    for k, v in json_ex.items():
        if k in ['inputs', 'outputs', 'values']:
            # Create an entry for each input with "element" as key and "value" as value
            for i in v:
                data.append(i['value'])
                column_names.append(i['element'])
        elif k == 'metadata':
            # Create an entry for each metadata with "element" as key and "value" as value
            # Add 'metadata.' prefix to key to properly identify metadata columns
            for i in v:
                data.append(i['value'])
                column_names.append(f"metadata.{i['element']}")
        elif k == 'shapes':
            # ToDo: make a proper transformation of shapes
            # For now encode the shapes as JSON string
            data.append(json.dumps(v))
            column_names.append(k)
        elif k == 'tags':
            data.append(str(v))
            column_names.append(k)
        else:
            # Append key and value
            data.append(v)
            column_names.append(k)

    # Create a series and return
    return pd.Series(data=data, index=column_names)


def json_examples_to_data_frame(json_ex_list: List[Dict]) -> pd.DataFrame:
    """
    Function that transforms a list of examples that follow the JSON format to a DataFrame object
    Check the API for more info about the JSON format.

    Args:
        json_ex_list (List[Dict]): list with all examples in JSON format

    Returns:
        DataFrame with all converted examples
    """
    # Parse each example
    list_series = list(map(json_example_to_series, json_ex_list))
    # Create DataFrame and return
    return pd.DataFrame(list_series)


def json_file_to_data_frame(json_file: str) -> pd.DataFrame:
    """
    Function that reads the JSON file and transform the list of read examples that follow the JSON format
    to a DataFrame object.

    Check the API for more info about the JSON format.

    Args:
        json_file (str): path to JSON file to be read with all examples

    Returns:
        DataFrame with all converted examples
    """
    with open(json_file, 'r') as f:
        json_examples_list = json.load(f)
    return json_examples_to_data_frame(json_examples_list)


def json_file_to_csv(json_file: str, out_csv_file: str):
    """
    Function that transforms a JSON file with example data to a CSV file.
    Check the API for more info about the JSON format.

    Args:
        json_file (str): path to JSON file to be read with all examples
        out_csv_file (str): path where store the CSV file with the transformed data

    Returns:

    """
    df = json_file_to_data_frame(json_file)
    df.to_csv(out_csv_file, index=False)


def get_shapes_from_targets(targets: np.ndarray, shapes_column: np.ndarray) -> List[List[Any]]:
    """
    Function that gets the shapes of the examples given the targets and the shapes column

    Args:
        targets (np.ndarray): array with the targets of the examples
        shapes_column (np.ndarray): array with the shapes of the examples

    Returns:
        list: List of lists with the shapes of each example
    """
    all_shapes = []
    for i, target in enumerate(targets):
        example_shapes = []
        shapes_json = json.loads(shapes_column[i])

        if isinstance(target, list):
            # If it is a list, iterate through the shapes ids of the example
            for shape_id in target:
                for shape in shapes_json:
                    if shape['id'] == shape_id:
                        example_shapes.append(shape)
                        break
        else:
            # If it is just a string, the example has only one shape
            for shape in shapes_json:
                if shape['id'] == target:
                    example_shapes.append(shape)
                    break

        all_shapes.append(example_shapes)

    return all_shapes


def predictions_to_example_format(predictions: dict,
                                  output_transforms: ElementTransforms,
                                  limit_classes: int = ENGINE_LIMIT_SCORE_CLASSES) -> list:
    """
    Function that transform the predictions done by a model to NexusML example format

    Args:
        predictions (dict): dict returned by the model having the prediction for each output element
        output_transforms (ElementTransforms): all the output transforms, so we can get the MLProblemType of the output
        limit_classes (int): in some cases we can have a lot of classes, and it can cause a problem when returning
                        the results as a response of an API request. Furthermore, the most valuable prediction
                        scores are for few classes with high confidence. So this argument controls how many
                        classes' score return (those with higher confidence)

    Returns:
        The predictions of each example but in NexusML format
    """
    # Check that all predicted arrays have the same number of elements
    n = None
    for k, v in predictions.items():
        if n is None:
            n = len(v)
        if n != len(v):
            raise DataError('Different number of examples for predicted elements')
    # Create a list to store predictions
    prediction_list = []
    # For each element
    for i in range(n):
        # For each output
        outputs = []
        for k, v in predictions.items():
            problem_type = output_transforms.element_transform_map[k].get_transform_output_info().output_problem_type
            if problem_type == MLProblemType.REGRESSION:
                outputs.append({'element': k, 'value': v[i].item()})
            elif problem_type in [MLProblemType.BINARY_CLASSIFICATION, MLProblemType.MULTI_CLASS_CLASSIFICATION]:
                assert isinstance(v, pd.DataFrame)
                series = v.iloc[i, :].nlargest(limit_classes)
                scores = dict(zip(series.index, series.to_list()))
                predicted_class = series.index[series.argmax()]
                outputs.append({'element': k, 'value': {'category': predicted_class, 'scores': scores}})
            elif problem_type in [MLProblemType.OBJECT_DETECTION, MLProblemType.OBJECT_SEGMENTATION]:
                outputs.append({'element': k, 'value': v[i]})
            else:
                raise Exception(f"Unexpected problem type '{str(problem_type)}'")

        prediction_list.append({'outputs': outputs})

    return prediction_list
