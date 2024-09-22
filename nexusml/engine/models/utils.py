from typing import List

import pandas as pd

from nexusml.engine.data.utils import json_file_to_data_frame
from nexusml.engine.models.base import Model


def smooth(scalars: List[float], weight: float) -> List[float]:
    """
    Function used for making the smoothness of a curve (to remove high frequency peaks)
    Args:
        scalars (List[float]): list of values to be smoothed
        weight (float): smooth weight between 0 and 1. Smaller values, less smooth effect

    Returns:
        A list with the values after smooth process
    """
    # Get the first value
    last = scalars[0]
    # Empty list to store the smoothed values
    smoothed = list()
    # For each value
    for point in scalars:
        # Calculate the smoothed value using the given weight and the current value and the previous one
        smoothed_val = last * weight + (1 - weight) * point
        # Add to list
        smoothed.append(smoothed_val)
        # Update last
        last = smoothed_val

    return smoothed


def compute_and_save_monitoring_templates(data_path: str, model_path: str, output_path: str):
    """
    Loads data from a specified file path, either in JSON or CSV format, and applies a model to compute templates.
    The templates are then saved to the provided output path. The function first checks the file type of the input
    data based on its extension and loads it accordingly. Then, it loads the model and uses it to compute the monitoring
    templates with the given data, which are saved to the specified output location.

    Args:
        data_path (str): Path to the input data file, expected to be either JSON or CSV format.
        model_path (str): Path to the model file to load.
        output_path (str): Path where the computed templates will be saved.

    Returns:
        None

    Raises:
        ValueError: If the input data file is neither a JSON nor CSV file.
    """
    if data_path.endswith('.json'):
        df = json_file_to_data_frame(json_file=data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f'File {data_path} is neither JSON not CSV file')

    m = Model.load(input_file=model_path)
    m.compute_templates(data=df, output_file_path=output_path)
