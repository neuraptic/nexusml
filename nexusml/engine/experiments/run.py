import importlib
import os
import random
import shutil
import tempfile
import time
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.transforms.base import get_dataframe_transforms
from nexusml.engine.data.transforms.base import get_element_transforms
from nexusml.engine.data.utils import json_file_to_data_frame
from nexusml.engine.exceptions import ConfigFileError
from nexusml.engine.exceptions import DataError
from nexusml.engine.experiments.tracking.base import ExperimentLogger
from nexusml.engine.experiments.tracking.mlflow import get_exp_model
from nexusml.engine.experiments.tracking.mlflow import MlFlowExperimentLogger
from nexusml.engine.experiments.utils import get_metrics_and_figures
from nexusml.engine.experiments.utils import join_data_and_predictions_df
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.engine.utils import get_random_string


def _get_model_creator(uri: str) -> Callable:
    """
    Function that returns a callable function given its path
    Args:
        uri (str): whole path to the function or class to be imported as callable

    Returns:
        Callable object of the given funciton or class path
    """
    # Split by '.'
    module_and_class = uri.split('.')
    # All but last part is the module
    module = '.'.join(module_and_class[:-1])
    # The las part is the function or class constructor
    class_name = module_and_class[-1]
    # Import module
    module = importlib.import_module(module)
    # Get callable from module
    return getattr(module, class_name)


def _set_random(seed: int):
    """
    Function for setting the random seed for numpy, random and torch
    Args:
        seed: seed to be applied

    Returns:

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(config: Dict):
    """
    Function that runs one experiment getting model configuration and data paths from the given configuration
    Args:
        config (Dict): dict with the configuration to be used for the experiment (for example, read it from file)

    Returns:

    """
    # Start time
    t_start = time.time()

    # Get seed or set 0 if not exist
    if 'seed' in config:
        seed = config['seed']
    else:
        seed = 0
    _set_random(seed)

    # Read schema
    schema = Schema.create_schema_from_json(json_file=config['schema'])

    # Initialize element categories as None
    element_categories = None
    # Get categorical elements
    categorical_type_elements = schema.categorical_inputs() + schema.categorical_outputs()
    if len(categorical_type_elements) > 0:
        if 'categories' not in config:
            raise ConfigFileError("Missing 'categories' key on config file")
        # Parse the element categories (get the possible values for each category element)
        element_categories = Categories.create_category_from_json(json_file=config['categories'])
        # Assert that all categorical elements have entry on categories
        if not element_categories.has_entry(list(map(lambda x: x['name'], categorical_type_elements))):
            raise ConfigFileError('Missing the category list for some category type elements')

    # Create the experiment tracker. In this case using MlFlow
    tracker = MlFlowExperimentLogger(mlflow_uri=config['experiment']['mlflow_uri'],
                                     experiment_name=config['experiment']['name'])
    # Sometimes figures and predictions could be big (talking about size). Get by arguments if we store them or not
    # By default store them
    save_figures = True
    save_predictions = True
    # If the parameter is given
    if 'save_figures' in config['experiment']:
        # Get it and assert that is a boolean
        save_figures = config['experiment']['save_figures']
        if not isinstance(save_figures, bool):
            raise ConfigFileError("Unexpected value type for 'save_figures' parameter")
    if 'save_predictions' in config['experiment']:
        # Get it and assert that is a boolean
        save_predictions = config['experiment']['save_predictions']
        if not isinstance(save_predictions, bool):
            raise ConfigFileError("Unexpected value type for 'save_predictions' parameter")
    # By default, make the model evaluation
    eval_model = True
    # Update parameter if it is given
    if 'evaluate_model' in config['experiment']:
        # Get it and assert that is a boolean
        eval_model = config['experiment']['evaluate_model']
        if not isinstance(eval_model, bool):
            raise ConfigFileError("Unexpected value type for 'evaluate_model' parameter")

    # If there are global (dataframe) transformations, get them
    # Otherwise, initialize with empty list
    if 'dataframe_transforms' in config and config['dataframe_transforms'] is not None:
        dataframe_transforms = get_dataframe_transforms(schema=schema,
                                                        categories=element_categories,
                                                        transforms_config=config['dataframe_transforms'])
    else:
        # Empty transform list
        dataframe_transforms = DataFrameTransforms(transforms=[])

    # Get input and output transformations
    input_transforms, output_transforms = get_element_transforms(schema=schema,
                                                                 transforms_config=config['transforms'],
                                                                 categories=element_categories)

    # Get train and test file paths
    train_data_path = config['data']['train_data']
    test_data_path = config['data']['test_data'] if 'test_data' in config['data'] else None
    # They must be either JSON files or CSV files
    if train_data_path.endswith('.json'):
        # Parse JSON file to DataFrame
        if test_data_path is not None and not test_data_path.endswith('.json'):
            raise ConfigFileError('Different file type for train and test files')
        train_df = json_file_to_data_frame(json_file=train_data_path)
        test_df = json_file_to_data_frame(json_file=test_data_path) if test_data_path is not None else None
    elif train_data_path.endswith('.csv'):
        # Load CSV file as DataFrame
        if test_data_path is not None and not test_data_path.endswith('.csv'):
            raise ConfigFileError('Different file type for train and test files')
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path) if test_data_path is not None else None
    else:
        raise ConfigFileError(f'File {train_data_path} is neither JSON not CSV file')

    # Check and get the data frames columns and schema required elements
    elements = schema.required_inputs() + schema.required_outputs()
    # The elements that are type shape will be under 'shapes' column. So filter them asserting that 'shapes' is on DFs
    shape_type_elements = list(filter(lambda x: x['type'] == 'shape', elements))
    if len(shape_type_elements) > 0:
        if 'shapes' not in train_df.columns:
            raise DataError('Missing shapes on training data')
        if test_df is not None and 'shapes' not in test_df.columns:
            raise DataError('Missing shapes on testing data')
    element_ids = list(map(lambda x: x['name'], elements))
    if not np.isin(element_ids, train_df.columns).all():
        raise DataError('Missing some required elements on training data')
    if test_df is not None and not np.isin(element_ids, test_df.columns).all():
        raise DataError('Missing some required elements on testing data')

    # Log config as parameter
    tracker.log_params(config, flat_params=True)

    # Get model creator class or function
    model_creator_function = _get_model_creator(uri=config['model']['class'])
    # Create the model
    model = model_creator_function(schema=schema,
                                   categories=element_categories,
                                   dataframe_transforms=dataframe_transforms,
                                   input_transforms=input_transforms,
                                   output_transforms=output_transforms,
                                   model_config=config['model']['args'])
    # Get training args if any
    train_args = config['training'] if 'training' in config else None
    # Train the model with train data
    train_log = model.fit(train_data=train_df, train_args=train_args)
    # Store training output figures if any and if store flas is on
    if train_log is not None and train_log.figures is not None and save_figures:
        tracker.log_figures(figures_dict=train_log.figures)

    # Store training output parameters if any
    if train_log is not None and train_log.params is not None:
        tracker.log_params(param_dict=train_log.params)

    if eval_model:
        # Get train
        train_predictions = model.predict(data=train_df, split_predictions_by_output=True, train_args=train_args)
        # Evaluate train
        evaluate_predictions(exp_logger=tracker,
                             data=train_df,
                             predictions=train_predictions,
                             schema=schema,
                             output_transforms=output_transforms,
                             prefix='train',
                             save_figures=save_figures,
                             save_predictions=save_predictions)

    # Test predictions if given
    if test_df is not None and eval_model:
        test_predictions = model.predict(data=test_df, split_predictions_by_output=True, train_args=train_args)
        # Evaluate test
        evaluate_predictions(exp_logger=tracker,
                             data=test_df,
                             predictions=test_predictions,
                             schema=schema,
                             output_transforms=output_transforms,
                             prefix='test',
                             save_figures=save_figures,
                             save_predictions=save_predictions)

    # Save the model and log it
    # The same with config, but renaming the original config file to 'config.yaml'
    with tempfile.TemporaryDirectory() as d:
        model.save(os.path.join(d, 'model.pkl'))
        tracker.log_artifact(os.path.join(d, 'model.pkl'))
        with open(os.path.join(d, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        tracker.log_artifact(os.path.join(d, 'config.yaml'))
        # Track schema and categories
        shutil.copyfile(config['schema'], os.path.join(d, 'schema.json'))
        tracker.log_artifact(os.path.join(d, 'schema.json'))
        if element_categories is not None:
            shutil.copyfile(config['categories'], os.path.join(d, 'categories.json'))
            tracker.log_artifact(os.path.join(d, 'categories.json'))
        # Get model summary and save it
        model_summary = model.summary()
        # If there is no summary, initialize as empty line
        if model_summary is None:
            model_summary = '\n'
        with open(os.path.join(d, 'summary.txt'), 'w') as f:
            f.write(model_summary)
        tracker.log_artifact(os.path.join(d, 'summary.txt'))

    # Store for each output element the problem type as parameter
    for o in schema.outputs:
        # Get the transformation output info to get the problem type
        tfm_out_info = output_transforms.get_transform(o['name']).get_transform_output_info()
        # Ensure that the problem type is set
        if tfm_out_info.output_problem_type is None:
            raise DataError(f"The transform output information for {o['name']} element is None")
        tracker.log_param(f'output_{o["name"]}_problem_type', tfm_out_info.output_problem_type)

    # End time
    t_end = time.time()
    execution_time = t_end - t_start

    # Store time as metric
    tracker.log_metric('execution_time', execution_time)

    # End the experiment
    tracker.end_run()


def evaluate_predictions(exp_logger: ExperimentLogger, data: Union[pd.DataFrame,
                                                                   dict], predictions: dict, schema: Schema,
                         output_transforms: ElementTransforms, prefix: str, save_figures: bool, save_predictions: bool):
    """
    Function that evaluates the predictions and stores the results.

    Args:
        exp_logger (ExperimentLogger): experiment logger to store the results
        data (Union[pd.DataFrame, dict]): data used for the predictions
        predictions (dict): predictions made by the model
        schema (Schema): schema with the task information
        output_transforms (ElementTransforms): output transformations
        prefix (str): prefix to be used for the results
        save_figures (bool): flag to store figures
        save_predictions (bool): flag to store predictions
    """
    # Get metrics and figures. And log them with tracker
    m, f = get_metrics_and_figures(targets=data,
                                   predictions=predictions,
                                   outputs=schema.outputs,
                                   output_transforms=output_transforms,
                                   prefix=prefix)
    exp_logger.log_metrics(m)
    # Only store figures if the flag is on
    if save_figures:
        exp_logger.log_figures(f)

    # Join data and predictions if the flag is on
    data_and_predictions = None
    if save_predictions:
        data_and_predictions = join_data_and_predictions_df(data=data,
                                                            predictions=predictions,
                                                            outputs=schema.outputs,
                                                            output_transforms=output_transforms,
                                                            inplace=True)

    # Save data and predictions
    with tempfile.TemporaryDirectory() as d:
        if data_and_predictions is not None:
            data_and_predictions.to_csv(os.path.join(d, f'{prefix}_predictions.csv'), index=False)
            exp_logger.log_artifact(os.path.join(d, f'{prefix}_predictions.csv'))


def run_experiment_from_config_file(config_file_path: str, callback: Optional[Callable] = None):
    """
    Function that runs one experiment getting model configuration and data paths from the given configuration file
    Args:
        config_file_path (str): path to configuration file to be used for the experiment
        callback (Callable): callable function that will be called (if it is not None) after the experiment finish

    Returns:

    """
    # Load config file
    print(f'\t[+] Running config file {config_file_path}')
    with open(config_file_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)

    run_experiment(config)

    if callback is not None:
        callback()


def retrain_model(base_config_file: str, schema: str, categories: str, train_data: str, output_path: str):
    """
    Retrains a model.

    Args:
        base_config_file (str): path to YAML config file used to train the model
        schema (Schema): JSON file path to task schema
        categories (Categories): JSON file path tp possible values for category elements
        train_data (str): path to train data file
        output_path (str): output path where store the new trained model and config file

    Returns:
        Model with the new trained model
    """
    # Load config file
    with open(base_config_file, 'r') as f:
        config = yaml.load(f, Loader=Loader)

    # Tempdir for storing the experiment
    temp_dir = tempfile.TemporaryDirectory()

    # Change experiment, data, schema and categories
    config['categories'] = categories
    config['data'] = {'train_data': train_data}
    mlflow_uri = f'file:///{temp_dir.name}'
    exp_name = get_random_string(15)
    config['experiment'] = {
        'mlflow_uri': mlflow_uri,
        'name': exp_name,
        'save_figures': False,
        'save_predictions': False,
        'evaluate_model': False
    }
    config['schema'] = schema

    # Train the model
    run_experiment(config)

    # Get the model and copy it to output path
    model_path = get_exp_model(mlflow_uri=mlflow_uri, experiment_name=exp_name)
    if model_path.startswith('file:///'):
        model_path = model_path.replace('file:///', '')

    # Copy model to output path
    os.makedirs(output_path, exist_ok=True)
    shutil.copyfile(model_path, os.path.join(output_path, 'model.pkl'))

    # Save also new config file on output path
    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Clearn temp dir
    temp_dir.cleanup()
