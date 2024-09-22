import itertools
import os
from typing import List

import yaml

from nexusml.engine.exceptions import SchemaError
from nexusml.engine.schema.base import get_pipeline_type
from nexusml.engine.schema.base import Schema
from nexusml.enums import PipelineType


def default_dataframe_transforms():
    """
    Returns the default dataframe transforms for sklearn models

    Returns:
        list: List of dictionaries with the default dataframe transforms
    """
    return [
        {
            'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements',
            'args': {
                'select_shapes': False
            }
        },
        {
            'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
            'args': None
        },
    ]


def sklearn_default_output_transforms():
    """
    Returns the default output transforms for sklearn models

    Returns:
        dict: Dictionary with the default output transforms
    """
    return {
        'global': {
            'float': {
                'class': 'nexusml.engine.data.transforms.sklearn.MinMaxScalerTransform',
                'args': None
            },
            'category': {
                'class': 'nexusml.engine.data.transforms.sklearn.LabelEncoderTransform',
                'args': None
            }
        },
        'specific': None
    }


def nlp_input_transforms(path: str):
    """
    Returns the default input transforms for NLP models

    Args:
        path (str): Path to the transform

    Returns:
        dict: Dictionary with the default input transforms
    """
    nlp_transform = {'class': 'nexusml.engine.data.transforms.nlp.text.BasicNLPTransform', 'args': {'path': path}}

    return {'global': {'text': nlp_transform}, 'specific': None}


def get_data_section_config(train_data: str, test_data: str):
    """
    Returns the data section configuration

    Args:
        train_data (str): Path to the training data
        test_data (str): Path to the testing data

    Returns:
        dict: Dictionary with the data section configuration
    """
    return {'train_data': train_data, 'test_data': test_data}


def get_experiment_tracking_section_config(experiment_name: str, experiment_save_path: str, save_figures: bool,
                                           save_predictions: bool):
    """
    Returns the experiment tracking section configuration.

    Args:
        experiment_name (str): Name of the experiment
        experiment_save_path (str): Path to save the experiment
        save_figures (bool): Whether to save the figures
        save_predictions (bool): Whether to save the predictions

    Returns:
        dict: Dictionary with the experiment tracking section configuration
    """
    return {
        'name': experiment_name,
        'mlflow_uri': experiment_save_path,
        'save_figures': save_figures,
        'save_predictions': save_predictions
    }


def pytorch_nlp_classification_model_config(setup_function: str, path: str, dropout_p1: float = 0.25):
    """
    Returns the configuration for a PyTorch NLP classification model.

    Args:
        setup_function (str): Setup function for the model
        path (str): Path to the transform
        dropout_p1 (float): Dropout probability

    Returns:
        dict: Dictionary with the configuration
    """
    input_transforms = nlp_input_transforms(path=path)
    output_transforms = sklearn_default_output_transforms()
    dataframe_transforms = default_dataframe_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms
        },
        'dataframe_transforms': dataframe_transforms,
        'model': {
            'class': 'nexusml.engine.models.nlp.roberta.TransformersNLPModel',
            'args': {
                'setup_function': 'nexusml.engine.models.nlp.roberta.' + setup_function,
                'setup_args': {
                    'dropout_p1': dropout_p1
                },
                'pretrained_kwargs': {
                    'pretrained_model_name_or_path': path
                }
            }
        }
    }
    return config


def get_pytorch_nlp_training_args(epochs: List = [2, 5],
                                  lr: List = [0.005],
                                  batch_size: List = [4],
                                  num_workers: List = [0]):
    """
    Returns the training arguments for a PyTorch NLP model.

    Args:
        epochs (List): List of epochs
        lr (List): List of learning rates
        batch_size (List): List of batch sizes
        num_workers (List): List of number of workers

    Returns:
        dict: Dictionary with the training arguments
    """
    return {
        'loss_function': [{
            'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction',
            'args': {
                'classification_cost_sensitive': True
            }
        }],
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'num_workers': num_workers
    }


def generate_pytorch_nlp_classification_model_configs(output_path: str,
                                                      schema_path: str,
                                                      train_data: str,
                                                      test_data: str,
                                                      experiment_name: str,
                                                      experiment_save_path: str,
                                                      save_figures: bool,
                                                      save_predictions: bool,
                                                      categories_path: str = None,
                                                      seed: int = 0):
    """
    Generates the PyTorch NLP classification model configurations.

    Args:
        output_path (str): Path to save the configurations
        schema_path (str): Path to the schema
        train_data (str): Path to the training data
        test_data (str): Path to the testing data
        experiment_name (str): Name of the experiment
        experiment_save_path (str): Path to save the experiment
        save_figures (bool): Whether to save the figures
        save_predictions (bool): Whether to save the predictions
        categories_path (str): Path to the categories
        seed (int): Seed

    Returns:
        None
    """
    # Check that the Pipeline Type is text classification/regression
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.NLP_CLASSIFICATION_REGRESSION:
        raise SchemaError("The schema does not follow the 'nlp_classification_regression' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)

    training_args = get_pytorch_nlp_training_args(epochs=[10], batch_size=[32])

    os.makedirs(output_path, exist_ok=True)

    nlp_classification_model_configs = {
        'setup_function': ['create_transformers_classifier_model'],
        'path': ['xlm-roberta-base', 'microsoft/Multilingual-MiniLM-L12-H384'],
        'dropout_p1': [0.25]
    }

    training_model_configs = [[(k, v) for v in vs] for k, vs in training_args.items()]
    training_model_configs = list(map(dict, itertools.product(*training_model_configs)))
    nlp_classification_model_configs = [[(k, v) for v in vs] for k, vs in nlp_classification_model_configs.items()]
    nlp_classification_model_configs = list(map(dict, itertools.product(*nlp_classification_model_configs)))

    count = 0
    for j, training_params in enumerate(training_model_configs):
        for i, model_params in enumerate(nlp_classification_model_configs):
            output_file = os.path.join(output_path, f'torch_nlp_classification_{count + 1}.yaml')
            if os.path.exists(output_file):
                print(f'The output file "{output_file} exist. Not overwritten')
            else:
                save_config = pytorch_nlp_classification_model_config(**model_params)
                save_config['data'] = data_config
                save_config['experiment'] = experiment_config
                save_config['schema'] = schema_path
                save_config['seed'] = seed
                save_config['training'] = training_params
                if categories_path is not None:
                    save_config['categories'] = categories_path
                with open(output_file, 'w') as f:
                    yaml.dump(save_config, f)
                    count += 1
