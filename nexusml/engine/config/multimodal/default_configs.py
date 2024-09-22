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
    Returns the default dataframe transforms for the sklearn pipeline

    Returns:
        list: List of dictionaries with the configuration of the transforms
    """
    return [{
        'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements',
        'args': {
            'select_shapes': False
        }
    }, {
        'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
        'args': None
    }]


def sklearn_default_output_transforms():
    """
    Returns the default output transforms for the sklearn pipeline

    Returns:
        dict: Dictionary with the configuration of the transforms
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


def multimodal_input_transforms(path: str):
    """
    Returns the default input transforms for the multimodal pipeline.

    Args:
        path (str): Path to the NLP transform.

    Returns:
        dict: Dictionary with the configuration of the transforms
    """
    image_transform = {'class': 'nexusml.engine.data.transforms.vision.torchvision.BasicImageTransform', 'args': None}
    nlp_transform = {'class': 'nexusml.engine.data.transforms.nlp.text.BasicNLPTransform', 'args': {'path': path}}
    float_transform = {'class': 'nexusml.engine.data.transforms.sklearn.StandardScalerTransform', 'args': None}

    return {
        'global': {
            'image_file': image_transform,
            'text': nlp_transform,
            'float': float_transform,
            'category': {
                'class': 'nexusml.engine.data.transforms.sklearn.LabelEncoderTransform',
                'args': None
            }
        },
        'specific': None
    }


def get_data_section_config(train_data: str, test_data: str):
    """
    Returns the data section configuration for the multimodal pipeline.

    Args:
        train_data (str): Path to the training data.
        test_data (str): Path to the test data.

    Returns:
        dict: Dictionary with the configuration of the data section.
    """
    return {'train_data': train_data, 'test_data': test_data}


def get_experiment_tracking_section_config(experiment_name: str, experiment_save_path: str, save_figures: bool,
                                           save_predictions: bool):
    """
    Returns the experiment tracking section configuration for the multimodal pipeline.

    Args:
        experiment_name (str): Name of the experiment.
        experiment_save_path (str): Path to save the experiment.
        save_figures (bool): Whether to save the figures.
        save_predictions (bool): Whether to save the predictions.

    Returns:
        dict: Dictionary with the configuration of the experiment tracking section.
    """
    return {
        'name': experiment_name,
        'mlflow_uri': experiment_save_path,
        'save_figures': save_figures,
        'save_predictions': save_predictions
    }


def pytorch_multimodal_model_config(setup_function: str,
                                    path: str,
                                    hidden_size: int = 512,
                                    batch_norm: bool = True,
                                    dropout_p1: float = 0.25,
                                    dropout_p2: float = 0.5):
    """
    Returns the configuration for the multimodal model.

    Args:
        setup_function (str): Name of the setup function.
        path (str): Path to the model.
        hidden_size (int): Size of the hidden layer.
        batch_norm (bool): Whether to use batch normalization.
        dropout_p1 (float): Dropout probability 1.
        dropout_p2 (float): Dropout probability 2.

    Returns:
        dict: Dictionary with the configuration of the model.
    """
    input_transforms = multimodal_input_transforms(path=path)
    output_transforms = sklearn_default_output_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms
        },
        'dataframe_transforms': default_dataframe_transforms(),
        'model': {
            'class': 'nexusml.engine.models.multimodal.magnum.MagnumModel',
            'args': {
                'setup_function': 'nexusml.engine.models.multimodal.magnum.' + setup_function,
                'setup_args': {
                    'emb_size': hidden_size,
                    'batch_norm': batch_norm,
                    'dropout_p1': dropout_p1,
                    'dropout_p2': dropout_p2
                },
                'pretrained_kwargs': {
                    'pretrained': True,
                    'norm_layer': None
                }
            }
        }
    }
    return config


def get_pytorch_multimodal_training_args(epochs: List = [30],
                                         lr: List = [0.00325],
                                         batch_size: List = [8],
                                         num_workers: List = [4]):
    """
    Returns the training arguments for the multimodal pipeline.

    Args:
        epochs (List): List of epochs.
        lr (List): List of learning rates.
        batch_size (List): List of batch sizes.
        num_workers (List): List of number of workers.

    Returns:
        dict: Dictionary with the configuration of the training arguments.
    """
    return {
        'loss_function': [{
            'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction',
            'args': {}
        }],
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'num_workers': num_workers
    }


def generate_multimodal_classification_model_configs(output_path: str,
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
    Generates the multimodal classification model configurations.

    Args:
        output_path (str): Path to save the configurations.
        schema_path (str): Path to the schema.
        train_data (str): Path to the training data.
        test_data (str): Path to the test data.
        experiment_name (str): Name of the experiment.
        experiment_save_path (str): Path to save the experiment.
        save_figures (bool): Whether to save the figures.
        save_predictions (bool): Whether to save the predictions.
        categories_path (str): Path to the categories.
        seed (int): Seed for reproducibility.

    Returns:
        None
    """
    # Check that the Pipeline Type is image classification/regression
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.MULTIMODAL:
        raise SchemaError("The schema does not follow the 'multimodal_classification_regression' PipelineType")
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)

    training_args = get_pytorch_multimodal_training_args()

    os.makedirs(output_path, exist_ok=True)

    embedding_linear_model_configs = {
        'setup_function': ['create_multimodal_magnum_model'],
        'hidden_size': [512],
        'dropout_p1': [0.25],
        'dropout_p2': [0.5],
        'path': ['roberta-large']
    }

    training_model_configs = [[(k, v) for v in vs] for k, vs in training_args.items()]
    training_model_configs = list(map(dict, itertools.product(*training_model_configs)))
    embedding_linear_model_configs = [[(k, v) for v in vs] for k, vs in embedding_linear_model_configs.items()]
    embedding_linear_model_configs = list(map(dict, itertools.product(*embedding_linear_model_configs)))

    count = 0
    for j, training_params in enumerate(training_model_configs):
        for i, model_params in enumerate(embedding_linear_model_configs):
            output_file = os.path.join(output_path, f'torch_emb_multimodal_{count + 1}.yaml')
            if os.path.exists(output_file):
                print(f'The output file "{output_file} exist. Not overwritten')
            else:
                save_config = pytorch_multimodal_model_config(**model_params)
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
