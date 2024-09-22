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
    Returns a list of default data transformation configurations.

    This function generates a list of dictionaries that define the default
    transformations to be applied to a DataFrame. Each dictionary specifies
    the class of the transformation and any relevant arguments required
    for the transformation.

    The first transformation selects required elements without selecting shapes,
    while the second transformation drops rows with NaN values from the DataFrame.

    Steps:
    1. Defines the selection of required elements without shape selection.
    2. Defines the drop of NaN values transformation.

    Returns:
        list: A list of dictionaries representing the default transformations.
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
    Returns the default output transforms for sklearn-based data processing.

    This function provides a dictionary that defines the global transformations
    for different data types (`float` and `category`). It maps these data types
    to specific sklearn-based transformation classes. For `float`, it uses
    `MinMaxScalerTransform`, and for `category`, it uses `LabelEncoderTransform`.
    The `specific` key is set to None, implying that no specific transformations
    are defined.

    Returns:
        dict: A dictionary containing global transformations and a placeholder for specific transformations.
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


def speech_classification_default_input_transforms(path: str, sample_rate: int):
    """
    Generates a dictionary of default input transforms for speech classification tasks.

    This function returns a configuration dictionary for handling audio input files
    in a speech classification task. It includes a global transform that applies
    a default speech transform with the specified audio file path and target sample rate.
    The transform class is specified as 'DefaultSpeechTransform', and it takes two
    arguments: the audio file path and the target sample rate. The 'specific' key is
    currently set to None, indicating no specific transforms at this point.

    Steps:
    1. Creates a dictionary structure for audio transforms.
    2. Populates the 'global' section with audio file transformation details, using the
       provided path and sample rate.
    3. Leaves the 'specific' section as None for potential future customization.

    Args:
        path (str): The file path of the audio file to be processed.
        sample_rate (int): The target sample rate to which the audio file should be resampled.

    Returns:
        dict: A dictionary containing the configuration for global and specific audio transforms.
    """
    return {
        'global': {
            'audio_file': {
                'class': 'nexusml.engine.data.transforms.audio.speech.DefaultSpeechTransform',
                'args': {
                    'path': path,
                    'target_sr': sample_rate
                }
            }
        },
        'specific': None
    }


def get_data_section_config(train_data: str, test_data: str):
    """
    Returns a dictionary containing the data section configuration.

    Args:
        train_data (str): The path to the training data.
        test_data (str): The path to the testing data.

    Returns:
        dict: A dictionary containing the data section configuration.
    """
    return {'train_data': train_data, 'test_data': test_data}


def get_experiment_tracking_section_config(experiment_name: str, experiment_save_path: str, save_figures: bool,
                                           save_predictions: bool):
    """
    Returns a dictionary containing the experiment tracking section configuration.

    Args:
        experiment_name (str): The name of the experiment.
        experiment_save_path (str): The path to save the experiment.
        save_figures (bool): A flag indicating whether to save figures.
        save_predictions (bool): A flag indicating whether to save predictions.

    Returns:
        dict: A dictionary containing the experiment tracking section configuration.
    """
    return {
        'name': experiment_name,
        'mlflow_uri': experiment_save_path,
        'save_figures': save_figures,
        'save_predictions': save_predictions
    }


def get_pytorch_audio_training_args(epochs: List = None,
                                    lr: List = None,
                                    batch_size: List = None,
                                    num_workers: List = None):
    """
    Returns a dictionary containing the training arguments for PyTorch audio models.

    Args:
        epochs (List): A list of epochs to train the model.
        lr (List): A list of learning rates to use during training.
        batch_size (List): A list of batch sizes to use during training.
        num_workers (List): A list of the number of workers to use during training.

    Returns:
        dict: A dictionary containing the training arguments for PyTorch audio models.
    """
    if epochs is None:
        epochs = [10]
    if lr is None:
        lr = [0.005]
    if batch_size is None:
        batch_size = [4]
    if num_workers is None:
        num_workers = [0]

    return {
        'loss_function': [{
            'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction',
            'args': None
        }],
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'num_workers': num_workers
    }


def pytorch_audio_classification_model_config(setup_function: str, path: str, target_sr: int, dropout_p1: float = 0.25):
    """
    Generates a PyTorch audio classification model configuration.

    Args:
        setup_function (str): The setup function to use for the model.
        path (str): The path to the pre-trained model.
        target_sr (int): The target sample rate for the audio files.
        dropout_p1 (float): The dropout probability for the first dropout layer.

    Returns:
        dict: A dictionary containing the PyTorch audio classification model configuration.
    """
    input_transforms = speech_classification_default_input_transforms(path=path, sample_rate=target_sr)
    output_transforms = sklearn_default_output_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms
        },
        'dataframe_transforms': default_dataframe_transforms(),
        'model': {
            'class': 'nexusml.engine.models.audio.wav2vec2.CustomWav2Vec2ModelForClassification',
            'args': {
                'setup_function': 'nexusml.engine.models.audio.wav2vec2.' + setup_function,
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


def generate_pytorch_audio_classification_model_configs(output_path: str,
                                                        schema_path: str,
                                                        train_data: str,
                                                        test_data: str,
                                                        experiment_name: str,
                                                        experiment_save_path: str,
                                                        save_figures: bool,
                                                        save_predictions: bool,
                                                        categories_path: str = None,
                                                        seed: int = 42):
    """
    Generates PyTorch audio classification model configurations.

    Args:
        output_path (str): The path to save the generated configurations.
        schema_path (str): The path to the schema file.
        train_data (str): The path to the training data.
        test_data (str): The path to the testing data.
        experiment_name (str): The name of the experiment.
        experiment_save_path (str): The path to save the experiment.
        save_figures (bool): A flag indicating whether to save figures.
        save_predictions (bool): A flag indicating whether to save predictions.
        categories_path (str): The path to the categories file.
        seed (int): The seed value for reproducibility.

    Returns:
        None
    """
    # Check that the Pipeline Type is audio classification/regression
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.AUDIO_CLASSIFICATION_REGRESSION:
        raise SchemaError("The schema does not follow the 'audio_classification_regression' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)

    training_args = get_pytorch_audio_training_args(epochs=[10], batch_size=[8])

    os.makedirs(output_path, exist_ok=True)

    audio_classification_model_configs = {
        'setup_function': ['create_speech_classification_model'],
        'path': ['jonatasgrosman/wav2vec2-large-xlsr-53-english', 'facebook/wav2vec2-xls-r-300m'],
        'target_sr': [16000],
        'dropout_p1': [0.25]
    }

    training_model_configs = [[(k, v) for v in vs] for k, vs in training_args.items()]
    training_model_configs = list(map(dict, itertools.product(*training_model_configs)))
    audio_classification_model_configs = [[(k, v) for v in vs] for k, vs in audio_classification_model_configs.items()]
    audio_classification_model_configs = list(map(dict, itertools.product(*audio_classification_model_configs)))

    count = 0
    for i, training_params in enumerate(training_model_configs):
        for j, model_params in enumerate(audio_classification_model_configs):
            output_file = os.path.join(output_path, f'torch_audio_classification_{count + 1}.yaml')
            if os.path.exists(output_file):
                print(f'{output_file} already exists, skipping')
                continue

            save_config = pytorch_audio_classification_model_config(**model_params)
            save_config['data'] = data_config
            save_config['experiment'] = experiment_config
            save_config['training'] = training_params
            save_config['seed'] = seed
            save_config['schema'] = schema_path
            if categories_path is not None:
                save_config['categories'] = categories_path

            with open(output_file, 'w') as f:
                yaml.dump(save_config, f)
                count += 1
