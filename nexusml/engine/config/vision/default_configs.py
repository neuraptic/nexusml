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
    Returns the default dataframe transforms for sklearn models.

    Returns:
        list: List of dictionaries with the default dataframe transforms.
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


def detection_default_dataframe_transforms():
    """
    Returns the default dataframe transforms for object detection models.

    Returns:
        list: List of dictionaries with the default dataframe transforms.
    """
    return [{
        'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements',
        'args': {
            'exclusion': None,
            'select_shapes': True
        }
    }, {
        'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
        'args': None
    }, {
        'class': 'nexusml.engine.data.transforms.vision.detectron.RegisterDetectionDatasetTransform',
        'args': None
    }]


def segmentation_default_dataframe_transforms():
    """
    Returns the default dataframe transforms for object segmentation models.

    Returns:
        list: List of dictionaries with the default dataframe transforms.
    """
    return [{
        'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements',
        'args': {
            'exclusion': None,
            'select_shapes': True
        }
    }, {
        'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
        'args': None
    }, {
        'class': 'nexusml.engine.data.transforms.vision.detectron.RegisterSegmentationDatasetTransform',
        'args': None
    }]


def sklearn_default_output_transforms():
    """
    Returns the default output transforms for sklearn models.

    Returns:
        dict: Dictionary with the default output transforms.
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


def vision_input_transforms():
    """
    Returns the default input transforms for vision models.

    Returns:
        dict: Dictionary with the default input transforms.
    """
    image_transform = {'class': 'nexusml.engine.data.transforms.vision.torchvision.BasicImageTransform', 'args': None}

    return {'global': {'image_file': image_transform}, 'specific': None}


def detectron_default_input_transforms():
    """
    Returns the default input transforms for Detectron models.

    Returns:
        dict: Dictionary with the default input transforms.
    """
    return {
        'global': {
            'image_file': {
                'class': 'nexusml.engine.data.transforms.vision.detectron.IdentityImageTransform',
                'args': []
            }
        },
        'specific': None
    }


def detection_default_output_transforms():
    """
    Returns the default output transforms for object detection models.

    Returns:
        dict: Dictionary with the default output transforms.
    """
    return {
        'global': {
            'shape': {
                'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform',
                'args': {
                    'problem_type': 4
                }
            },
            'category': {
                'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform',
                'args': {
                    'problem_type': 4
                }
            }
        },
        'specific': None
    }


def segmentation_default_output_transforms():
    """
    Returns the default output transforms for object segmentation models.

    Returns:
        dict: Dictionary with the default output transforms.
    """
    return {
        'global': {
            'shape': {
                'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform',
                'args': {
                    'problem_type': 5
                }
            },
            'category': {
                'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform',
                'args': {
                    'problem_type': 4
                }
            }
        },
        'specific': None
    }


def get_data_section_config(train_data: str, test_data: str):
    """
    Returns the data section configuration.

    Args:
        train_data (str): Path to the training data.
        test_data (str): Path to the testing data.

    Returns:
        dict: Dictionary with the data section configuration.
    """
    return {'train_data': train_data, 'test_data': test_data}


def get_experiment_tracking_section_config(experiment_name: str, experiment_save_path: str, save_figures: bool,
                                           save_predictions: bool):
    """
    Returns the experiment tracking section configuration.

    Args:
        experiment_name (str): Name of the experiment.
        experiment_save_path (str): Path to save the experiment.
        save_figures (bool): Whether to save the figures.
        save_predictions (bool): Whether to save the predictions.

    Returns:
        dict: Dictionary with the experiment tracking section configuration.
    """
    return {
        'name': experiment_name,
        'mlflow_uri': experiment_save_path,
        'save_figures': save_figures,
        'save_predictions': save_predictions
    }


def pytorch_vision_model_config(setup_function: str,
                                hidden_size: int = 512,
                                batch_norm: bool = True,
                                dropout_p1: float = 0.25,
                                dropout_p2: float = 0.5):
    """
    Returns the configuration for a Pytorch vision model.

    Args:
        setup_function (str): Setup function for the model.
        hidden_size (int): Size of the hidden layer.
        batch_norm (bool): Whether to use batch normalization.
        dropout_p1 (float): Dropout probability for the first layer.
        dropout_p2 (float): Dropout probability for the second layer.

    Returns:
        dict: Dictionary with the configuration.
    """
    input_transforms = vision_input_transforms()
    output_transforms = sklearn_default_output_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms
        },
        'dataframe_transforms': default_dataframe_transforms(),
        'model': {
            'class': 'nexusml.engine.models.vision.cnns.PytorchVisionModel',
            'args': {
                'setup_function': 'nexusml.engine.models.vision.cnns.' + setup_function,
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


def detectron_detection_model_config(setup_function: str, checkpoint_url: str):
    """
    Returns the configuration for a Detectron object detection model.

    Args:
        setup_function (str): Setup function for the model.
        checkpoint_url (str): URL to the checkpoint.

    Returns:
        dict: Dictionary with the configuration.
    """
    input_transforms = detectron_default_input_transforms()
    output_transforms = detection_default_output_transforms()
    dataframe_transforms = detection_default_dataframe_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms,
        },
        'dataframe_transforms': dataframe_transforms,
        'model': {
            'class': 'nexusml.engine.models.vision.detectron.DetectronObjectDetectionModel',
            'args': {
                'setup_function': setup_function,
                'setup_args': {
                    'checkpoint_url': checkpoint_url
                }
            }
        }
    }
    return config


def detectron_segmentation_model_config(setup_function: str, checkpoint_url: str):
    """
    Returns the configuration for a Detectron object segmentation model.

    Args:
        setup_function (str): Setup function for the model.
        checkpoint_url (str): URL to the checkpoint.

    Returns:
        dict: Dictionary with the configuration.
    """
    input_transforms = detectron_default_input_transforms()
    output_transforms = segmentation_default_output_transforms()
    dataframe_transforms = segmentation_default_dataframe_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms,
        },
        'dataframe_transforms': dataframe_transforms,
        'model': {
            'class': 'nexusml.engine.models.vision.detectron.DetectronObjectSegmentationModel',
            'args': {
                'setup_function': setup_function,
                'setup_args': {
                    'checkpoint_url': checkpoint_url
                }
            }
        }
    }
    return config


def get_pytorch_vision_training_args(epochs: List = [2, 5],
                                     lr: List = [0.005],
                                     batch_size: List = [32],
                                     num_workers: List = [4]):
    """
    Returns the training arguments for a PyTorch vision model.

    Args:
        epochs (List): List of epochs.
        lr (List): List of learning rates.
        batch_size (List): List of batch sizes.
        num_workers (List): List of number of workers.

    Returns:
        dict: Dictionary with the training arguments.
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


def generate_pytorch_vision_classification_model_configs(output_path: str,
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
    Generate PyTorch vision classification model configurations.

    Args:
        output_path (str): Path to save the configurations.
        schema_path (str): Path to the schema.
        train_data (str): Path to the training data.
        test_data (str): Path to the testing data.
        experiment_name (str): Name of the experiment.
        experiment_save_path (str): Path to save the experiment.
        save_figures (bool): Whether to save the figures.
        save_predictions (bool): Whether to save the predictions.
        categories_path (str): Path to the categories.
        seed (int): Seed for the experiment.

    Returns:
        None
    """
    # Check that the Pipeline Type is image classification/regression
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.IMAGE_CLASSIFICATION_REGRESSION:
        raise SchemaError("The schema does not follow the 'image_classification_regression' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)

    training_args = get_pytorch_vision_training_args()

    os.makedirs(output_path, exist_ok=True)

    embedding_linear_model_configs = {
        'setup_function': [
            'create_pytorch_resnet50_model', 'create_pytorch_resnet152_model', 'create_pytorch_efficientnet_model'
        ],
        'hidden_size': [512],
        'dropout_p1': [0.25],
        'dropout_p2': [0.5],
    }

    training_model_configs = [[(k, v) for v in vs] for k, vs in training_args.items()]
    training_model_configs = list(map(dict, itertools.product(*training_model_configs)))
    embedding_linear_model_configs = [[(k, v) for v in vs] for k, vs in embedding_linear_model_configs.items()]
    embedding_linear_model_configs = list(map(dict, itertools.product(*embedding_linear_model_configs)))

    count = 0
    for j, training_params in enumerate(training_model_configs):
        for i, model_params in enumerate(embedding_linear_model_configs):
            output_file = os.path.join(output_path, f'torch_emb_vision_{count + 1}.yaml')
            if os.path.exists(output_file):
                print(f'The output file "{output_file} exist. Not overwritten')
            else:
                save_config = pytorch_vision_model_config(**model_params)
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


def generate_pytorch_vision_detection_model_configs(output_path: str,
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
    Generate PyTorch vision detection model configurations.

    Args:
        output_path (str): Path to save the configurations.
        schema_path (str): Path to the schema.
        train_data (str): Path to the training data.
        test_data (str): Path to the testing data.
        experiment_name (str): Name of the experiment.
        experiment_save_path (str): Path to save the experiment.
        save_figures (bool): Whether to save the figures.
        save_predictions (bool): Whether to save the predictions.
        categories_path (str): Path to the categories.
        seed (int): Seed for the experiment.

    Returns:
        None
    """
    # Check that the Pipeline Type is object detection
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.OBJECT_DETECTION:
        raise SchemaError("The schema does not follow the 'object_detection' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)

    training_args = get_pytorch_vision_training_args(epochs=[300], batch_size=[8])

    os.makedirs(output_path, exist_ok=True)

    detectron_detection_model_configs = {
        'setup_function': ['nexusml.engine.models.vision.detectron.create_detection_model'],
        'checkpoint_url': ['COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml']
    }

    training_model_configs = [[(k, v) for v in vs] for k, vs in training_args.items()]
    training_model_configs = list(map(dict, itertools.product(*training_model_configs)))
    detectron_detection_model_configs = [[(k, v) for v in vs] for k, vs in detectron_detection_model_configs.items()]
    detectron_detection_model_configs = list(map(dict, itertools.product(*detectron_detection_model_configs)))

    count = 0
    for j, training_params in enumerate(training_model_configs):
        for i, model_params in enumerate(detectron_detection_model_configs):
            output_file = os.path.join(output_path, f'detectron_detection_{count + 1}.yaml')
            if os.path.exists(output_file):
                print(f'The output file "{output_file} exist. Not overwritten')
            else:
                save_config = detectron_detection_model_config(**model_params)
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


def generate_pytorch_vision_segmentation_model_configs(output_path: str,
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
    Generate PyTorch vision segmentation model configurations.

    Args:
        output_path (str): Path to save the configurations.
        schema_path (str): Path to the schema.
        train_data (str): Path to the training data.
        test_data (str): Path to the testing data.
        experiment_name (str): Name of the experiment.
        experiment_save_path (str): Path to save the experiment.
        save_figures (bool): Whether to save the figures.
        save_predictions (bool): Whether to save the predictions.
        categories_path (str): Path to the categories.
        seed (int): Seed for the experiment.

    Returns:
        None
    """
    # Check that the Pipeline Type is object segmentation
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.OBJECT_SEGMENTATION:
        raise SchemaError("The schema does not follow the 'object_segmentation' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)

    training_args = get_pytorch_vision_training_args(epochs=[200], batch_size=[8])

    os.makedirs(output_path, exist_ok=True)

    detectron_detection_model_configs = {
        'setup_function': ['nexusml.engine.models.vision.detectron.create_segmentation_model'],
        'checkpoint_url': ['COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml']
    }

    training_model_configs = [[(k, v) for v in vs] for k, vs in training_args.items()]
    training_model_configs = list(map(dict, itertools.product(*training_model_configs)))
    detectron_detection_model_configs = [[(k, v) for v in vs] for k, vs in detectron_detection_model_configs.items()]
    detectron_detection_model_configs = list(map(dict, itertools.product(*detectron_detection_model_configs)))

    count = 0
    for j, training_params in enumerate(training_model_configs):
        for i, model_params in enumerate(detectron_detection_model_configs):
            output_file = os.path.join(output_path, f'detectron_segmentation_{count + 1}.yaml')
            if os.path.exists(output_file):
                print(f'The output file "{output_file} exist. Not overwritten')
            else:
                save_config = detectron_segmentation_model_config(**model_params)
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
