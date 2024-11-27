import functools
import io
import pickle
from typing import Dict, IO, List, Optional, Tuple, Union

from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from nexusml.engine.data.datasets.audio.pytorch import SpeechDataset
from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.utils import predictions_to_example_format
from nexusml.engine.exceptions import ConfigFileError
from nexusml.engine.models.audio.collator import DataCollatorWithPadding
from nexusml.engine.models.base import Model
from nexusml.engine.models.base import TrainingOutputInfo
from nexusml.engine.models.common.pytorch import _from_class_name_to_constructor
from nexusml.engine.models.common.pytorch import _get_loss_function_from_config
from nexusml.engine.models.common.pytorch import basic_predict_loop
from nexusml.engine.models.common.pytorch import basic_train_loop
from nexusml.engine.models.common.pytorch import BasicLossFunction
from nexusml.engine.models.utils import smooth
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import TaskType


def _get_default_optimizer(train_args: Dict = None):
    if 'lr' in train_args:
        lr = train_args['lr']
    else:
        lr = 1e-3
    return functools.partial(optim.AdamW, lr=lr)


def _get_default_scheduler(train_dl: DataLoader, train_args: Dict = None):
    """
    Function that gets the default Learning Rate Scheduler (OneCycleLR)
    If not learning rate (lr) is given in train_args, the default value is used (1e-3)
    'epochs' must be present on train_args for calculating the number of iterations
    Args:
        train_dl (DataLoader): train DataLoader used for calculating the total number of iterations
        train_args (Dict): arguments used for training, where the lr will be get if present. It requires
                        an 'epochs' element for calculating the number of iterations

    Returns:
        Learning Rate Scheduler constructor already filled with learning rate
    """
    if 'lr' in train_args:
        lr = train_args['lr']
    else:
        lr = 1e-3

    # Calculate total of steps
    total_steps = train_args['epochs'] * len(train_dl)

    return functools.partial(optim.lr_scheduler.OneCycleLR, max_lr=lr, total_steps=total_steps)


def _get_optimizer_from_config(optimizer_config: Dict):
    raise NotImplementedError()


def _get_scheduler_from_config(scheduler_config: Dict):
    raise NotImplementedError()


class CustomWav2Vec2ModelForClassification(Model):
    """
    Custom model for Wav2Vec2 for classification tasks.
    """

    def __init__(self,
                 schema: Schema,
                 model_config: Dict,
                 categories: Categories,
                 dataframe_transforms: DataFrameTransforms,
                 input_transforms: ElementTransforms,
                 output_transforms: ElementTransforms,
                 inference_mode: bool = False):
        super().__init__(schema=schema,
                         categories=categories,
                         model_config=model_config,
                         dataframe_transforms=dataframe_transforms,
                         input_transforms=input_transforms,
                         output_transforms=output_transforms,
                         inference_mode=inference_mode)
        self.transformers_model = None
        self.data_collator = None
        self.loss_function = None
        self.train_args = None

    def _setup_model(self):
        """
        Set up the model for training.
        """
        if 'setup_function' not in self.model_config:
            raise ConfigFileError("'setup_function' key missing")
        if 'setup_args' not in self.model_config:
            raise ConfigFileError("'setup_args' key missing")
        setup_function = _from_class_name_to_constructor(self.model_config['setup_function'])
        self.transformers_model = setup_function(schema=self.schema,
                                                 output_transforms=self.output_transforms,
                                                 **self.model_config['setup_args'],
                                                 **self.model_config['pretrained_kwargs'])

    def fit(self,
            train_data: Union[pd.DataFrame, dict, List[dict]],
            val_data: Union[pd.DataFrame, dict, List[dict]] = None,
            train_args: Dict = None) -> TrainingOutputInfo:

        if isinstance(train_data, dict) or isinstance(train_data, list):
            train_data = Model.examples_to_dataframe(train_data)

        if isinstance(val_data, dict) or isinstance(val_data, list):
            val_data = Model.examples_to_dataframe(val_data)

        self.input_transforms.fit(train_data)
        self.output_transforms.fit(train_data)

        self.input_transforms.train()
        self.output_transforms.train()

        self._setup_model()

        # If given train_args is None, get the saved args
        if train_args is None:
            train_args = self.train_args
        else:
            # We have new training args, save them
            self.train_args = train_args

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_ds = SpeechDataset(df=train_data,
                                 input_transforms=self.input_transforms,
                                 output_transforms=self.output_transforms)

        self.data_collator = DataCollatorWithPadding(input_transforms=self.input_transforms)

        num_workers = train_args['num_workers'] if 'num_workers' in train_args else 0
        # Only drop last if there is more than one batch on Dataset
        drop_last = len(train_ds) > train_args['batch_size']
        train_dl = DataLoader(train_ds,
                              batch_size=train_args['batch_size'],
                              drop_last=drop_last,
                              shuffle=True,
                              collate_fn=self.data_collator,
                              num_workers=num_workers)

        # Create loss function if it is given in training args
        # Otherwise get the default loss function
        if 'loss_function' in train_args:
            self.loss_function = _get_loss_function_from_config(loss_function_args=train_args['loss_function'],
                                                                outputs_info=self.schema.outputs,
                                                                output_transforms=self.output_transforms,
                                                                device=device)
        else:
            self.loss_function = _get_default_loss_function(outputs_info=self.schema.outputs,
                                                            output_transforms=self.output_transforms,
                                                            device=device)

        if 'optimizer' in train_args:
            optimizer = _get_optimizer_from_config(optimizer_config=train_args['optimizer'])
        else:
            optimizer = _get_default_optimizer(train_args=train_args)
        optimizer = optimizer(self.transformers_model.parameters())

        if 'scheduler' in train_args:
            scheduler = _get_scheduler_from_config(scheduler_config=train_args['scheduler'])
        else:
            scheduler = _get_default_scheduler(train_dl=train_dl, train_args=train_args)
        scheduler = scheduler(optimizer=optimizer)

        train_hist = basic_train_loop(model=self.transformers_model,
                                      loss_function=self.loss_function,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      epochs=train_args['epochs'],
                                      train_dl=train_dl,
                                      device=device)

        loss_hist_figures = {}
        for k, v in train_hist.items():
            fig = plt.figure(figsize=(12, 9))
            plt.plot(v)
            plt.title(f'Training loss for "{k}"')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            loss_hist_figures[k] = fig

            fig = plt.figure(figsize=(12, 9))
            plt.plot(smooth(v, weight=0.9))
            plt.title(f'Smooth (0.9) training loss for "{k}"')
            plt.xlabel('Steps')
            plt.ylabel('Smooth Loss')
            loss_hist_figures[f'smooth_{k}'] = fig

        return TrainingOutputInfo(figures=loss_hist_figures)

    def predict(self,
                data: Union[pd.DataFrame, dict, List[dict]],
                split_predictions_by_output: bool = False,
                train_args: dict = None) -> Union[Dict, List]:
        """
        Function called to make predictions on the given data
        Args:
            data (Union[pd.DataFrame, dict, List[dict]]): data that could be a DataFrame, a single example
                                                    as dict, or a list of dict examples
            split_predictions_by_output (bool): if False, a list will be returned with the NexusML example format
                                                if True, a dict will be returned with one key per output with the
                                                predictions as value
            train_args (Dict): dict with extra arguments for training. It is used to get the
                            'batch_size' and create the DataLoader. If not given, batch_size=1 will be used

        Returns:
            It can be one of this two:
                - List of predictions following the NexusML example format (if split_predictions_by_output is False)
                - Dict with the prediction for each output element (if split_predictions_by_output is True)
        """

        if isinstance(data, dict) or isinstance(data, list):
            data = Model.examples_to_dataframe(data)

        self.input_transforms.eval()
        self.output_transforms.eval()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if train_args is None:
            train_args = self.train_args

        predict_ds = SpeechDataset(df=data,
                                   input_transforms=self.input_transforms,
                                   output_transforms=self.output_transforms,
                                   train=False)
        bs = 1 if train_args is None or 'batch_size' not in train_args else train_args['batch_size']
        num_workers = 0 if train_args is None or 'num_workers' not in train_args else train_args['num_workers']
        predict_dl = DataLoader(predict_ds,
                                batch_size=bs,
                                drop_last=False,
                                shuffle=False,
                                collate_fn=self.data_collator,
                                num_workers=num_workers)
        predictions = basic_predict_loop(model=self.transformers_model, dl=predict_dl, device=device)

        predictions = self.output_transforms.inverse_transform(predictions)
        # If not split_predictions_by_output, return as example
        if split_predictions_by_output:
            return predictions
        else:
            return predictions_to_example_format(predictions=predictions, output_transforms=self.output_transforms)

    @classmethod
    def supports_schema(cls, schema: Schema) -> bool:
        """
        Determine if the model can run given a specific schema.

        This method checks whether the current model is compatible with the provided
        schema. It inspects the schema and returns True if the model can successfully
        run with the provided schema, or False otherwise.

        Args:
            schema (Schema): The schema object to validate against the model.

        Returns:
            bool: True if the model can run with the provided schema, False otherwise.
        """
        if schema.task_type not in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            return False
        for i in schema.inputs:
            if i['type'] != 'audio_file':
                return False
        return True

    @classmethod
    def get_default_configs(cls) -> List[dict]:
        """
        Retrieve all possible default configurations for the model.

        This method returns a list of dictionaries representing various default
        configurations that the model supports.

        Args:
            None.

        Returns:
            List[dict]: A list of dictionaries, each representing a different default
            configuration supported by the model.
        """
        return [{
            'dataframe_transforms': [{
                'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements',
                'args': {
                    'select_shapes': False
                },
            }, {
                'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
                'args': None
            }],
            'model': {
                'class': 'nexusml.engine.models.audio.wav2vec2.CustomWav2Vec2ModelForClassification',
                'args': {
                    'pretrained_kwargs': {
                        'pretrained_model_name_or_path': 'facebook/wav2vec2-xls-r-300m'
                    },
                    'setup_args': {
                        'dropout_p1': 0.25
                    },
                    'setup_function': 'nexusml.engine.models.audio.wav2vec2.create_speech_classification_model'
                }
            },
            'training': {
                'batch_size': 8,
                'epochs': 25,
                'loss_function': {
                    'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction',
                    'args': None
                },
                'lr': 0.005,
                'num_workers': 0
            },
            'transforms': {
                'input_transforms': {
                    'global': {
                        'audio_file': {
                            'class': 'nexusml.engine.data.transforms.audio.speech.DefaultSpeechTransform',
                            'args': {
                                'path': 'facebook/wav2vec2-xls-r-300m',
                                'target_sr': 16000
                            },
                        }
                    },
                    'specific': None
                },
                'output_transforms': {
                    'global': {
                        'category': {
                            'class': 'nexusml.engine.data.transforms.sklearn.LabelEncoderTransform',
                            'args': None
                        },
                        'float': {
                            'class': 'nexusml.engine.data.transforms.sklearn.MinMaxScalerTransform',
                            'args': None
                        }
                    },
                    'specific': None
                }
            }
        }]

    def summary(self) -> Optional[str]:
        return str(self.transformers_model)

    def save_model(self, output_file: Union[str, IO]):
        """
        Method that saves all the information needed to create the PyTorch model serialized in the given output_file
        In this case, we will store the information needed to create the model (all information
        used inside _setup_model function). Then, when the model is created, we have to load the weights.
        So we store the state_dict of the model too
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        # Things to be saved
        to_store = {
            'setup_function': self.model_config['setup_function'],
            'setup_args': self.model_config['setup_args'],
            'train_args': self.train_args
        }
        # Also, the state dict as bytes
        state_dict_buff = io.BytesIO()
        torch.save(self.transformers_model.to('cpu').state_dict(), state_dict_buff)
        state_dict_buff.seek(0)
        to_store['state_dict'] = state_dict_buff.read()

        # If the given output file is a string, open the file and write the object (serialized with pickle)
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                pickle.dump(to_store, f)
        else:
            # If is not a string, write the object there
            pickle.dump(to_store, output_file)

    @classmethod
    def load_model(cls, input_file: Union[str, IO], schema: Schema, input_transforms: ElementTransforms,
                   output_transforms: ElementTransforms, dataframe_transforms: DataFrameTransforms) -> Dict:
        """
        Class method that loads all the needed information for making predictions with the model
        If the given input file is string, it will be the path from where read the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) from where read the object
        Args:
            schema (Schema): schema used for training the model
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read object
            input_transforms (ElementTransforms): input transforms already load that mau be needed for creating model
            output_transforms (ElementTransforms): output transforms already load that mau be needed for creating model
            dataframe_transforms (DataFrameTransforms): dataframe transforms already load that mau be needed for
                                                creating model

        Returns:
            Dict with key/value pairs to be set on the model object with setattr
        """
        # If is string, open the file and read the object (serialized with pickle)
        if isinstance(input_file, str):
            with open(input_file, 'rb') as f:
                model_info = pickle.load(f)
        else:
            # If is not a string, read the object there
            model_info = pickle.load(input_file)

        # Get the setup function
        setup_function = _from_class_name_to_constructor(model_info['setup_function'])

        # Get setup_args
        setup_args = model_info['setup_args']

        # Create the model
        pytorch_model = setup_function(schema=schema, output_transforms=output_transforms, **setup_args)

        # Create a buffer with state dict
        state_dict_buff = io.BytesIO()
        state_dict_buff.write(model_info['state_dict'])
        state_dict_buff.seek(0)

        # Set the state dict to model
        pytorch_model.load_state_dict(torch.load(state_dict_buff))

        # Return pytorch model as dict


class SpeechModule(nn.Module):
    """
    Speech Module.
    """

    def __init__(self, backbone: nn.Module, output_layers: Dict, name_mapping: Dict):
        super().__init__()
        self.backbone = backbone
        self.output_layers = output_layers
        self.name_mapping = name_mapping

    def forward(self, x):
        """ Forward pass through the model """
        features = self.backbone(x)
        outputs = {self.name_mapping[k]: v(features) for k, v in self.output_layers.items()}
        return outputs


class SharedWav2Vec2Backbone(nn.Module):
    """
    Shared Wav2Vec2 Backbone.
    """

    def __init__(self, base_model: Wav2Vec2Model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: dict):
        """ Forward pass through the model """
        inputs = []
        for k, v in x.items():
            k_input = self.base_model(**v)
            inputs.append(k_input.last_hidden_state[:, 0, :])

        inputs = torch.cat(inputs, dim=1)

        return inputs


class Wav2Vec2ClassificationHead(nn.Module):
    """
    Head for the wav2vec2 model for classification.
    """

    def __init__(self, hidden_size: int, dropout_prob: float, num_labels: int):
        super().__init__()
        self.projector = nn.Linear(hidden_size, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(256, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features: torch.Tensor, **kwargs):
        """ Forward pass through the model """
        x = features
        x = self.projector(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        if not self.training:
            x = self.softmax(x)
        return x


class Wav2Vec2RegressionHead(nn.Module):
    """
    Head for the wav2vec2 model for regression.
    """

    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.projector = nn.Linear(hidden_size, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, **kwargs):
        """ Forward pass through the model """
        x = features
        x = self.projector(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.regressor(x)
        return x


def _setup_output_layers(last_num_features: int,
                         outputs_info: List[Dict],
                         output_transforms: ElementTransforms,
                         dropout_p1: float = None) -> Tuple[nn.ModuleDict, dict]:
    output_layers = {}
    # Name mapping (points "." are not allowed on nn.ModuleDict, so we replace them)
    name_mapping = {}
    for i, output_info in enumerate(outputs_info):
        output_id = output_info['name'].replace('.', '#_:_#')
        # Add to mapping
        name_mapping[output_id] = output_info['name']
        output_column_transform_info = output_transforms.get_transform(output_info['name']).get_transform_output_info()

        if output_column_transform_info.output_type == 'category':
            output_layers[output_id] = Wav2Vec2ClassificationHead(hidden_size=last_num_features,
                                                                  dropout_prob=dropout_p1,
                                                                  num_labels=len(output_column_transform_info.choices))
        elif output_column_transform_info.output_type in ['float', 'int']:
            output_layers[output_id] = Wav2Vec2RegressionHead(hidden_size=last_num_features, dropout_prob=dropout_p1)
        else:
            raise ValueError(f'Unknown output type {output_column_transform_info.output_type}')

    output_layers = nn.ModuleDict(output_layers)
    return output_layers, name_mapping


def create_speech_classification_model(schema: Schema,
                                       output_transforms: ElementTransforms,
                                       dropout_p1: float = None,
                                       **kwargs: dict) -> SpeechModule:
    """
    Create a speech classification model

    Args:
        schema (Schema): schema used for training the model
        output_transforms (ElementTransforms): output transforms already load that mau be needed for creating model
        dropout_p1 (float): dropout probability for the first dropout layer

    Returns:
        SpeechModule: speech classification model
    """
    wav2vec2backbone = SharedWav2Vec2Backbone(Wav2Vec2Model.from_pretrained(**kwargs))
    # wav2vec2backbone.apply(freeze_all_but_bn)
    wav2vec2backbone.base_model.feature_extractor._freeze_parameters()
    for param in wav2vec2backbone.base_model.parameters():
        param.requires_grad = False
    output_layers, name_mapping = _setup_output_layers(
        last_num_features=wav2vec2backbone.base_model.config.output_hidden_size * len(schema.inputs),
        outputs_info=schema.outputs,
        output_transforms=output_transforms,
        dropout_p1=dropout_p1)
    model = SpeechModule(backbone=wav2vec2backbone, output_layers=output_layers, name_mapping=name_mapping)
    return model


def _get_default_loss_function(outputs_info: List[Dict], output_transforms: ElementTransforms, device: str):
    """
    Function that returns the default loss function, in this case, BaisLossFunction
    Args:
        outputs_info (List[Dict]): the information of each output element
        output_transforms (ElementTransforms): transformations applied to output elements
        device (str): device to use for training

    Returns:
        BasicLossFunction object already build
    """
    loss_function = BasicLossFunction(outputs_info=outputs_info, output_transforms=output_transforms, device=device)
    return loss_function
