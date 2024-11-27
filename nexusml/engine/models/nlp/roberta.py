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
from transformers import RobertaModel

from nexusml.engine.data.datasets.nlp.classification import NLPDataset
from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.utils import predictions_to_example_format
from nexusml.engine.exceptions import ConfigFileError
from nexusml.engine.models.base import Model
from nexusml.engine.models.base import TrainingOutputInfo
from nexusml.engine.models.common.pytorch import _from_class_name_to_constructor
from nexusml.engine.models.common.pytorch import _get_loss_function_from_config
from nexusml.engine.models.common.pytorch import basic_predict_loop
from nexusml.engine.models.common.pytorch import basic_train_loop
from nexusml.engine.models.common.pytorch import BasicLossFunction
from nexusml.engine.models.common.pytorch import freeze_all_but_bn
from nexusml.engine.models.nlp.data_collator import DataCollatorWithPadding
from nexusml.engine.models.utils import smooth
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import TaskType


class ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks based on RoBerta model.
    It contains a linear that matches the input features to the number of classes
    And, in case of being in evaluation model, it applies the Softmax activation so the output is viewed as probability
    """

    def __init__(self, hidden_size: int, dropout_prob: float, num_labels: int):
        """
        Default constructor
        Args:
            hidden_size (int): number of features that will be input
            dropout_prob (float): the probability of applying DropOut after Linear layer (None or 0 to no use)
            num_labels (int): the number of output classes
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features: torch.Tensor, **kwargs):
        """
        Forward call of the module
        Args:
            features: input tensor to the module

        Returns:
            tensor after apply the linear layers and the softmax activation (if is in eval mode)
        """
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if not self.training:
            x = self.softmax(x)
        return x


class RegressionHead(nn.Module):
    """
    Head for sentence-level regression tasks based on RoBerta model.
    It simply contains a linear
    """

    def __init__(self, hidden_size: int, dropout_prob: float):
        """
        Default constructor
        Args:
            hidden_size (int): number of features that will be input
            dropout_prob (float): the probability of applying DropOut after Linear layer (None or 0 to no use)
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor, **kwargs):
        """
        Forward call of the module
        Args:
            features: input tensor to the module

        Returns:
            tensor after apply the linear layers
        """
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SharedTransformerBackbone(nn.Module):
    """
    General classifier model
    The model will extract features with base model and will pass the features to the classifiers
    It will concatenate the inputs features and pass them to each classifier
    It returns a list with the output of each classifier
    """

    def __init__(self, base_model: RobertaModel):
        """
        Constructor
        Args:
            base_model(nn.Module): feature extractor model
        """
        super().__init__()
        self.base_model = base_model

    def forward(self, x: dict):
        inputs = []
        for k, v in x.items():
            k_input = self.base_model(**v)
            inputs.append(k_input.last_hidden_state[:, 0, :])

        inputs = torch.cat(inputs, dim=1)

        return inputs


class NLPModule(nn.Module):
    """
    PyTorch module for NLP that connects the given backbone layer with each output
    """

    def __init__(self, backbone: nn.Module, output_layers: nn.ModuleDict, name_mapping: dict):
        """
        Default constructor
        Args:
            backbone (nn.Module): PyTorch module for NLP that contains the model backbone where the input will be passed
            output_layers (nn.ModuleDict): a pytorch module for each output. The backbone output will be passed to each
                                        output layer
            name_mapping (dict): a dict with the name mapping. The layer names (keys) are modified to delete
                                    the points "." because they are not allowed. We have to reconvert the output name
                                    to the original name
        """
        super().__init__()
        self.backbone = backbone
        self.output_layers = output_layers
        self.name_mapping = name_mapping

    def forward(self, x: dict):
        """
        Forward method called with input data to make predictions
        Args:
            x: dict with one element per input as {'input_id': tensor_data}

        Returns:
            Dict with the prediction of each output
        """
        features = self.backbone(x)
        outputs = {self.name_mapping[k]: v(features) for k, v in self.output_layers.items()}
        return outputs


def _get_default_optimizer(train_args: Dict = None):
    """
    Function that gets the default optimizer (Adam)
    If not learning rate (lr) is given in train_args, the default value is used (1e-3)
    Args:
        train_args (Dict): arguments used for training, where the lr will be get if present

    Returns:
        Optimizer constructor already filled with learning rate
    """
    if 'lr' in train_args:
        lr = train_args['lr']
    else:
        lr = 1e-3
    return functools.partial(optim.AdamW, lr=lr)


def _get_default_scheduler(train_dl: DataLoader, train_args: Dict = None):
    """
    Function that gets the default Learning Rate Scheduler (CosineAnnealingLR)
    If not learning rate (lr) is given in train_args, the default value is used (1e-3)
    'epochs' must be present on train_args for calculating the number of iterations
    Args:
        train_dl (DataLoader): train DataLoader used for calculating the total number of iterations
        train_args (Dict): arguments used for training, where the lr will be get if present. It requires
                        an 'epochs' element for calculating the number of iterations

    Returns:
        Learning Rate Scheduler constructor already filled with learning rate
    """
    total_steps = train_args['epochs'] * len(train_dl)

    return functools.partial(optim.lr_scheduler.CosineAnnealingLR, T_max=total_steps)


def _get_optimizer_from_config(optimizer_config: Dict):
    raise NotImplementedError()


def _get_scheduler_from_config(scheduler_config: Dict):
    raise NotImplementedError()


class TransformersNLPModel(Model):
    """
    Model class specialization for PyTorch NLP models
    """

    def __init__(self,
                 schema: Schema,
                 model_config: Dict,
                 categories: Categories,
                 dataframe_transforms: DataFrameTransforms,
                 input_transforms: ElementTransforms,
                 output_transforms: ElementTransforms,
                 inference_mode: bool = False):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            model_config (Dict): the configuration to be used for model construction
            dataframe_transforms (DataFrameTransforms): global transformation that are applied to whole DataFrame
            input_transforms (ElementTransforms): transformations that are applied on each input element
            output_transforms (ElementTransforms): transformations that are applied on each output element
            inference_mode (bool): argument that allows us to create the model as inference mode so the schema
                                and model configuration won't be needed. In this mode, we only will be able
                                to call predict method (cannot fit the model)
        """
        super().__init__(schema=schema,
                         categories=categories,
                         model_config=model_config,
                         dataframe_transforms=dataframe_transforms,
                         input_transforms=input_transforms,
                         output_transforms=output_transforms,
                         inference_mode=inference_mode)
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.transformers_model = None
        self.dc = None
        self.loss_function = None
        self.train_args = None

    def _setup_model(self):
        """
        Function called to set up the model using the configuration given in the constructor
        Returns:

        """
        # 'setup_function' and 'setup_args' are required
        if 'setup_function' not in self.model_config:
            raise ConfigFileError("'setup_function' key missing")
        if 'setup_args' not in self.model_config:
            raise ConfigFileError("'setup_args' key missing")
        # Get setup function callable
        setup_function = _from_class_name_to_constructor(self.model_config['setup_function'])
        # Create the PyTorch model using the previously built function
        self.transformers_model = setup_function(schema=self.schema,
                                                 input_transforms=self.input_transforms,
                                                 output_transforms=self.output_transforms,
                                                 **self.model_config['setup_args'],
                                                 **self.model_config['pretrained_kwargs'])

    def fit(self,
            train_data: Union[pd.DataFrame, dict, List[dict]],
            valid_data: Union[pd.DataFrame, dict, List[dict]] = None,
            train_args: Dict = None) -> TrainingOutputInfo:
        """
        Function called to train the model
        Args:
            train_data (Union[pd.DataFrame, dict, List[dict]]): train data that could be a pd.DataFrame, a
                                                                single example as dict, or a list of dict examples
            valid_data (Union[pd.DataFrame, dict, List[dict]]): validation data that could be a pd.DataFrame, a
                                                                single example as dict, or a list of dict examples
            train_args (Dict): dict with extra arguments for training like number of epochs.
                            Required keys: 'batch_size' and 'epochs'

        Returns:
            TrainingOutputInfo filled with the train history figures for each output
        """
        if isinstance(train_data, dict) or isinstance(train_data, list):
            train_data = Model.examples_to_dataframe(train_data)

        if isinstance(valid_data, dict) or isinstance(valid_data, list):
            valid_data = Model.examples_to_dataframe(valid_data)
        self.input_transforms.fit(train_data)
        self.output_transforms.fit(train_data)

        self.input_transforms.train()
        self.output_transforms.train()

        # Set up the PyTorch model
        self._setup_model()

        # If given train_args is None, get the saved args
        if train_args is None:
            train_args = self.train_args
        else:
            # We have new training args, save them
            self.train_args = train_args

        # Get training device. 'cuda' if GPU is available. 'cpu' otherwise
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_ds = NLPDataset(df=train_data,
                              input_transforms=self.input_transforms,
                              output_transforms=self.output_transforms,
                              train=True)

        self.dc = DataCollatorWithPadding(input_transforms=self.input_transforms)
        num_workers = train_args['num_workers'] if 'num_workers' in train_args else 0
        # Only drop last if there is more than one batch on Dataset
        drop_last = len(train_ds) > train_args['batch_size']
        train_dl = DataLoader(train_ds,
                              batch_size=train_args['batch_size'],
                              drop_last=drop_last,
                              shuffle=True,
                              collate_fn=self.dc,
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
            optimizer = _get_optimizer_from_config(train_args['optimizer'])
        else:
            optimizer = _get_default_optimizer(train_args)
        optimizer = optimizer(self.transformers_model.parameters())

        if 'scheduler' in train_args:
            scheduler = _get_scheduler_from_config(train_args['scheduler'])
        else:
            scheduler = _get_default_scheduler(train_dl, train_args)
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
            data (Union[pd.DataFrame, dict, List[dict]]): data that could be a pd.DataFrame, a single example
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

        self.dc = DataCollatorWithPadding(input_transforms=self.input_transforms)

        # Get the device. If not initialized, set device as 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ds = NLPDataset(df=data,
                        input_transforms=self.input_transforms,
                        output_transforms=self.output_transforms,
                        train=False)

        if train_args is None:
            train_args = self.train_args

        bs = 1 if train_args is None or 'batch_size' not in train_args else train_args['batch_size']
        num_workers = 0 if train_args is None or 'num_workers' not in train_args else train_args['num_workers']
        dl = DataLoader(ds, batch_size=bs, drop_last=False, shuffle=False, collate_fn=self.dc, num_workers=num_workers)

        predictions = basic_predict_loop(model=self.transformers_model, dl=dl, device=device)

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
            if i['type'] != 'text':
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
                'args': {
                    'select_shapes': False
                },
                'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements'
            }, {
                'args': None,
                'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues'
            }],
            'model': {
                'args': {
                    'pretrained_kwargs': {
                        'pretrained_model_name_or_path': 'xlm-roberta-base'
                    },
                    'setup_args': {
                        'dropout_p1': 0.25
                    },
                    'setup_function': 'nexusml.engine.models.nlp.roberta.create_transformers_classifier_model'
                },
                'class': 'nexusml.engine.models.nlp.roberta.TransformersNLPModel'
            },
            'training': {
                'batch_size': 32,
                'epochs': 30,
                'loss_function': {
                    'args': {
                        'classification_cost_sensitive': True
                    },
                    'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction'
                },
                'lr': 0.005,
                'num_workers': 0
            },
            'transforms': {
                'input_transforms': {
                    'global': {
                        'text': {
                            'args': {
                                'path': 'xlm-roberta-base'
                            },
                            'class': 'nexusml.engine.data.transforms.nlp.text.BasicNLPTransform'
                        }
                    },
                    'specific': None
                },
                'output_transforms': {
                    'global': {
                        'category': {
                            'args': None,
                            'class': 'nexusml.engine.data.transforms.sklearn.LabelEncoderTransform'
                        },
                        'float': {
                            'args': None,
                            'class': 'nexusml.engine.data.transforms.sklearn.MinMaxScalerTransform'
                        }
                    },
                    'specific': None
                }
            }
        }]

    def save_model(self, output_file: Union[str, IO]) -> None:
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
            'pretrained_kwargs': self.model_config['pretrained_kwargs'],
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
        pytorch_model = setup_function(schema=schema,
                                       output_transforms=output_transforms,
                                       **setup_args,
                                       **model_info['pretrained_kwargs'])

        # Create a buffer with state dict
        state_dict_buff = io.BytesIO()
        state_dict_buff.write(model_info['state_dict'])
        state_dict_buff.seek(0)

        # Set the state dict to model
        pytorch_model.load_state_dict(torch.load(state_dict_buff))

        # Return pytorch model as dict
        return {'transformers_model': pytorch_model, 'train_args': model_info['train_args']}

    def summary(self) -> Optional[str]:
        """
        Returns the summary of the trained model. In this case, just the string representation of the
        PyTorch model is returned

        Returns:
            string that will contain the summary of the PyTorch model (just string representation)
        """
        return str(self.transformers_model)


def create_classifier_model(inputs_info: List[Dict],
                            outputs_info: List[Dict],
                            output_column_transforms: ElementTransforms,
                            dropout_p1: float = 0.25,
                            **kwargs: dict) -> NLPModule:
    """
    Creates a PyTorch NLP Model using the Backbone from a pretrained RoBerta model
    Args:
        inputs_info (List[Dict]): list with the information of each input element
        outputs_info (List[Dict]): list with the information of each output element
        output_column_transforms (ElementTransforms): transformations that are applied on each output element
        dropout_p1 (float): the probability of applying DropOut after Linear layer (None or 0 to no use)

    Returns:
        NLPModule with the SharedTransformerBackbone
    """
    base_model = SharedTransformerBackbone(base_model=RobertaModel.from_pretrained(**kwargs))

    base_model.apply(freeze_all_but_bn)

    output_layers, name_mapping = _setup_output_layers(last_num_feat=base_model.base_model.config.hidden_size *
                                                       len(inputs_info),
                                                       outputs_info=outputs_info,
                                                       output_column_transforms=output_column_transforms,
                                                       dropout_p1=dropout_p1)

    model = NLPModule(backbone=base_model, output_layers=output_layers, name_mapping=name_mapping)
    return model


def create_transformers_classifier_model(schema: Schema,
                                         input_transforms: ElementTransforms,
                                         output_transforms: ElementTransforms,
                                         dropout_p1: float = None,
                                         **kwargs: dict) -> NLPModule:
    """
    Creates a PyTorch NLP Model using the Backbone from a pretrained RoBerta model
    Args:
        schema (Schema): the task schema
        input_transforms (ElementTransforms): Transforms for input columns
        output_transforms (ElementTransforms): transformations that are applied on each output element
        dropout_p1 (float): the probability of applying DropOut after Linear layer (None or 0 to no use)

    Returns:
        NLPModule with the SharedTransformerBackbone
    """
    transformers_model = create_classifier_model(inputs_info=schema.inputs,
                                                 outputs_info=schema.outputs,
                                                 output_column_transforms=output_transforms,
                                                 dropout_p1=dropout_p1,
                                                 **kwargs)
    return transformers_model


def _setup_output_layers(last_num_feat: int,
                         outputs_info: List[Dict],
                         output_column_transforms: ElementTransforms,
                         dropout_p1: float = None) -> Tuple[nn.ModuleDict, dict]:
    """
    Function that set up an output layer for each output of the schema
    Args:
        last_num_feat (int): how many features does the backbone output
        outputs_info (List[Dict]): list with the infor of all outputs of the schema
        output_column_transforms (ElementTransforms): the transformations that are applied to output elements
        dropout_p1 (float): the probability of applying DropOut after Linear layer (None or 0 to no use)

    Returns:
        Tuple:
            - nn.ModuleDict where the key is the output id and the value a PyTorch module
            - dict with the mapping (modified output name => original output name)
    """
    output_layers = {}
    # Name mapping (points "." are not allowed on nn.ModuleDict, so we replace them)
    name_mapping = {}
    for i in outputs_info:
        output_id = i['name'].replace('.', '#_:_#')
        # Add to mapping
        name_mapping[output_id] = i['name']
        tfm_out_info = output_column_transforms.get_transform(i['name']).get_transform_output_info()
        if tfm_out_info.output_type == 'category':
            output_layers[output_id] = ClassificationHead(hidden_size=last_num_feat,
                                                          dropout_prob=dropout_p1,
                                                          num_labels=len(tfm_out_info.choices))
        elif tfm_out_info.output_type in ['float', 'int']:
            output_layers[output_id] = RegressionHead(hidden_size=last_num_feat, dropout_prob=dropout_p1)
        else:
            raise ValueError(f'Output type "{i["type"]}" not supported')

    output_layers = nn.ModuleDict(output_layers)
    return output_layers, name_mapping


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
