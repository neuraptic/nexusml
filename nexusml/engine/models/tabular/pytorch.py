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

from nexusml.engine.data.datasets.tabular.pytorch import TabularDataset
from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.utils import predictions_to_example_format
from nexusml.engine.exceptions import ConfigFileError
from nexusml.engine.models.base import Model
from nexusml.engine.models.base import TrainingOutputInfo
from nexusml.engine.models.common.pytorch import _from_class_name_to_constructor
from nexusml.engine.models.common.pytorch import _get_loss_function_from_config
from nexusml.engine.models.common.pytorch import _get_optimizer_from_config
from nexusml.engine.models.common.pytorch import _get_scheduler_from_config
from nexusml.engine.models.common.pytorch import _setup_simple_output_layers
from nexusml.engine.models.common.pytorch import basic_predict_loop
from nexusml.engine.models.common.pytorch import basic_train_loop
from nexusml.engine.models.common.pytorch import BasicLossFunction
from nexusml.engine.models.utils import smooth
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import TaskType


def _setup_embedding_input_layer(inputs_info: List[Dict],
                                 input_transforms: ElementTransforms,
                                 emb_size: int,
                                 ensure_all_categorical: bool = False) -> Tuple[nn.ModuleDict, int, dict]:
    """
    Function that creating input layers using embedding layers for categorical inputs
    Args:
        inputs_info (List[Dict]): list with the information of each input element
        input_transforms (ElementTransforms): the transformations that are applied to each input element
        emb_size (int): the size used on embedding layer
        ensure_all_categorical (bool): if True, all input elements must be categorical for example, applying
                                    discretization. If False, the float inputs are forwarded untouched (Identity)

    Returns:
        Tuple:
            nn.ModuleDict that contains the input layer for each input element
            int with the total number of features that will be after the concatenation of all layer outputs
            dict with the name mapping (original name => modified name)
    """
    # To store input layers
    input_layer = {}
    # Initialize the number of features as 0
    total_num_features = 0
    # Name mapping (points "." are not allowed on nn.ModuleDict, so we replace them)
    name_mapping = {}
    # For each input
    for i in inputs_info:
        # Get ID
        input_id = i['name']
        # Replace "."
        input_id = input_id.replace('.', '#_:_#')
        # Add to mapping
        name_mapping[i['name']] = input_id
        # Check if the current input is used (it could be discarded by feature selection)
        if i['name'] in input_transforms.fit_columns:
            # Get the transformation output info
            tfm = input_transforms.get_transform(i['name'])
            tfm_out_info = tfm.get_transform_output_info()
            # If the output is a float
            if tfm_out_info.output_type == 'float' or tfm_out_info.output_type == 'int':
                # It all inputs must be categorical, raise Exception because the input is a float
                if ensure_all_categorical:
                    raise ValueError('Got a non categorical feature when "ensure_all_categorical" is set')
                # If floats are allowed, add an identity layer for this input, so the value is untouched
                input_layer[input_id] = nn.Identity()
                # Update total number of features
                total_num_features += tfm_out_info.num_features
            elif tfm_out_info.output_type == 'category':
                # If input is categorical, create an Embedding layer and update the number of features
                input_layer[input_id] = nn.Embedding(len(tfm_out_info.choices), emb_size)
                total_num_features += emb_size
            else:
                raise ValueError(f'Input feature type "{i["type"]}" not supported')
    # Convert to nn.ModuleDict
    input_layer = nn.ModuleDict(input_layer)
    return input_layer, total_num_features, name_mapping


def _setup_linear_backbone(num_input_features: int,
                           features_per_layer: List[int],
                           batch_norm: bool = True,
                           dropout_p: float = None) -> nn.ModuleDict:
    """
    Function that uses a set of linear layers as model backbone
    Builds a set of:
        BatchNorm (optional)
        Linear
        ReLU
        DropOut (optional)
    Args:
        num_input_features (int): number of total input features (concatenation of all input features)
        features_per_layer (List[int]): one element per Linear layer indicating the number of features that outputs each
        batch_norm (bool): use or not BatchNorm layer before Linear layer
        dropout_p (float): the probability of applying DropOut after Linear layer (None or 0 to no use)

    Returns:
        nn.ModuleDict where each element is a Linear block composed as explained before
    """
    # To store linear layers
    linear_layers = {}
    # Set the previous number of features as the number of features that will be input
    prev_layer_output = num_input_features
    # For each linear layer to build
    for i, n_feat in enumerate(features_per_layer):
        # Create a list to store layer of the current linear block
        sub_layers = []
        # If apply batch norm, add it to lost
        if batch_norm:
            sub_layers.append(nn.BatchNorm1d(prev_layer_output))
        # Add Linear layer followed by ReLU to list
        sub_layers.append(nn.Linear(prev_layer_output, n_feat))
        sub_layers.append(nn.ReLU())
        # If apply DropOut, add it to list
        if dropout_p is not None and (0.0 < dropout_p < 1.0):
            sub_layers.append(nn.Dropout(p=dropout_p))
        # Build Sequential with current layers and add it to dict
        linear_layers[f'lin{i + 1}'] = nn.Sequential(*sub_layers)
        # Update number of features
        prev_layer_output = n_feat
    # Convert to nn.ModuleDict
    linear_layers = nn.ModuleDict(linear_layers)
    return linear_layers


def _setup_transformer_based_backbone(input_emb_size: int, n_heads: int, dim_feedforward: int,
                                      num_encoder_layers: int) -> nn.Module:
    """
    Function that set a backbone using transformer layers
    Args:
        input_emb_size (int): the size of the input token lengths
        n_heads (int): number of heads of the transformer encoder layer
        dim_feedforward (int): dimension of the feed forward layer
        num_encoder_layers (int): number of encoder layer to be stacked

    Returns:
        nn.Module with the set of transform encoder layers
    """
    transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_emb_size,
                                                           nhead=n_heads,
                                                           dim_feedforward=dim_feedforward)
    transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_encoder_layers)
    return transformer_encoder


class EmbeddingLinearBackbone(nn.Module):
    """
    PyTorch module that wraps the Linear backbone using Embedding input layers
    """

    def __init__(self,
                 inputs_info: List[Dict],
                 input_transforms: ElementTransforms,
                 emb_size: int,
                 features_per_layer: List[int],
                 batch_norm: bool = True,
                 dropout_p: float = None):
        """
        Default constructor
        Args:
            inputs_info (List[Dict]): information of each input element
            input_transforms (ElementTransforms): the transformations that are applied to each input element
            emb_size (int): the embedding size used in the Embedding layer
            features_per_layer (List[int]): one element per Linear layer indicating the number of features that outputs
            batch_norm (bool): use or not BatchNorm layer before Linear layer
            dropout_p (float): the probability of applying DropOut after Linear layer (None or 0 to no use)
        """
        super().__init__()
        # Create the input layers and get the total number of input features
        self.input_layers, self.num_input_features, self.name_mapping = _setup_embedding_input_layer(
            inputs_info=inputs_info, input_transforms=input_transforms, emb_size=emb_size, ensure_all_categorical=False)
        # Create the backbone composed of linear layers
        self.linear_layers = _setup_linear_backbone(num_input_features=self.num_input_features,
                                                    features_per_layer=features_per_layer,
                                                    batch_norm=batch_norm,
                                                    dropout_p=dropout_p)

    def forward(self, x):
        """
        Forward method called to encode the input data
        Args:
            x: dict with one element per input as {'input_id': tensor_data}

        Returns:
            tensor after applying the input layers and the linear backbone
        """
        # To store all features
        features = []
        # For each input
        for k, v in x.items():
            # Pass it over its input layer
            k_feature = self.input_layers[self.name_mapping[k]](v)
            # It could have 3 dimensions, but the second one is single element. So convert to 2 dimensions removing it
            if k_feature.ndim == 3:
                assert k_feature.shape[1] == 1
                k_feature = k_feature[:, 0, :]
            # Append to feature list
            features.append(k_feature)
        # Concatenate all features by dim 1
        features = torch.cat(features, dim=1)

        # For each linear layer
        for k, v in self.linear_layers.items():
            # Forward features and update them
            features = v(features)

        # Return features after passing them for all linear layers
        return features


class TransformerBasedBackbone(nn.Module):
    """
    PyTorch module that wraps the Transformer backbone using Embedding input layers
    """

    def __init__(self, inputs_info: List[Dict], input_transforms: ElementTransforms, emb_size: int, n_heads: int,
                 dim_feedforward: int, num_encoder_layers: int):
        """
        Default constructor
        Args:
            inputs_info (List[Dict]): information of each input element
            input_transforms (ElementTransforms): the transformations that are applied to each input element
            emb_size (int): the embedding size used in the Embedding layer
            n_heads (int): number of heads of the transformer encoder layer
            dim_feedforward (int): dimension of the feed forward layer
            num_encoder_layers (int): number of encoder layer to be stacked
        """
        super().__init__()
        # Create the input layers and get the total number of input features
        self.input_layers, self.num_input_features, self.name_mapping = _setup_embedding_input_layer(
            inputs_info=inputs_info, input_transforms=input_transforms, emb_size=emb_size, ensure_all_categorical=True)
        # Create the backbone composed of transformer layers
        self.transformer_layer = _setup_transformer_based_backbone(input_emb_size=emb_size,
                                                                   n_heads=n_heads,
                                                                   dim_feedforward=dim_feedforward,
                                                                   num_encoder_layers=num_encoder_layers)

    def forward(self, x):
        """
        Forward method called to encode the input data
        Args:
            x: dict with one element per input as {'input_id': tensor_data}

        Returns:
            tensor after applying the input layers and the transformer backbone
        """
        # To store all features
        features = []
        # For each input
        for k, v in x.items():
            # Feed to embedding, getting (bs, emb_size) or (bs, 1, emb_size)
            # Make all have (1, bs, emb_size), so then we can concatenate all on dim 0
            # getting (sl, bs, emb_size) being sl sequence length
            k_feature = self.input_layers[self.name_mapping[k]](v)
            if k_feature.ndim == 3:
                assert k_feature.shape[1] == 1
                k_feature = k_feature[:, 0, :]
            features.append(k_feature.unsqueeze(dim=0))
        # Cat all in sequence dim (0)
        features = torch.cat(features, dim=0)

        # Feed to transformer layer
        features = self.transformer_layer(features)

        # We have (sl, bs, emb_size)
        # We want (bs, sl*emb_size)
        # So swap dim=0 and dim=1 axes => (bs, sl, emb_size) and then flatten
        features = features.swapaxes(0, 1).flatten(start_dim=1, end_dim=-1)
        return features


class TabularPytorchModule(nn.Module):
    """
    PyTorch module that connects the given backbone layer with each output
    """

    def __init__(self, backbone: nn.Module, output_layers: nn.ModuleDict, output_naming_map: dict):
        """
        Default constructor
        Args:
            backbone (nn.Module): PyTorch module that contains the model backbone where the input will be passed
            output_layers (nn.ModuleDict): a pytorch module for each output. The backbone output will be passed to each
                                        output layer
            output_naming_map (dict): a dict with the name mapping. The layer names (keys) are modified to delete
                                    the points "." because they are not allowed. We have to reconvert the output name
                                    to the original name
        """
        super().__init__()
        # Store params
        self.backbone = backbone
        self.output_layers = output_layers
        self.output_naming_map = output_naming_map

    def forward(self, x):
        """
        Forward method called with input data to make predictions
        Args:
            x: dict with one element per input as {'input_id': tensor_data}

        Returns:
            Dict with the prediction of each output
        """
        # Encode input in features with backbone
        features = self.backbone(x)
        # Apply each output layer using the previous features
        outputs = {self.output_naming_map[k]: v(features) for k, v in self.output_layers.items()}
        return outputs


def create_pytorch_embedding_module(schema: Schema,
                                    input_transforms: ElementTransforms,
                                    output_transforms: ElementTransforms,
                                    emb_size: int,
                                    features_per_layer: List[int],
                                    batch_norm: bool = None,
                                    dropout_p: float = None) -> TabularPytorchModule:
    """
    Creates a PyTorch Tabular Model using the Embedding Linear Backbone
    Args:
        schema (Schema): the task schema
        input_transforms (ElementTransforms): transformations that are applied on each input element
        output_transforms (ElementTransforms): transformations that are applied on each output element
        emb_size (int): the embedding size used in the Embedding layer
        features_per_layer (List[int]): one element per Linear layer indicating the number of features that outputs
        batch_norm (bool): use or not BatchNorm layer before Linear layer
        dropout_p (float): the probability of applying DropOut after Linear layer (None or 0 to no use)

    Returns:
        TabularPytorchModule with the EmbeddingLinearBackbone
    """
    # Create backbone
    backbone = EmbeddingLinearBackbone(inputs_info=schema.inputs,
                                       input_transforms=input_transforms,
                                       emb_size=emb_size,
                                       features_per_layer=features_per_layer,
                                       batch_norm=batch_norm,
                                       dropout_p=dropout_p)
    # Create output layers
    output_layers, output_naming_map = _setup_simple_output_layers(last_num_feat=features_per_layer[-1],
                                                                   outputs_info=schema.outputs,
                                                                   output_transforms=output_transforms)
    return TabularPytorchModule(backbone=backbone, output_layers=output_layers, output_naming_map=output_naming_map)


def create_pytorch_transformer_module(schema: Schema, input_transforms: ElementTransforms,
                                      output_transforms: ElementTransforms, emb_size: int, n_heads: int,
                                      dim_feedforward: int, num_encoder_layers: int) -> TabularPytorchModule:
    """
    Creates a PyTorch Tabular Model using the Transformer Backbone
    Args:
        schema (Schema): the task schema
        input_transforms (ElementTransforms): transformations that are applied on each input element
        output_transforms (ElementTransforms): transformations that are applied on each output element
        emb_size (int): the embedding size used in the Embedding layer
        n_heads (int): number of heads of the transformer encoder layer
        dim_feedforward (int): dimension of the feed forward layer
        num_encoder_layers (int): number of encoder layer to be stacked

    Returns:
        TabularPytorchModule with the TransformerBasedBackbone
    """
    # Create backbone
    backbone = TransformerBasedBackbone(inputs_info=schema.inputs,
                                        input_transforms=input_transforms,
                                        emb_size=emb_size,
                                        n_heads=n_heads,
                                        dim_feedforward=dim_feedforward,
                                        num_encoder_layers=num_encoder_layers)
    # Create output layers
    output_layers, output_naming_map = _setup_simple_output_layers(last_num_feat=emb_size *
                                                                   len(input_transforms.fit_columns),
                                                                   outputs_info=schema.outputs,
                                                                   output_transforms=output_transforms)
    return TabularPytorchModule(backbone=backbone, output_layers=output_layers, output_naming_map=output_naming_map)


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


class PytorchTabularModel(Model):
    """
    Model class specialization for PyTorch tabular models
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
        # Call superclass constructor
        super().__init__(schema=schema,
                         categories=categories,
                         model_config=model_config,
                         dataframe_transforms=dataframe_transforms,
                         input_transforms=input_transforms,
                         output_transforms=output_transforms,
                         inference_mode=inference_mode)
        # Initialize PyTorch model, loss function and device as None
        self.pytorch_model = None
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
        self.pytorch_model = setup_function(schema=self.schema,
                                            input_transforms=self.input_transforms,
                                            output_transforms=self.output_transforms,
                                            **self.model_config['setup_args'])

    def fit(self,
            train_data: Union[pd.DataFrame, dict, List[dict]],
            valid_data: Union[pd.DataFrame, dict, List[dict]] = None,
            train_args: Dict = None) -> TrainingOutputInfo:
        """
        Function called to train the model
        Args:
            train_data (Union[pd.DataFrame, dict, List[dict]]): train data that could be a DataFrame, a single example
                                                            as dict, or a list of dict examples
            valid_data (Union[pd.DataFrame, dict, List[dict]]): validation data that could be a DataFrame, a
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

        # Fit the global DataFrame transformations and put it in train mode
        self.dataframe_transforms.fit(train_data)
        self.dataframe_transforms.train()
        # Apply DataFrame transformation to train data
        train_data = self.dataframe_transforms.transform(train_data)

        # Fit the transformed data to both, input and output transforms
        # Note: the data is not transformed here, the dataset class does it
        self.input_transforms.fit(train_data)
        self.output_transforms.fit(train_data)

        # Put them on train mode
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

        # Create the training dataset and the training DataLoader
        train_ds = TabularDataset(df=train_data,
                                  input_transforms=self.input_transforms,
                                  output_transforms=self.output_transforms,
                                  train=True)
        num_workers = train_args['num_workers'] if 'num_workers' in train_args else 0
        # Only drop last if there is more than one batch on Dataset
        drop_last = len(train_ds) > train_args['batch_size']
        train_dl = DataLoader(train_ds,
                              batch_size=train_args['batch_size'],
                              drop_last=drop_last,
                              shuffle=True,
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

        # Create the optimizer if it is given in train args
        # Otherwise, get the default optimizer
        if 'optimizer' in train_args:
            optimizer = _get_optimizer_from_config(optimizer_config=train_args['optimizer'])
        else:
            optimizer = _get_default_optimizer(train_args=train_args)
        # Create the optimizer with the model parameters
        optimizer = optimizer(self.pytorch_model.parameters())

        # Create the learning rate scheduler if it is given in train args
        # Otherwise, get the default scheduler
        if 'scheduler' in train_args:
            scheduler = _get_scheduler_from_config(scheduler_config=train_args['scheduler'])
        else:
            scheduler = _get_default_scheduler(train_dl=train_dl, train_args=train_args)
        # Create scheduler giving the created optimizer
        scheduler = scheduler(optimizer=optimizer)

        # Call basic train loop and get the train history of each output
        train_hist = basic_train_loop(model=self.pytorch_model,
                                      loss_function=self.loss_function,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      epochs=train_args['epochs'],
                                      train_dl=train_dl,
                                      device=device)

        # To store the train hist figures
        loss_hist_figures = {}
        # For each output
        for k, v in train_hist.items():
            # Plot the train loss evolution of the current output
            fig = plt.figure(figsize=(12, 9))
            plt.plot(v)
            plt.title(f'Training loss for "{k}"')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            loss_hist_figures[k] = fig

            # Plot the same line, but after applying the smooth function
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

        # Put global transformations, input transformations and output transformations on eval mode
        self.dataframe_transforms.eval()
        self.input_transforms.eval()
        self.output_transforms.eval()

        # Get the device. If not initialized, set device as 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Transform input data
        data = self.dataframe_transforms.transform(data)

        if train_args is None:
            train_args = self.train_args

        # Create DataSet and DataLoader
        ds = TabularDataset(df=data, input_transforms=self.input_transforms, output_transforms=None, train=False)
        bs = 1 if train_args is None or 'batch_size' not in train_args else train_args['batch_size']
        num_workers = 0 if train_args is None or 'num_workers' not in train_args else train_args['num_workers']
        dl = DataLoader(ds, batch_size=bs, drop_last=False, shuffle=False, num_workers=num_workers)
        # Make predictions
        predictions = basic_predict_loop(model=self.pytorch_model, dl=dl, device=device)
        # Apply inverse transform
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
            if i['type'] not in ['boolean', 'integer', 'float', 'category']:
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
        input_transforms = {
            'global': {
                'float': {
                    'class': 'nexusml.engine.data.transforms.sklearn.StandardScalerTransform',
                    'args': None
                },
                'category': {
                    'class': 'nexusml.engine.data.transforms.sklearn.OrdinalEncoderTransform',
                    'args': None
                }
            },
            'specific': None
        }
        output_transforms = {
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
        dataframe_transforms = [{
            'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements',
            'args': {
                'shapes': False
            }
        }, {
            'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
            'args': None
        }, {
            'class': 'nexusml.engine.data.transforms.sklearn.SimpleMissingValueImputation',
            'args': None
        }]
        return [{
            'transforms': {
                'input_transforms': input_transforms,
                'output_transforms': output_transforms
            },
            'dataframe_transforms': dataframe_transforms,
            'model': {
                'class': 'nexusml.engine.models.tabular.pytorch.PytorchTabularModel',
                'args': {
                    'setup_function': 'nexusml.engine.models.tabular.pytorch.create_pytorch_embedding_module',
                    'setup_args': {
                        'emb_size': 512,
                        'features_per_layer': (100, 50, 10),
                        'batch_norm': True,
                        'dropout_p': 0.5
                    }
                }
            }
        }]

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
        torch.save(self.pytorch_model.to('cpu').state_dict(), state_dict_buff)
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
                                       input_transforms=input_transforms,
                                       output_transforms=output_transforms,
                                       **setup_args)

        # Create a buffer with state dict
        state_dict_buff = io.BytesIO()
        state_dict_buff.write(model_info['state_dict'])
        state_dict_buff.seek(0)

        # Set the state dict to model
        pytorch_model.load_state_dict(torch.load(state_dict_buff))

        # Return pytorch model as dict
        return {'pytorch_model': pytorch_model, 'train_args': model_info['train_args']}

    def summary(self) -> Optional[str]:
        """
        Returns the summary of the trained model. In this case, just the string representation of the
        PyTorch model is returned

        Returns:
            string that will contain the summary of the PyTorch model (just string representation)
        """
        return str(self.pytorch_model)
