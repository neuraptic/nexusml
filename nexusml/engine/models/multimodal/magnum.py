import functools
import io
import pickle
from typing import Dict, IO, List, Optional, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nexusml.engine.data.dataset.multimodal.magnum import MagnumDataset
from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.utils import predictions_to_example_format
from nexusml.engine.exceptions import ConfigFileError
from nexusml.engine.models.base import Model
from nexusml.engine.models.base import TrainingOutputInfo
from nexusml.engine.models.common.pytorch import _from_class_name_to_constructor
from nexusml.engine.models.common.pytorch import _get_loss_function_from_config
from nexusml.engine.models.common.pytorch import _join_torch_dict
from nexusml.engine.models.common.pytorch import BasicLossFunction
from nexusml.engine.models.multimodal.magnum_model.data_collator import MultimodalDataCollatorWithPadding
from nexusml.engine.models.multimodal.magnum_model.low_level_module import RoBERTaPromptBottleneck
from nexusml.engine.models.multimodal.magnum_model.low_level_module import TabularMapper
from nexusml.engine.models.multimodal.magnum_model.low_level_module import ViTPromptBottleneck
from nexusml.engine.models.multimodal.magnum_model.wrapper import BottomLevelModule
from nexusml.engine.models.multimodal.magnum_model.wrapper import Magnum
from nexusml.engine.models.multimodal.magnum_model.wrapper import TopLevelModule
from nexusml.engine.models.utils import smooth
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories


class MagnumModule(nn.Module):
    """
    Magnum classifier model
    The model will extract features with base model and will pass the features to the classifiers
    It will concatenate the inputs features and pass them to each classifier
    It returns a list with the output of each classifier
    """

    def __init__(self,
                 output_layers: nn.ModuleDict,
                 output_naming_map: dict,
                 d_model: int = 256,
                 n_prompts: int = 8,
                 knn_k: int = 3,
                 gate_input_type: str = 'same',
                 gate_output_type: str = 'softmax-scalar',
                 modalities: List = ['tabular', 'language', 'vision'],
                 n_num_vars: int = None,
                 n_cat_vars: int = None,
                 num_cat_vars_classes: List = None):
        """
        Constructor
        Args:
            output_layers(Dict[str, nn.Module]): list of classifiers for each output of the model
            output_naming_map (dict): a dict with the name mapping. The layer names (keys) are modified to delete
                                    the points "." because they are not allowed. We have to reconvert the output name
                                    to the original name
        """
        super().__init__()
        self.output_layers = output_layers
        self.output_naming_map = output_naming_map

        self.d_model = d_model
        self.n_prompts = n_prompts
        self.knn_k = knn_k
        self.gate_input_type = gate_input_type
        self.gate_output_type = gate_output_type
        self.d_hidden = d_model

        tabular_model = TabularMapper(d_model=self.d_model,
                                      n_num_vars=n_num_vars,
                                      n_cat_vars=n_cat_vars,
                                      num_cat_vars_classes=num_cat_vars_classes)
        tabular_mapper = nn.Linear(self.d_model, self.d_model)

        language_model = RoBERTaPromptBottleneck(self.n_prompts)
        language_mapper = nn.Linear(language_model.d_model, self.d_model)

        vision_model = ViTPromptBottleneck(self.n_prompts)
        vision_mapper = nn.Linear(vision_model.d_model, self.d_model)

        bottom_level_module = BottomLevelModule(d_model=self.d_model,
                                                tabular_model=tabular_model,
                                                tabular_mapper=tabular_mapper,
                                                language_model=language_model,
                                                language_mapper=language_mapper,
                                                vision_model=vision_model,
                                                vision_mapper=vision_mapper)

        top_level_module = TopLevelModule(d_model=self.d_model,
                                          hidden_size=self.d_hidden,
                                          gate_input_type=self.gate_input_type,
                                          gate_output_type=self.gate_output_type,
                                          k=self.knn_k,
                                          output_layers=output_layers,
                                          output_naming_map=output_naming_map,
                                          modalities=modalities)

        self.magnum = Magnum(bottom_level_module, top_level_module)

    def forward(self, x):
        """ Forward pass """
        tab_data = x['tabular'] if 'tabular' in x else None
        vis_data = x['image'] if 'image' in x else None
        lan_data = x['text'] if 'text' in x else None
        output = self.magnum(tab_data=tab_data, vis_data=vis_data, lan_data=lan_data)

        return output


class MagnumModel(Model):
    """
    Magnum model for vision, text and tabular
    It transforms data, create datasets and trains the given model
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
        Constructor
        Args:
            schema(Schema): schema of the task
            categories (Categories): the possible values for categorical features
            model_config (Dict): the configuration to be used for model construction
            dataframe_transforms (DataFrameTransforms): global transformation that are applied to whole DataFrame
            input_transforms(ElementTransforms): transformations for input columns
            output_transforms(ElementTransforms): transformations for output columns
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
        self.magnum_model = None
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
        self.magnum_model = setup_function(schema=self.schema,
                                           input_transforms=self.input_transforms,
                                           output_transforms=self.output_transforms,
                                           **self.model_config['setup_args'],
                                           **self.model_config['pretrained_kwargs'])

    def fit(self,
            train_data: Union[pd.DataFrame, dict, List[dict]],
            valid_data: Union[pd.DataFrame, dict, List[dict]] = None,
            train_args: Dict = None) -> TrainingOutputInfo:
        """
        Model's fit
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

        train_ds = MagnumDataset(df=train_data,
                                 input_transform_functions=self.input_transforms.element_transform_map,
                                 output_transform_functions=self.output_transforms.element_transform_map,
                                 train=True)
        text_input = [inp for inp in self.schema.inputs if inp['type'] == 'text']
        dc = MultimodalDataCollatorWithPadding(
            text_transform=self.input_transforms.element_transform_map[text_input[0]['name']])
        num_workers = train_args['num_workers'] if 'num_workers' in train_args else 0
        # Only drop last if there is more than one batch on Dataset
        drop_last = len(train_ds) > train_args['batch_size']
        train_dl = DataLoader(train_ds,
                              batch_size=train_args['batch_size'],
                              drop_last=drop_last,
                              shuffle=True,
                              collate_fn=dc,
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
        optimizer = optimizer(self.magnum_model.parameters())

        if 'scheduler' in train_args:
            scheduler = _get_scheduler_from_config(scheduler_config=train_args['scheduler'])
        else:
            scheduler = _get_default_scheduler(train_dl=train_dl, train_args=train_args)
        scheduler = scheduler(optimizer=optimizer)

        train_hist = basic_train_loop_magnum(model=self.magnum_model,
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
        # Put global transformations, input transformations and output transformations on eval mode
        self.dataframe_transforms.eval()
        self.input_transforms.eval()
        self.output_transforms.eval()

        # Get the device. If not initialized, set device as 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Transform input data
        data = self.dataframe_transforms.transform(data)

        ds = MagnumDataset(df=data,
                           input_transform_functions=self.input_transforms.element_transform_map,
                           output_transform_functions=self.output_transforms.element_transform_map,
                           train=False)

        if train_args is None:
            train_args = self.train_args

        bs = 1 if train_args is None or 'batch_size' not in train_args else train_args['batch_size']
        num_workers = 0 if train_args is None or 'num_workers' not in train_args else train_args['num_workers']
        text_input = [inp for inp in self.schema.inputs if inp['type'] == 'text']
        dc = MultimodalDataCollatorWithPadding(
            text_transform=self.input_transforms.element_transform_map[text_input[0]['name']])
        dl = DataLoader(ds, batch_size=bs, drop_last=False, shuffle=False, collate_fn=dc, num_workers=num_workers)
        predictions = basic_predict_loop_magnum(model=self.magnum_model, dl=dl, device=device)
        predictions = self.output_transforms.inverse_transform(predictions)
        # If not split_predictions_by_output, return as example
        if split_predictions_by_output:
            return predictions
        else:
            return predictions_to_example_format(predictions=predictions, output_transforms=self.output_transforms)

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
        torch.save(self.magnum_model.to('cpu').state_dict(), state_dict_buff)
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
        magnum_model = setup_function(schema=schema,
                                      input_transforms=input_transforms,
                                      output_transforms=output_transforms,
                                      **setup_args)

        # Create a buffer with state dict
        state_dict_buff = io.BytesIO()
        state_dict_buff.write(model_info['state_dict'])
        state_dict_buff.seek(0)

        # Set the state dict to model
        magnum_model.load_state_dict(torch.load(state_dict_buff))

        # Return pytorch model as dict
        return {'magnum_model': magnum_model, 'train_args': model_info['train_args']}

    def summary(self) -> Optional[str]:
        """
        Returns the summary of the trained model. In this case, just the string representation of the
        PyTorch model is returned

        Returns:
            string that will contain the summary of the PyTorch model (just string representation)
        """
        return str(self.magnum_model)


def _build_model(clfs: nn.ModuleDict, output_naming_map: dict, d_model: int, inputs_info: List[Dict],
                 input_transforms: ElementTransforms) -> MagnumModule:
    """
    Function that builds a model from a base model and a list of classifier heads
    Args:
        clfs(ModuleDict): ModuleDict containing as many classifier heads as number of outputs of the model
        output_naming_map (dict): a dict with the name mapping. The layer names (keys) are modified to delete
                                    the points "." because they are not allowed. We have to reconvert the output name
                                    to the original name

    Returns:
        MagnumModule: constructed model
    """
    modalities = []
    for i in range(len(inputs_info)):
        if inputs_info[i]['type'] == 'image_file' and 'vision' not in modalities:
            modalities.append('vision')
        if inputs_info[i]['type'] in ['category', 'float', 'int'] and 'tabular' not in modalities:
            modalities.append('tabular')
        if inputs_info[i]['type'] == 'text' and 'language' not in modalities:
            modalities.append('language')
    n_num_vars = len([i for i in inputs_info if i['type'] in ['float', 'int']])
    n_num_vars = None if n_num_vars == 0 else n_num_vars
    n_cat_vars = len([i for i in inputs_info if i['type'] == 'category'])
    n_cat_vars = None if n_cat_vars == 0 else n_cat_vars
    num_cat_vars_classes = None
    if n_cat_vars is not None:
        num_cat_vars_classes = [
            len(input_transforms.element_transform_map[i['name']].get_transform_output_info().choices)
            for i in inputs_info
            if i['type'] == 'category'
        ]
    model = MagnumModule(modalities=modalities,
                         n_num_vars=n_num_vars,
                         n_cat_vars=n_cat_vars,
                         num_cat_vars_classes=num_cat_vars_classes,
                         output_layers=clfs,
                         output_naming_map=output_naming_map,
                         d_model=d_model)
    return model


def create_magnum_model(inputs_info: List[Dict],
                        outputs_info: List[Dict],
                        input_transforms: ElementTransforms,
                        output_transforms: ElementTransforms,
                        emb_size: int,
                        dropout_p1: float = None,
                        dropout_p2: float = None,
                        d_model: int = 256,
                        **kwargs: dict) -> MagnumModule:
    """
    Function that creates a MAGNUM model.
    It removes the default classifier head and adds as many classifier heads as number of outputs of the model
    Args:
        inputs_info (List[Dict]): inputs of the model
        outputs_info (List[Dict]): outputs of the model
        output_transforms (ElementTransforms): transformations that are applied on each output element
        emb_size (int): embedding size
        dropout_p1 (float): probability for first dropout layer
        dropout_p2 (float): probability for second dropout layer
        **kwargs(dict): arguments for the ResNet model

    Returns:
        MagnumModule
    """
    # ToDo: set d_model by config
    clfs, name_mapping = _setup_output_layers(last_num_feat=d_model,
                                              outputs_info=outputs_info,
                                              output_transforms=output_transforms,
                                              emb_size=emb_size,
                                              dropout_p1=dropout_p1,
                                              dropout_p2=dropout_p2)

    return _build_model(clfs=clfs,
                        output_naming_map=name_mapping,
                        d_model=d_model,
                        inputs_info=inputs_info,
                        input_transforms=input_transforms)


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
    return functools.partial(optim.AdamW, lr=lr, weight_decay=1e-5)


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

    total_steps = train_args['epochs'] * len(train_dl)

    return functools.partial(optim.lr_scheduler.CosineAnnealingLR, T_max=total_steps)


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


def _get_optimizer_from_config(optimizer_config: Dict):
    raise NotImplementedError()


def _get_scheduler_from_config(scheduler_config: Dict):
    raise NotImplementedError()


class ClassificationOutputLayer(nn.Module):
    """
    Classification head
    """

    def __init__(self,
                 input_features: int,
                 num_classes: int,
                 emb_size: int = 512,
                 dropout_p_1: float = 0.25,
                 dropout_p_2: float = 0.5):
        """
        Constructor
        Args:
            input_features(int): number of input features
            num_classes(int): number of classes
            emb_size(int): embedding size
            dropout_p_1(float): dropout probability for first dropout layer
            dropout_p_2(float): dropout probability for second dropout layer
        """
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_features)
        self.dropout1 = nn.Dropout(dropout_p_1)
        self.linear1 = nn.Linear(input_features, emb_size, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(emb_size)
        self.dropout2 = nn.Dropout(dropout_p_2)
        self.out = nn.Linear(in_features=emb_size, out_features=num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ Forward pass """
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.out(x)
        if not self.training:
            x = self.softmax(x)
        return x


class RegressionOutputLayer(nn.Module):
    """
    Regression head
    """

    def __init__(self, input_features: int, emb_size: int = 512, dropout_p_1: float = 0.25, dropout_p_2: float = 0.5):
        """
        Constructor
        Args:
            input_features(int): number of input features
            emb_size(int): embedding size
            dropout_p_1(float): dropout probability for first dropout layer
            dropout_p_2(float): dropout probability for second dropout layer
        """
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_features)
        self.dropout1 = nn.Dropout(dropout_p_1)
        self.linear1 = nn.Linear(input_features, emb_size, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(emb_size)
        self.dropout2 = nn.Dropout(dropout_p_2)
        self.out = nn.Linear(in_features=emb_size, out_features=1, bias=True)

    def forward(self, x):
        """ Forward pass """
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.out(x)
        return x


def _setup_output_layers(last_num_feat: int,
                         outputs_info: List[Dict],
                         output_transforms: ElementTransforms,
                         emb_size: int,
                         dropout_p1: float = None,
                         dropout_p2: float = None) -> Tuple[nn.ModuleDict, dict]:
    """
    Function that creates the model heads for each output
    Args:
        last_num_feat(int): number of input features for the head
        outputs_info(List[Dict]): outputs information
        output_transforms (ElementTransforms): the transformations that are applied to output elements
        emb_size(int): embedding size
        dropout_p1(float): dropout probability for first dropout layer
        dropout_p2(float): dropout probability for second dropout layer

    Returns:
        nn.ModuleDict where the key is the output id and the value a PyTorch module
    """
    output_layers = {}
    # Name mapping (points "." are not allowed on nn.ModuleDict, so we replace them)
    name_mapping = {}
    for i in outputs_info:
        output_id = i['name'].replace('.', '#_:_#')
        # Add to mapping
        name_mapping[output_id] = i['name']
        tfm_out_info = output_transforms.get_transform(i['name']).get_transform_output_info()
        if tfm_out_info.output_type in ['float', 'int']:
            output_layers[output_id] = RegressionOutputLayer(input_features=last_num_feat)
        elif tfm_out_info.output_type == 'category':
            output_layers[output_id] = ClassificationOutputLayer(input_features=last_num_feat,
                                                                 num_classes=len(tfm_out_info.choices),
                                                                 emb_size=emb_size,
                                                                 dropout_p_1=dropout_p1,
                                                                 dropout_p_2=dropout_p2)
        else:
            raise ValueError(f'Output type "{i["type"]}" not supported')

    output_layers = nn.ModuleDict(output_layers)
    return output_layers, name_mapping


def create_multimodal_magnum_model(schema: Schema,
                                   input_transforms: ElementTransforms,
                                   output_transforms: ElementTransforms,
                                   emb_size: int,
                                   dropout_p1: float = None,
                                   dropout_p2: float = None,
                                   **kwargs: dict) -> MagnumModule:
    """
    Creates MAGNUM model

    Args:
        schema (Schema): task schema
        input_transforms (ElementTransforms): Transforms for input columns
        output_transforms (ElementTransforms): Transforms for output columns
        emb_size (int): embedding size
        dropout_p1 (float): probability for first dropout layer
        dropout_p2 (float): probability for second dropout layer

    Returns:
        MagnumModule: MAGNUM model
    """
    magnum_model = create_magnum_model(inputs_info=schema.inputs,
                                       outputs_info=schema.outputs,
                                       input_transforms=input_transforms,
                                       output_transforms=output_transforms,
                                       emb_size=emb_size,
                                       dropout_p1=dropout_p1,
                                       dropout_p2=dropout_p2,
                                       **kwargs)
    return magnum_model


def basic_train_loop_magnum(model: nn.Module,
                            loss_function: nn.Module,
                            optimizer: optim.Optimizer,
                            scheduler: optim.lr_scheduler._LRScheduler,
                            epochs: int,
                            train_dl: DataLoader,
                            device: str,
                            verbose: bool = False) -> Dict:
    """
    Perform basic training of the given model
    Args:
        model (nn.Module): PyTorch model to be trained
        loss_function (nn.Module): loss function to use for training
        optimizer (optim.Optimizer): optimizer to use for training
        scheduler: the learning rate scheduler to use for training
        epochs (int): number of epochs to train the model
        train_dl (DataLoader): loader with training data
        device (str): device to use for training
        verbose (bool): to plot loss of each iteration or not

    Returns:
        Dict with the loss history of each output for each iteration
    """
    # Move model to the given device and set it to train mode
    model.to(device)
    model.train()
    # Init history as None
    loss_hist = None
    # For each epoch
    for epoch in range(1, epochs + 1):
        print(f'[+] Starting epoch {epoch}')
        # For each batch
        for x, y in tqdm(train_dl):
            # Reset gradients
            optimizer.zero_grad()
            # Move data to device
            x = {
                k: {
                    s: t.to(device) for s, t in v.items()
                } if isinstance(v, dict) else v.to(device) for k, v in x.items()
            }
            y = {k: v.to(device) for k, v in y.items()}
            # Make the prediction
            prediction = model(x)
            # Get the loss
            loss_by_output, loss = loss_function(prediction, y)
            if verbose:
                print(loss_by_output, loss)
            # Update loss history
            loss_hist = _join_torch_dict(gd=loss_hist, pd=loss_by_output)
            # Compute gradients
            loss.backward()
            # Update model
            optimizer.step()
            # Get next lr calling the scheduler
            if scheduler is not None:
                scheduler.step()
    return loss_hist


def basic_predict_loop_magnum(model: nn.Module, dl: DataLoader, device: str) -> Dict[str, np.ndarray]:
    """
    Perform basic prediction loop with the given model and return only the predictions
    Args:
        model (nn.Module): PyTorch model to be trained
        dl (DataLoader): loader with data to be predicted. It does not include the true target
        device (str): device to use for training
    Returns:
        Tuple:
            Dict with the predictions of each schema output
            Dict with the targets of each schema output, get from given DataLoader
    """
    # Move model to device and set as eval mode
    model.to(device)
    model.eval()
    # Initialize predictions
    predictions = {}
    n_examples = len(dl.dataset)
    # Predict one batch to get shapes and types
    for x in dl:
        # Move to device
        x = {k: {s: t.to(device) for s, t in v.items()} if isinstance(v, dict) else v.to(device) for k, v in x.items()}
        # Predict and concatenate
        prediction = model(x)

        for k, v in prediction.items():
            v = v.detach().cpu().numpy()
            shape = (n_examples,) if v.ndim == 1 else (n_examples, v.shape[1])
            dtype = v.dtype
            predictions[k] = np.empty(shape, dtype=dtype)
        break

    idx = 0
    # To no compute gradients
    with torch.no_grad():
        print('[+] Making predictions')
        # For each batch
        for x in tqdm(dl):
            # Move to device
            x = {
                k: {
                    s: t.to(device) for s, t in v.items()
                } if isinstance(v, dict) else v.to(device) for k, v in x.items()
            }
            # Predict and concatenate
            prediction = model(x)
            next_idx = None
            for k, v in prediction.items():
                if next_idx is None:
                    next_idx = idx + v.shape[0]
                predictions[k][idx:next_idx] = v.detach().cpu().numpy()
            idx = next_idx

    return predictions
