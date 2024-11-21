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

from nexusml.engine.data.datasets.multimodal.magnum import MagnumDataset
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
from nexusml.engine.models.multimodal.magnum.data_collator import MultimodalDataCollatorWithPadding
from nexusml.engine.models.multimodal.magnum.low_level_module import RoBERTaPromptBottleneck
from nexusml.engine.models.multimodal.magnum.low_level_module import TabularMapper
from nexusml.engine.models.multimodal.magnum.low_level_module import ViTPromptBottleneck
from nexusml.engine.models.multimodal.magnum.wrapper import BottomLevelModule
from nexusml.engine.models.multimodal.magnum.wrapper import Magnum
from nexusml.engine.models.multimodal.magnum.wrapper import TopLevelModule
from nexusml.engine.models.utils import smooth
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import TaskType


class MagnumModule(nn.Module):
    """
    Magnum module that encodes multiple modalities (tabular, vision, language), applies fusion, and outputs predictions
    using classification or regression layers. The module handles tabular, vision, and language data separately at the
    bottom level and fuses them at the top level before passing them through output layers.
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
        Initializes the Magnum module.

        The module processes tabular, vision, and language data, performs modality-specific encoding,
        applies a fusion mechanism, and uses classification/regression layers to generate the final output.

        Args:
            output_layers (nn.ModuleDict): A dictionary of output layers for each prediction task.
            output_naming_map (dict): A mapping to rename the output layers.
            d_model (int): The hidden size of the embeddings.
            n_prompts (int): The number of prompt tokens to retain.
            knn_k (int): The number of nearest neighbors for graph pooling.
            gate_input_type (str): The input type for the gating mechanism in the fusion layer.
            gate_output_type (str): The output type for the gating mechanism in the fusion layer.
            modalities (list): The list of modalities (e.g., "tabular", "language", "vision").
            n_num_vars (int): The number of numerical variables in the tabular data.
            n_cat_vars (int): The number of categorical variables in the tabular data.
            num_cat_vars_classes (List[int]): A list of class counts for each categorical variable in the tabular data.
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
        """
        Forward pass.

        Args:
            x (dict): A dictionary containing the input data for each modality (tabular, image, and text).

        Returns:
            dict: The output of the model after applying the classification/regression heads.
        """
        tab_data = x['tabular'] if 'tabular' in x else None
        vis_data = x['image'] if 'image' in x else None
        lan_data = x['text'] if 'text' in x else None
        output = self.magnum(tab_data=tab_data, vis_data=vis_data, lan_data=lan_data)

        return output


class MagnumModel(Model):
    """
    Magnum model for vision, text, and tabular data.
    It handles data transformation, dataset creation, model setup, training, and prediction for multimodal tasks.
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
        Initializes the MagnumModel with the provided configuration, transformations, and schema.

        Args:
            schema (Schema): The schema of the task, describing inputs, outputs, and their types.
            model_config (Dict): Configuration dict for model construction, containing setup functions and arguments.
            categories (Categories): Possible values for categorical features.
            dataframe_transforms (DataFrameTransforms): Global transformations applied to the entire DataFrame.
            input_transforms (ElementTransforms): Transformations applied to input columns.
            output_transforms (ElementTransforms): Transformations applied to output columns.
            inference_mode (bool): If True, the model is set to inference mode.
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
        Sets up the MAGNUM model using the configuration provided in the constructor.
        It retrieves the setup function and arguments from the model configuration and initializes the PyTorch model.

        Raises:
            ConfigFileError: If required setup keys are missing in the model configuration.
        """
        # 'setup_function' and 'setup_args' are required
        if 'setup_function' not in self.model_config:
            raise ConfigFileError('"setup_function" key missing')
        if 'setup_args' not in self.model_config:
            raise ConfigFileError('"setup_args" key missing')
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
        Trains the MAGNUM model using the provided training data and arguments.

        Args:
            train_data (Union[pd.DataFrame, dict, List[dict]]): The training data in DataFrame or dict format.
            valid_data (Union[pd.DataFrame, dict, List[dict]], optional): The validation data.
            train_args (Dict): Training arguments such as batch size, epochs, etc.

        Returns:
            TrainingOutputInfo: Contains the training history and figures of the loss curves.
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

        train_ds = MagnumDataset(schema=self.schema,
                                 df=train_data,
                                 input_transform_functions=self.input_transforms.element_transform_map,
                                 output_transform_functions=self.output_transforms.element_transform_map,
                                 train=True)
        # Get the text type input element (only one allowed)
        text_input = [inp for inp in self.schema.inputs if inp['type'] == 'text'][0]
        # Get the tokenizer
        tokenizer = self.input_transforms.element_transform_map[text_input['name']].tokenizer_transform.tokenizer
        # Create Data Collator
        dc = MultimodalDataCollatorWithPadding(tokenizer=tokenizer)
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
        Generates predictions from the input data using the trained MAGNUM model.

        Args:
            data (Union[pd.DataFrame, dict, List[dict]]): Input data in DataFrame or dict format.
            split_predictions_by_output (bool): If True, returns a dict with separate predictions per output,
                                                otherwise returns a list of predictions.
            train_args (dict, optional): Training arguments for DataLoader creation.

        Returns:
            Union[Dict, List]: Predictions either as a dict (split by output) or as a list in predictions format.
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

        ds = MagnumDataset(schema=self.schema,
                           df=data,
                           input_transform_functions=self.input_transforms.element_transform_map,
                           output_transform_functions=self.output_transforms.element_transform_map,
                           train=False)

        if train_args is None:
            train_args = self.train_args

        bs = 1 if train_args is None or 'batch_size' not in train_args else train_args['batch_size']
        num_workers = 0 if train_args is None or 'num_workers' not in train_args else train_args['num_workers']
        # Get the text type input element (only one allowed)
        text_input = [inp for inp in self.schema.inputs if inp['type'] == 'text'][0]
        # Get the tokenizer
        tokenizer = self.input_transforms.element_transform_map[text_input['name']].tokenizer_transform.tokenizer
        # Create Data Collator
        dc = MultimodalDataCollatorWithPadding(tokenizer=tokenizer)
        dl = DataLoader(ds, batch_size=bs, drop_last=False, shuffle=False, collate_fn=dc, num_workers=num_workers)
        predictions = basic_predict_loop_magnum(model=self.magnum_model, dl=dl, device=device)
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
        has_image = False
        has_text = False
        has_tabular = False
        for i in schema.inputs:
            if i['type'] == 'image_file':
                if has_image:
                    # Only one input image supported
                    return False
                else:
                    has_image = True
            elif i['type'] == 'text':
                if has_text:
                    # Only one input text supported
                    return False
                else:
                    has_text = True
            elif i['type'] in ['category', 'integer', 'float']:
                has_tabular = True
            else:
                return False

        # At least two modalities
        n_modalities = sum([has_image, has_text, has_tabular])
        if n_modalities < 2:
            return False
        else:
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
                        'norm_layer': None,
                        'pretrained': True
                    },
                    'setup_args': {
                        'batch_norm': True,
                        'dropout_p1': 0.25,
                        'dropout_p2': 0.5,
                        'emb_size': 512,
                        'd_model': 256
                    },
                    'setup_function': 'nexusml.engine.models.multimodal.magnum.create_multimodal_magnum_model'
                },
                'class': 'nexusml.engine.models.multimodal.magnum.MagnumModel'
            },
            'training': {
                'batch_size': 8,
                'epochs': 30,
                'loss_function': {
                    'args': {},
                    'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction'
                },
                'lr': 0.00325,
                'num_workers': 4
            },
            'transforms': {
                'input_transforms': {
                    'global': {
                        'category': {
                            'args': None,
                            'class': 'nexusml.engine.data.transforms.sklearn.LabelEncoderTransform'
                        },
                        'float': {
                            'args': None,
                            'class': 'nexusml.engine.data.transforms.sklearn.StandardScalerTransform'
                        },
                        'image_file': {
                            'args': None,
                            'class': 'nexusml.engine.data.transforms.vision.torchvision.SquareImageTransform'
                        },
                        'text': {
                            'args': {
                                'path': 'roberta-large'
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
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read object
            schema (Schema): schema used for training the model
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


def _build_model(classifiers: nn.ModuleDict, output_naming_map: dict, d_model: int, inputs_info: List[Dict],
                 input_transforms: ElementTransforms) -> MagnumModule:
    """
    Builds a MAGNUM model by processing the input modalities and creating classifier heads for the outputs.

    This function identifies the modalities (vision, tabular, language) from the input information and
    creates the corresponding MagnumModule with the specified number of numerical and categorical variables
    for the tabular data. It also sets up classifier heads for each output of the model.

    Args:
        classifiers (nn.ModuleDict): A ModuleDict containing classifier heads for each model output.
        output_naming_map (dict): A mapping that reverts sanitized layer names (e.g., names without ".")
                                  back to the original names.
        d_model (int): The hidden size of the model.
        inputs_info (List[Dict]): Information about the model's inputs, including types and names.
        input_transforms (ElementTransforms): Transformations applied to each input element.

    Returns:
        MagnumModule: A constructed MagnumModule with the configured modalities and classifier heads.
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
                         output_layers=classifiers,
                         output_naming_map=output_naming_map,
                         d_model=d_model)
    return model


def create_magnum_model(inputs_info: List[Dict],
                        outputs_info: List[Dict],
                        input_transforms: ElementTransforms,
                        output_transforms: ElementTransforms,
                        emb_size: int,
                        d_model: int,
                        dropout_p1: float = None,
                        dropout_p2: float = None,
                        **kwargs: dict) -> MagnumModule:
    """
    Creates a MAGNUM model by setting up the output classifier heads and building the core model.

    Args:
        inputs_info (List[Dict]): Information about the model's inputs.
        outputs_info (List[Dict]): Information about the model's outputs.
        input_transforms (ElementTransforms): Transformations applied to input elements.
        output_transforms (ElementTransforms): Transformations applied to output elements.
        emb_size (int): The size of the embedding layer.
        d_model (int): The hidden size for the model layers.
        dropout_p1 (float, optional): Dropout probability for the first dropout layer.
        dropout_p2 (float, optional): Dropout probability for the second dropout layer.
        kwargs (dict): Additional arguments for model configuration.

    Returns:
        MagnumModule: The constructed MAGNUM model with classifier heads and configured modalities.
    """
    classifiers, name_mapping = _setup_output_layers(last_num_feat=d_model,
                                                     outputs_info=outputs_info,
                                                     output_transforms=output_transforms,
                                                     emb_size=emb_size,
                                                     dropout_p1=dropout_p1,
                                                     dropout_p2=dropout_p2)

    return _build_model(classifiers=classifiers,
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
    Classification head for the MAGNUM model.

    This layer applies batch normalization, dropout, and linear transformations to the input features,
    followed by a softmax activation function during evaluation to produce class probabilities.
    """

    def __init__(self,
                 input_features: int,
                 num_classes: int,
                 emb_size: int = 512,
                 dropout_p_1: float = 0.25,
                 dropout_p_2: float = 0.5):
        """
        Initializes the classification output layer with two dropout layers, two batch normalization layers,
        and a final softmax output for classification.

        Args:
            input_features (int): The number of input features.
            num_classes (int): The number of output classes.
            emb_size (int): The size of the embedding layer.
            dropout_p_1 (float): The dropout probability for the first dropout layer.
            dropout_p_2 (float): The dropout probability for the second dropout layer.
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
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes), containing logits during training
            or softmax probabilities during evaluation.
        """
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
    Regression head for the MAGNUM model.

    This layer applies batch normalization, dropout, and linear transformations to the input features
    and produces a single output for regression tasks.
    """

    def __init__(self, input_features: int, emb_size: int = 512, dropout_p_1: float = 0.25, dropout_p_2: float = 0.5):
        """
        Initializes the regression output layer with two dropout layers, two batch normalization layers,
        and a final linear layer for regression output.

        Args:
            input_features (int): The number of input features.
            emb_size (int): The size of the embedding layer.
            dropout_p_1 (float): The dropout probability for the first dropout layer.
            dropout_p_2 (float): The dropout probability for the second dropout layer.
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
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), containing the regression output.
        """
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
    Creates model heads for each output based on the output type (regression or classification).

    This function builds output layers (heads) for each output defined in the `outputs_info` using
    either regression or classification layers, depending on the output type. It also handles a name
    mapping since PyTorch `ModuleDict` does not allow keys with dots in the name.

    Args:
        last_num_feat (int): The number of input features for the output head.
        outputs_info (List[Dict]): A list of dictionaries containing information about each output.
        output_transforms (ElementTransforms): Transformations applied to the output elements.
        emb_size (int): The size of the embedding layer.
        dropout_p1 (float, optional): Dropout probability for the first dropout layer.
        dropout_p2 (float, optional): Dropout probability for the second dropout layer.

    Returns:
        Tuple[nn.ModuleDict, dict]: A `ModuleDict` with output heads and a name mapping for outputs.
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
    Creates a MAGNUM model configured for multimodal input and output tasks.

    This function initializes the MAGNUM model by processing the task schema and setting up
    the model with the appropriate input and output transformations, embedding sizes, and dropout layers.

    Args:
        schema (Schema): The task schema containing input and output information.
        input_transforms (ElementTransforms): Transformations applied to input columns.
        output_transforms (ElementTransforms): Transformations applied to output columns.
        emb_size (int): The size of the embedding layer.
        dropout_p1 (float, optional): Dropout probability for the first dropout layer.
        dropout_p2 (float, optional): Dropout probability for the second dropout layer.

    Returns:
        MagnumModule: The constructed MAGNUM model.
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
