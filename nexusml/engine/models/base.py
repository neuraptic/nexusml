from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import importlib
import io
import json
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.utils import json_examples_to_data_frame
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories


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


@dataclass
class TrainingOutputInfo:
    """
    DataClass used for storing training output info, composed by parameters, figures and artifacts
    """
    params: Dict[str, Any] = None
    figures: Dict[str, Figure] = None
    artifacts: Dict[str, Path] = None


class Model(ABC):
    """
    Abstract class for models
    """

    def __init__(self,
                 schema: Optional[Schema],
                 categories: Categories,
                 model_config: Optional[Dict],
                 dataframe_transforms: DataFrameTransforms,
                 input_transforms: ElementTransforms,
                 output_transforms: ElementTransforms,
                 inference_mode: bool = False):
        """
        Default constructor
        Args:
            schema (Schema): the task schema. It could be None for inference mode
            categories (Categories): the possible values for categorical features
            model_config (Dict): the configuration to be used for model construction. It could be None for inference
            dataframe_transforms (DataFrameTransforms): global transformation that are applied to whole DataFrame
            input_transforms (ElementTransforms): transformations that are applied on each input element
            output_transforms (ElementTransforms): transformations that are applied on each output element
            inference_mode (bool): argument that allows us to create the model as inference mode so the schema
                                and model configuration won't be needed. In this mode, we only will be able
                                to call predict method (cannot fit the model)
        """
        self.schema = schema
        self.categories = categories
        self.model_config = model_config
        self.dataframe_transforms = dataframe_transforms
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.inference_mode = inference_mode

    @abstractmethod
    def fit(self,
            train_data: Union[pd.DataFrame, dict, List[dict]],
            valid_data: Union[pd.DataFrame, dict, List[dict]] = None,
            **kwargs) -> TrainingOutputInfo:
        """
        Abstract method that be called for making the model training
        Args:
            train_data (Union[pd.DataFrame, dict, List[dict]]): train data that could be a DataFrame, a single example
                                                            as dict, or a list of dict examples
            valid_data (Union[pd.DataFrame, dict, List[dict]]): validation data that could be a DataFrame,
                                                            a single example as dict, or a list of dict examples
            **kwargs: other arguments

        Returns:
            TrainingOutputInfo filled with the desired information
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self,
                data: Union[pd.DataFrame, dict, List[dict]],
                split_predictions_by_output: bool = False,
                **kwargs) -> Union[Dict, List]:
        """
        Abstract method that be called to make the prediction
        Args:
            data (Union[pd.DataFrame, dict, List[dict]]): data that could be a DataFrame, a single example
                                                    as dict, or a list of dict examples
            split_predictions_by_output (bool): if False, a list will be returned with the NexusML example format
                                                if True, a dict will be returned with one key per output with the
                                                predictions as value
            **kwargs: other arguments

        Returns:
            It can be one of this two:
                - List of predictions following the NexusML example format (if split_predictions_by_output is False)
                - Dict with the prediction for each output element (if split_predictions_by_output is True)
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def supports_schema(cls, schema: Schema) -> bool:
        """
        Determine if the model can run given a specific schema.

        This method checks whether the current model is compatible with the provided
        schema. Derived classes should implement logic that inspects the schema and
        returns True if the model can successfully run with the provided schema, or
        False otherwise.

        Args:
            schema (Schema): The schema object to validate against the model.

        Returns:
            bool: True if the model can run with the provided schema, False otherwise.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_default_configs(cls) -> List[dict]:
        """
        Retrieve all possible default configurations for the model.

        This method returns a list of dictionaries representing various default
        configurations that the model supports. This allows subclasses to define
        and return multiple default setups depending on the specific parameters
        or algorithms they can use.

        Args:
            None.

        Returns:
            List[dict]: A list of dictionaries, each representing a different default
            configuration supported by the model.
        """

    def explain(self, data: Optional[Union[pd.DataFrame, dict, List[dict]]] = None, **kwargs) -> Optional:
        """
        Explain the model or the data
        Note: we consider interpretability as a particular case of explainability where the AI model is
        "inherently explainable" and thus, we'll always be referring to explainability instead of interpretability.
        This terminology is based on MathWork's (https://es.mathworks.com/discovery/interpretability.html).
        Args:
            data (Optional[Union[pd.DataFrame, dict, List[dict]]]): data to be explained. If `None`, the
                                                                 model itself will be explained
            **kwargs: other arguments

        Returns:

        """
        return None

    def compute_templates(self,
                          data: Union[pd.DataFrame, dict, List[dict]],
                          output_file_path: Optional[str] = None) -> Optional[dict]:
        """
        Method that computes monitoring templates on a given data
        Args:
            data (Union[pd.DataFrame, dict, List[dict]]): data that could be a DataFrame, a single example
                                                    as dict, or a list of dict examples
            output_file_path (Optional[str]): JSON file path where store templates. If None, the templates
                                            will be returned instead of storing them

        Returns:
            Optional[dict]: if output_file_path a dict with templates will be returned. Otherwise None
        """
        if not isinstance(data, pd.DataFrame):
            data = Model.examples_to_dataframe(examples=data)
        predictions = self.predict(data=data, split_predictions_by_output=True)
        templates = {'outputs': []}
        for o in self.schema.outputs:
            if o['type'] in ['integer', 'float']:
                # Get mean and std
                m, s = np.mean(predictions[o['name']]), np.std(predictions[o['name']])
                templates['outputs'].append({'element': o['uuid'], 'template': {'mean': m.item(), 'std': s.item()}})
            elif o['type'] == 'category':
                # The output should be a DataFrame with the score for each example. Get the classified category
                df = predictions[o['name']]
                predicted_categories = df.columns[np.argmax(df.to_numpy(), axis=1)].to_numpy()
                # Init template
                output_template = []
                # Get the mean probability for each category
                for c in self.categories.categories_by_element[o['name']]:
                    cat_template = {'category': c['uuid'], 'template': []}
                    # Get the predictions of predicted category equal to c
                    sub_df = df.iloc[np.where(predicted_categories == c['name'])[0], :]
                    # Get the mean for each category
                    for c2 in self.categories.categories_by_element[o['name']]:
                        cat_template['template'].append({
                            'category':
                                c2['uuid'],
                            'mean':
                                sub_df[c2['name']].mean().item() if sub_df.shape[0] > 0 else 1.0 /
                                len(self.categories.categories_by_element[o['name']])
                        })
                    output_template.append(cat_template)
                templates['outputs'].append({'element': o['uuid'], 'template': output_template})
            else:
                continue

        if output_file_path is None:
            return templates
        else:
            with open(output_file_path, 'w') as f:
                json.dump(templates, f)
            return None

    @staticmethod
    def examples_to_dataframe(examples: Union[dict, List[dict]]) -> pd.DataFrame:
        """
        Function that converts the given example (dict) or examples (list of dicts) of NexusML format
        to pandas DataFrame (using pd.read_json)
        Args:
            examples (Union[dict, List[dict]]): single or multiple examples on NexusML format

        Returns:
            pd.DataFrame after converting the examples or examples
        """
        # If is a dict, it is a single example
        if isinstance(examples, dict):
            examples = [examples]
        # Convert to DataFrame
        return json_examples_to_data_frame(json_ex_list=examples)

    @abstractmethod
    def summary(self) -> Optional[str]:
        """
        Return a summary of the trained model. For example, in trees we can return the generated rules

        Returns:
            a model summary as string
        """
        raise NotImplementedError()

    def save(self, output_file: Union[str, IO]) -> None:
        """
        Method for saving the trained model
        It first save the schema, then the transformations and finally the model
        This class implements the method for saving the transformations, but the
        method to save the model must be implemented by each subclass
        Args:
           output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """

        schema_buff = io.BytesIO()
        pickle.dump(self.schema, schema_buff)
        schema_buff.seek(0)

        categories_buff = io.BytesIO()
        pickle.dump(self.categories, categories_buff)
        categories_buff.seek(0)

        model_config_buff = io.BytesIO()
        pickle.dump(self.model_config, model_config_buff)
        model_config_buff.seek(0)

        transforms_buff = io.BytesIO()
        self._save_transformations(transforms_buff)
        transforms_buff.seek(0)

        model_buff = io.BytesIO()
        self.save_model(model_buff)
        model_buff.seek(0)

        serialized_model = {
            'schema': schema_buff.read(),
            'categories': categories_buff.read(),
            'model_config': model_config_buff.read(),
            'transformations': transforms_buff.read(),
            'model_class': self.__class__,
            'model': model_buff.read()
        }

        # Serialize and save the transforms and model
        # If the given output file is a string, open the file and write the object (serialized with pickle)
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                pickle.dump(serialized_model, f)
        else:
            # If is not a string, write the object there
            pickle.dump(serialized_model, output_file)

    def _save_transformations(self, output_file: Union[str, IO]):
        """
        Serializes all transformations (input, output and DataFrame) calling all 'save' methods of each transformation
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        # Get the serialization of each transformation as byte array
        serialized_transformations = {}
        to_serialize = [self.input_transforms, self.output_transforms, self.dataframe_transforms]
        keys = ['input_transforms', 'output_transforms', 'dataframe_transforms']
        for k, v in zip(keys, to_serialize):
            buff = io.BytesIO()
            v.save(buff)
            buff.seek(0)
            serialized_transformations[k] = buff.read()

        # Serialize and save the transforms
        # If the given output file is a string, open the file and write the object (serialized with pickle)
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                pickle.dump(serialized_transformations, f)
        else:
            # If is not a string, write the object there
            pickle.dump(serialized_transformations, output_file)

    @abstractmethod
    def save_model(self, output_file: Union[str, IO]):
        """
        Abstract method for saving the model itself
        This does not include transformations
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def load(input_file: Union[str, IO]):
        """
        Method that loads all the needed information for creating the model
        First, the transformations are loaded and a Model subclass object is created with them
        Then, all needed information of the model for making inferences is loaded and set
        with the setattr method
        If the given input file is string, it will be the path from where read the objects
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) from where read the object
        Args:
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read object

        Returns:
            Model created and loaded from the given saved file
        """
        # If is string, open the file and read the object (serialized with pickle)
        if isinstance(input_file, str):
            with open(input_file, 'rb') as f:
                serialized_info = pickle.load(f)
        else:
            # If is not a string, read the object there
            serialized_info = pickle.load(input_file)

        # Get class constructor
        class_constructor = serialized_info['model_class']

        # Load schema
        schema_buff = io.BytesIO()
        schema_buff.write(serialized_info['schema'])
        schema_buff.seek(0)
        schema = pickle.load(schema_buff)

        # Load categories
        categories_buff = io.BytesIO()
        categories_buff.write(serialized_info['categories'])
        categories_buff.seek(0)
        categories = pickle.load(categories_buff)

        # Load model config
        model_config_buff = io.BytesIO()
        model_config_buff.write(serialized_info['model_config'])
        model_config_buff.seek(0)
        model_config = pickle.load(model_config_buff)

        # Load transformations
        transforms_buff = io.BytesIO()
        transforms_buff.write(serialized_info['transformations'])
        transforms_buff.seek(0)
        input_transforms, output_transforms, dataframe_transforms = Model._load_transformations(transforms_buff)

        # Create Model object with the loaded information and as inference mode
        model = class_constructor(schema=schema,
                                  categories=categories,
                                  model_config=model_config,
                                  dataframe_transforms=dataframe_transforms,
                                  input_transforms=input_transforms,
                                  output_transforms=output_transforms,
                                  inference_mode=True)

        # Get the model information
        model_buff = io.BytesIO()
        model_buff.write(serialized_info['model'])
        model_buff.seek(0)
        model_info = class_constructor.load_model(input_file=model_buff,
                                                  schema=schema,
                                                  input_transforms=input_transforms,
                                                  output_transforms=output_transforms,
                                                  dataframe_transforms=dataframe_transforms)
        # Set the infor with setattr
        for k, v in model_info.items():
            setattr(model, k, v)

        # Return the model
        return model

    @staticmethod
    def _load_from_bytes(byte_array: bytes, cls):
        """
        Aux function for loading an object from a given bytearray creating a buffer
        Args:
            byte_array (bytes): the bytes read from pickle to be loaded with the given class
            cls: a class object where the .load class method will be taking for loading the object

        Returns:
            cls object after calling 'load' method
        """
        buff = io.BytesIO()
        buff.write(byte_array)
        buff.seek(0)
        return cls.load(buff)

    @classmethod
    def _load_transformations(
            cls, input_file: Union[str, IO]) -> Tuple[ElementTransforms, ElementTransforms, DataFrameTransforms]:
        """
        Loads all transformations (input, output and DataFrame) calling all 'load' methods of each transformation
        If the given input file is string, it will be the path from where read the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) from where read the object
        Args:
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read object

        Returns:
            Tuple with (input_transforms, output_transforms, dataframe_transforms)
        """
        # If is string, open the file and read the object (serialized with pickle)
        if isinstance(input_file, str):
            with open(input_file, 'rb') as f:
                serialized_transformations = pickle.load(f)
        else:
            # If is not a string, read the object there
            serialized_transformations = pickle.load(input_file)

        # The transformations are also serialized, we have to load them
        transforms = (Model._load_from_bytes(serialized_transformations['input_transforms'], ElementTransforms),
                      Model._load_from_bytes(serialized_transformations['output_transforms'], ElementTransforms),
                      Model._load_from_bytes(serialized_transformations['dataframe_transforms'], DataFrameTransforms))
        return transforms

    @classmethod
    @abstractmethod
    def load_model(cls, input_file: Union[str, IO], schema: Schema, input_transforms: ElementTransforms,
                   output_transforms: ElementTransforms, dataframe_transforms: DataFrameTransforms) -> Dict:
        """
        Abstract class method that is used for creating and loading the model from the given saved file
        Args:
            schema (Schema): schema used for training the model
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read object
            input_transforms (ElementTransforms): input transforms already load that mau be needed for creating model
            output_transforms (ElementTransforms): output transforms already load that mau be needed for creating model
            dataframe_transforms (DataFrameTransforms): dataframe transforms already load that mau be needed for
                                                creating model

        Returns:
            Dict with all key/value pairs that are needed for making predictions
            The load function will take these items and will set them to the model (with setattr)
        """
        raise NotImplementedError()
