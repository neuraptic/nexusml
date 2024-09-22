from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import importlib
import io
import pickle
from typing import Dict, IO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nexusml.engine.exceptions import DataError
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import MLProblemType


@dataclass
class TransformOutputInfo:
    """
    DataClass for storing information about the result of a transformation
    """
    # Output type (float, categorical, etc.)
    output_type: str
    # How many output features have (for example in OneHotEncoding we will have 1 per category)
    num_features: int
    # In some cases, we have a list of possible values that are used by the model
    choices: Optional[List] = None
    # A counter that indicates how many times is repeated each choice. For example, class count for cost-sensitive
    choice_counter: Optional[Dict] = None
    # Mean and Std
    stats: Optional[Tuple[float, float]] = None
    # The problem-type. For the same output type we can have different problem type.
    output_problem_type: Optional[MLProblemType] = None


class Transform(ABC):
    """
    Abstract Transform class that follows the scikit-learn format with the fit, transform and fit_transform methods
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        Args:
            **kwargs:
        """
        self.kwargs = kwargs
        # For default, set the transformation in training mode. It is useful if we want different behaviors for
        # training and testing
        self.training = True

    @abstractmethod
    def fit(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        Fits data to create the transformation
        Args:
            x: data to fit the transformation

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform the given data
        Args:
            x: data to be transformed

        Returns:
            transformed data
        """
        raise NotImplementedError()

    def fit_transform(self, x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Method that first fit the transformation with the given data and then transforms it
        Args:
            x: data to fit transformation and to be transformed

        Returns:
            transformed data
        """
        return self.transform(self.fit(x))

    def train(self) -> None:
        """
        Set in training mode
        """
        self.training = True

    def eval(self) -> None:
        """
        Set on eval mode
        """
        self.training = False

    def save(self, output_file: Union[str, IO]):
        """
        Serializes self object with pickle into the given output file
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store self object

        Returns:

        """
        # If is string, open the file and write the object (serialized with pickle)
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                pickle.dump(self, f)
        else:
            # If is not a string, write the object there
            pickle.dump(self, output_file)

    @classmethod
    def load(cls, input_file: Union[str, IO]):
        """
        Creates a Transform object (same class as cls) lading it from a given input file
        If the given input file is string, it will be the path from where read the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) from where read the object
        Args:
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read self object

        Returns:
            Transform object loaded from the given input file
        """
        # If is string, open the file and read the object (serialized with pickle)
        if isinstance(input_file, str):
            with open(input_file, 'rb') as f:
                self = pickle.load(f)
        else:
            # If is not a string, read the object there
            self = pickle.load(input_file)

        # Return loaded object
        return self


class IndividualTransform(Transform, ABC):
    """
    Abstract transformation for single elements (single DataFrame column for example)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Abstract method that returns the transformation output info
        Returns:
            TransformOutputInfo object with the output info
        """
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(self, x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply inverse transform (for example, back to original scale after MinMaxScale)
        Args:
            x: transformed data to be recovered

        Returns:
            Given transformed data but transformed again to original
        """
        raise NotImplementedError()


class DataFrameTransform(Transform, ABC):
    """
    Abstract transformation for multimple elements (all DataFrame columns for example)
    """

    def __init__(self, schema: Schema, categories: Categories = None, **kwargs):
        """
        Default constructor where the task definition is given (schema and categories)
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            **kwargs:
        """
        super().__init__(**kwargs)
        # Store params
        self.schema = schema
        self.categories = categories
        self.input_features = None
        self.output_features = None


class ElementTransforms(object):
    """
    Class that groups the IndividualTransforms for each element
    """

    def __init__(self, element_transform_map: Dict[str, IndividualTransform]):
        """
        Default constructor
        Args:
            element_transform_map: Dict that maps each column (id) with the transformation to be applied
        """
        self.element_transform_map = element_transform_map
        # We also store witch columns are fitted to be transformed (maybe not all elements are used (feature selection))
        self.fit_columns = []

    def get_transform(self, k: str) -> IndividualTransform:
        """
        Method that returns the transformation given the key
        Args:
            k (str): transformation key (id) to be returned

        Returns:
            IndividualTransform of the given key
        """
        return self.element_transform_map[k]

    @staticmethod
    def _get_class_constructor(module_and_class: str):
        """
        Static method that create a class constructor (or a function caller) given the string path to it
        Args:
            module_and_class (str): with the module and class name (for example, sklearn.linear_model.LinearRegression)

        Returns:
            Constructor of the given path (for example, from "sklearn.linear_model.LinearRegression" it returns
            LinearRegression constructor ready to be called)
        """
        # Split by '.'
        module_and_class = module_and_class.split('.')
        # All but last part is the module (joined by '.')
        module = '.'.join(module_and_class[:-1])
        # The last part is the class name
        class_name = module_and_class[-1]
        # Import module
        module = importlib.import_module(module)
        # Get class from module and return
        return getattr(module, class_name)

    @staticmethod
    def _parse_transforms(transforms_dict: Dict) -> Dict[str, Tuple]:
        """
        Static method that builds transformations given the configuration dict
        Args:
            transforms_dict (dict): dict where we have the element id or element type as key and the transformation
                                    config as value. The config is also a Dict with 'class' key with the transformation
                                    module and class, and 'args' key that is another dict with the arguments

        Returns:
            Dict for each type or element id where the value is a tuple with the constructor (ready to be called) and
            the arguments for that constructor
        """
        # Create empty dict
        transforms_by_key = {}
        # For each element in the input dict
        for k, transform_conf in transforms_dict.items():
            # Get class constructor
            class_constructor = ElementTransforms._get_class_constructor(transform_conf['class'])
            # Get arguments
            args = transform_conf['args']
            # If there are no arguments, set as empty dict
            if args is None:
                args = {}
            # If the arguments is not a dict, create one (so it can be used as **args)
            if not isinstance(args, dict):
                args = {'args': args}
            # Store transformation and args
            transforms_by_key[k] = (class_constructor, args)
        return transforms_by_key

    @classmethod
    def create_from_config(cls, config: Dict, input_or_outputs: List[Dict], categories: Categories):
        """
        Class method (Factory method) that create the transformations form 'transforms' section of the config file
        Args:
            config (Dict): all information under 'transform' section of config dict
            input_or_outputs (List[Dict]): list of dict with all the information about inputs or outputs
                                            (for element specific transform)
            categories (Categories): the possible choices for all categorical elements

        Returns:
            ElementTransforms object with the parsed transformations
        """
        # First parse all global transformations (by type) if any
        if config['global'] is None:
            global_transforms = {}
        else:
            global_transforms = ElementTransforms._parse_transforms(config['global'])

        # Parse all specific transformations (by id) if any
        if config['specific'] is None:
            specific_transforms = {}
        else:
            specific_transforms = ElementTransforms._parse_transforms(config['specific'])

        # To store transformation map
        element_transform_map = {}
        # For each element
        for i in input_or_outputs:
            # If we have a specific transformation for that element, get it
            if i['name'] in specific_transforms:
                tfm, args = specific_transforms[i['name']]
            else:
                # Otherwise, get by type
                tfm, args = global_transforms[i['type']]
            # If the type is categorical, add the categories of that element to arguments
            if i['type'] == 'category':
                args['choices'] = categories.get_categories(i['name'])
            # Create transformation and store in map
            element_transform_map[i['name']] = tfm(**args)

        # Return an object of this class (ElementTransform) with the parsed transformations
        return cls(element_transform_map=element_transform_map)

    def fit(self, df: pd.DataFrame):
        """
        Method that fits all transformations given the data (pd.DataFrame)
        Args:
            df (pd.DataFrame): pd.DataFrame with the data to be fit

        Returns:

        """
        # For each transformation
        for k, v in self.element_transform_map.items():
            # Only if the id is in DataFrame (could be removed by 'feature selection')
            if k in df.columns:
                # Fit the transformation
                v.fit(df[k].to_numpy())
                # Add the element (and column) to fitted columns
                self.fit_columns.append(k)

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Method that transform the given data
        Args:
            df (pd.DataFrame): dataa to be transformed

        Returns:
            Dict[str, np.ndarray]: dict where the key is the element id and the value is a numpy array with the
                                transformed data
        """
        # Assert that all fitted columns are present in DataFrame to be transformed
        if not np.isin(self.fit_columns, df.columns).all():
            raise DataError('Missing some elements on the data to be transformed')
        # For each element, transform the data and return it
        return {k: self.element_transform_map[k].transform(df[k].to_numpy()) for k in self.fit_columns}

    def inverse_transform(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Method that applies the inverse transform of the given data
        Args:
            x (Dict): Dict with the data to be transformed where the key is the element id and the value the data

        Returns:
            Dict with transformed data following the input format ('element_id' => 'data')
        """
        # Assert that all fitted columns are present in the dictionary
        if not np.isin(self.fit_columns, list(x.keys())).all():
            raise DataError('Missing some elements on the data to be transformed (inverse)')
        # Apply inverse transform for each
        return {k: self.element_transform_map[k].inverse_transform(x[k]) for k in self.fit_columns}

    def train(self) -> None:
        """
        Set each transformation on train mode
        :return:
        """
        for k, v in self.element_transform_map.items():
            v.train()

    def eval(self) -> None:
        """
        Set each transformation on eval mode
        :return:
        """
        for k, v in self.element_transform_map.items():
            v.eval()

    def save(self, output_file: Union[str, IO]):
        """
        Serializes all transformations calling all 'save' methods of each transformation
        Also, it saves the fitted columns list
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        # Get the serialization of each transformation as byte array
        serialized_transformations = {}
        for k, v in self.element_transform_map.items():
            buff = io.BytesIO()
            v.save(buff)
            buff.seek(0)
            serialized_transformations[k] = buff.read()

        # Create a dictionary that contains the transformations for one side, and the fitted columns for other
        store_dict = {'transformations': serialized_transformations, 'fit_columns': self.fit_columns}

        # Serialize and save the dict
        # If the given output file is a string, open the file and write the object (serialized with pickle)
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                pickle.dump(store_dict, f)
        else:
            # If is not a string, write the object there
            pickle.dump(store_dict, output_file)

    @classmethod
    def load(cls, input_file: Union[str, IO]):
        """
        Creates a ElementTransforms object (same class as cls) lading it from a given input file
        If the given input file is string, it will be the path from where read the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) from where read the object
        Args:
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read object

        Returns:
            ElementTransforms object loaded from the given input file
        """
        # If is string, open the file and read the object (serialized with pickle)
        if isinstance(input_file, str):
            with open(input_file, 'rb') as f:
                stored_dict = pickle.load(f)
        else:
            # If is not a string, read the object there
            stored_dict = pickle.load(input_file)

        # In the stored dict we have the transformations and the fitted columns
        transformations = stored_dict['transformations']
        fit_columns = stored_dict['fit_columns']

        # The transformations are also serialized, we have to load them
        element_transform_map = {}
        for k, v in transformations.items():
            buff = io.BytesIO()
            buff.write(v)
            buff.seek(0)
            element_transform_map[k] = Transform.load(buff)

        # Create a ElementTransform object with loaded data
        self = ElementTransforms(element_transform_map=element_transform_map)
        # Set the fit columns
        self.fit_columns = fit_columns

        # Return loaded object
        return self


class DataFrameTransforms(object):
    """
    Class that groups all the DataFrame transform to be applied
    """

    def __init__(self, transforms: List[DataFrameTransform]):
        """
        Default constructor
        Args:
            transforms (List): list with the DataFrameTransforms to be applied
        """
        self.transforms = transforms

    @staticmethod
    def _get_class_constructor(module_and_class: str):
        """
        Static method that create a class constructor (or a function caller) given the string path to it
        Args:
            module_and_class (str): with the module and class name (for example, sklearn.linear_model.LinearRegression)

        Returns:
            Constructor of the given path (for example, from "sklearn.linear_model.LinearRegression" it returns
            LinearRegression constructor ready to be called)
        """
        # Split by '.'
        module_and_class = module_and_class.split('.')
        # All but last part is the module (joined by '.')
        module = '.'.join(module_and_class[:-1])
        # The last part is the class name
        class_name = module_and_class[-1]
        # Import module
        module = importlib.import_module(module)
        # Get class from module and return
        return getattr(module, class_name)

    @staticmethod
    def _parse_transforms(schema: Schema, categories: Categories, transform_conf: Dict) -> DataFrameTransform:
        """
        Static method that parses a DataFrameTransform given its config
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible choices for each categorical elements
            transform_conf (Dict): the config with 'class' key with the transformation module and class,
                                    and 'args' key that is another dict with the arguments of the transformation

        Returns:
            DataFrameTransform already built
        """
        # Get class constructor
        class_constructor = DataFrameTransforms._get_class_constructor(transform_conf['class'])
        # Get arguments
        args = transform_conf['args']
        # If there are no arguments, set as empty dict
        if args is None:
            args = {}
        # If the arguments is not a dict, create one (so it can be used as **args)
        if not isinstance(args, dict):
            args = {'args': args}
        # Build the transformation and return
        return class_constructor(schema=schema, categories=categories, **args)

    @classmethod
    def create_from_config(cls, schema: Schema, categories: Categories, config: List[Dict]):
        """
        Class method (factory method) that build the DataFrameTransforms given the configuration
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible choices for each categorical elements
            config (List[Dict]): list with the configuration of each transform

        Returns:
            DataFrameTransforms object with the parsed transformation
        """
        # Applies cls._parse_transform function to each transformation config to build all transformations
        transforms = list(
            map(lambda x: cls._parse_transforms(schema=schema, categories=categories, transform_conf=x), config))
        # Creates a DataFrameTransforms object with the parsed transforms and returns it
        return cls(transforms=transforms)

    def fit(self, df: pd.DataFrame):
        """
        Method to fit all transformation given data
        Args:
            df (pd.DataFrame): data to be fit

        Returns:

        """
        # For each transformation
        for tfm in self.transforms:
            # Fit it
            tfm.fit(df)
            # And transform so the next transform is fit with the transformed data
            df = tfm.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that applies the 'transform' method of all transformation over the given data
        Args:
            df (pd.DataFrame): data to be transformed

        Returns:
            pd.DataFrame with the transformed data
        """
        # For each transform
        for tfm in self.transforms:
            # Apply the transformation and get the transformed DataFrame for the next transformation
            df = tfm.transform(df)
        return df

    def train(self) -> None:
        """
        Set on train mode
        :return:
        """
        for tfm in self.transforms:
            tfm.train()

    def eval(self) -> None:
        """
        Set on eval mode
        :return:
        """
        for tfm in self.transforms:
            tfm.eval()

    def save(self, output_file: Union[str, IO]):
        """
        Serializes all transformations calling all 'save' methods of each transformation
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        # Get the serialization of each transformation as byte array
        serialized_transformations = []
        for i in self.transforms:
            buff = io.BytesIO()
            i.save(buff)
            buff.seek(0)
            serialized_transformations.append(buff.read())

        # Serialize and save the transforms list
        # If the given output file is a string, open the file and write the object (serialized with pickle)
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                pickle.dump(serialized_transformations, f)
        else:
            # If is not a string, write the object there
            pickle.dump(serialized_transformations, output_file)

    @classmethod
    def load(cls, input_file: Union[str, IO]):
        """
        Creates a DataFrameTransforms object (same class as cls) lading it from a given input file
        If the given input file is string, it will be the path from where read the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) from where read the object
        Args:
            input_file (Union[str, IO]): input file path or input buffer/descriptor from where read object

        Returns:
            DataFrameTransforms object loaded from the given input file
        """
        # If is string, open the file and read the object (serialized with pickle)
        if isinstance(input_file, str):
            with open(input_file, 'rb') as f:
                serialized_transformations = pickle.load(f)
        else:
            # If is not a string, read the object there
            serialized_transformations = pickle.load(input_file)

        # The transformations are also serialized, we have to load them
        transforms = []
        for i in serialized_transformations:
            buff = io.BytesIO()
            buff.write(i)
            buff.seek(0)
            transforms.append(Transform.load(buff))

        # Create a DataFrameTransforms object with loaded data
        self = DataFrameTransforms(transforms=transforms)
        # Return loaded object
        return self


def get_element_transforms(schema: Schema, categories: Categories,
                           transforms_config: Dict) -> Tuple[ElementTransforms, ElementTransforms]:
    """
    Function that builds all element transformation given config
    Args:
        schema (Schema): the task schema
        categories (Categories): the possible choices for each categorical elements
        transforms_config (Dict): dict with the config of 'transform' section with the inputs and output
                                    transformations config

    Returns:
        Tuple[ElementTransforms, ElementTransforms] with the transformations of input and output elements
    """
    # Get input transformations
    input_transforms = ElementTransforms.create_from_config(config=transforms_config['input_transforms'],
                                                            input_or_outputs=schema.inputs,
                                                            categories=categories)
    # Get output transformations
    output_transforms = ElementTransforms.create_from_config(config=transforms_config['output_transforms'],
                                                             input_or_outputs=schema.outputs,
                                                             categories=categories)
    # Return input and output transformations
    return input_transforms, output_transforms


def get_dataframe_transforms(schema: Schema, categories: Categories,
                             transforms_config: List[Dict]) -> DataFrameTransforms:
    """
    Function that build all DataFrame transformations
    Args:
        schema (Schema): the task schema
        categories (Categories): the possible choices for each categorical elements
        transforms_config (List[Dict]): list with the config of each DataFrameTransform

    Returns:
        DataFrameTransforms with all DataFrameTransform built
    """
    return DataFrameTransforms.create_from_config(schema=schema, categories=categories, config=transforms_config)
