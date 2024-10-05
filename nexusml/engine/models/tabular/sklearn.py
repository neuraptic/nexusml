import copy
import functools
import importlib
import pickle
from typing import Callable, Dict, IO, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.transforms.base import IndividualTransform
from nexusml.engine.data.transforms.sklearn import LabelEncoderTransform
from nexusml.engine.data.utils import np_column_cat
from nexusml.engine.data.utils import predictions_to_example_format
from nexusml.engine.exceptions import ConfigFileError
from nexusml.engine.models.base import Model
from nexusml.engine.models.base import TrainingOutputInfo
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import TaskType


class SKLearnModel(Model):
    """
    Model class specialization for SKLearn like tabular models
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
        # Call super class constructor
        super().__init__(schema=schema,
                         categories=categories,
                         model_config=model_config,
                         dataframe_transforms=dataframe_transforms,
                         input_transforms=input_transforms,
                         output_transforms=output_transforms,
                         inference_mode=inference_mode)
        # Create empty dict to store the model for each output
        self.models = {}

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
        # Create the models (one for each output) using the previous built function
        self.models = setup_function(schema=self.schema,
                                     input_transforms=self.input_transforms,
                                     output_transforms=self.output_transforms,
                                     **self.model_config['setup_args'])

    def fit(self,
            train_data: Union[pd.DataFrame, dict, List[dict]],
            valid_data: Union[pd.DataFrame, dict, List[dict]] = None,
            **kwargs) -> TrainingOutputInfo:
        """
        Function called to train the model
        Args:
            train_data (Union[pd.DataFrame, dict, List[dict]]): train data that could be a DataFrame, a single example
                                                            as dict, or a list of dict examples
            valid_data (Union[pd.DataFrame, dict, List[dict]]): validation data that could be a DataFrame, a
                                                                single example as dict, or a list of dict examples
            **kwargs: other args

        Returns:
            TrainingOutputInfo filled with the searched best parameters (only if GridSearch is used)
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

        # Fit the transformed data to both, input and output transforms. And put them in train mode
        self.input_transforms.fit(train_data)
        self.input_transforms.train()
        self.output_transforms.fit(train_data)
        self.output_transforms.train()

        # Set up the models
        self._setup_model()

        # Transform input and output data
        transformed_input_data = self.input_transforms.transform(train_data)
        transformed_output_data = self.output_transforms.transform(train_data)

        # To store best params in case GridSearch is applied
        models_best_params = {}

        # Initialize x as None to concatenate all input data
        x = None
        # For each input
        for i in self.schema.inputs:
            # If the data is used, concatenate it by column to the previous ones
            if i['name'] in transformed_input_data:
                x = np_column_cat(x, transformed_input_data[i['name']])
        # For each output (and hence, for each model)
        for i in self.schema.outputs:
            # Fit the model with the input data and the target of the current output
            self.models[i['name']].fit(x, transformed_output_data[i['name']])
            # If a parameter search is made, store the best params, so it can be returned
            if isinstance(self.models[i['name']], BaseSearchCV):
                models_best_params[i['name']] = self.models[i['name']].best_params_

        # Return the best params of each model if the search is made
        return TrainingOutputInfo(params=models_best_params)

    def predict(self,
                data: Union[pd.DataFrame, dict, List[dict]],
                split_predictions_by_output: bool = False,
                **kwargs) -> Union[Dict, List]:
        """
        Function called to make predictions on the given data
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
        if isinstance(data, dict) or isinstance(data, list):
            data = Model.examples_to_dataframe(data)
        # Put transformation on eval mode
        self.dataframe_transforms.eval()
        self.input_transforms.eval()
        self.output_transforms.eval()
        # Transform input data
        data = self.dataframe_transforms.transform(data)
        transformed_input_data = self.input_transforms.transform(data)
        # Concatenate all input data like is done on fit
        x = None
        for i in self.schema.inputs:
            if i['name'] in transformed_input_data:
                x = np_column_cat(x, transformed_input_data[i['name']])

        # To store predictions
        predictions = {}
        # For each output
        for i in self.schema.outputs:
            # If the type is float, call 'predict'
            # If the type is categorical, call 'predict_proba' to get the score for each class
            if i['type'] == 'float':
                predictions[i['name']] = self.models[i['name']].predict(x)
            elif i['type'] == 'category':
                predictions[i['name']] = self.models[i['name']].predict_proba(x)
            else:
                raise ValueError(f'Output "{i["name"]} is of type "{i["type"]}", that is not recognized')

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
        # To store all configs
        configs = []

        # Transforms section
        transforms_section = {
            'transforms': {
                'input_transforms': SKLearnModel._get_default_input_transforms_config(),
                'output_transforms': SKLearnModel._get_default_output_transforms_config()
            },
            'dataframe_transforms': SKLearnModel._get_default_dataframe_transforms_config()
        }

        # SVM
        svm_config = copy.deepcopy(transforms_section)
        svm_config['model'] = SKLearnModel._get_default_svm_config()
        configs.append(svm_config)

        # Knn
        knn_config = copy.deepcopy(transforms_section)
        knn_config['model'] = SKLearnModel._get_default_knn_config()
        configs.append(knn_config)

        # Random Forest
        random_forest_config = copy.deepcopy(transforms_section)
        random_forest_config['model'] = SKLearnModel._get_default_random_forest_config()
        configs.append(random_forest_config)

        # Gradient Boosting Tree
        gradient_boosting_config = copy.deepcopy(transforms_section)
        gradient_boosting_config['model'] = SKLearnModel._get_default_gradient_boosting_config()
        configs.append(gradient_boosting_config)

        return configs

    @staticmethod
    def _get_default_dataframe_transforms_config() -> List[dict]:
        """
        Returns the default configuration for DataFrame transformations applied to the data.

        Returns:
            List[dict]: A list of transformations applied to the DataFrame.
        """
        return [{
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

    @staticmethod
    def _get_default_input_transforms_config() -> dict:
        """
        Returns the default input transformation configuration applied globally and for specific input types.

        Returns:
            dict: Input transformations applied globally and specifically for each input type.
        """
        return {
            'global': {
                'float': {
                    'class': 'nexusml.engine.data.transforms.sklearn.StandardScalerTransform',
                    'args': None
                },
                'category': {
                    'class': 'nexusml.engine.data.transforms.sklearn.OneHotEncoderTransform',
                    'args': None
                },
                'text': {
                    'class': 'nexusml.engine.data.transforms.sklearn.TfIdfTransform',
                    'args': None
                }
            },
            'specific': None
        }

    @staticmethod
    def _get_default_output_transforms_config() -> dict:
        """
        Returns the default output transformation configuration applied globally for each output type.

        Returns:
            dict: Output transformations applied globally for each output type.
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

    @staticmethod
    def _get_default_svm_config() -> dict:
        """
        Returns the default configuration for an SVM model.

        Returns:
            dict: Configuration for SVM-based model.
        """
        return {
            'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
            'args': {
                'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
                'setup_args': {
                    'classification_model_class': 'sklearn.svm.SVC',
                    'classification_model_args': {
                        'probability': True
                    },
                    'regression_model_class': 'sklearn.svm.SVR'
                }
            }
        }

    @staticmethod
    def _get_default_knn_config() -> dict:
        """
        Returns the default configuration for a k-NN model.

        Returns:
            dict: Configuration for k-NN model.
        """
        return {
            'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
            'args': {
                'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
                'setup_args': {
                    'classification_model_class': 'sklearn.neighbors.KNeighborsClassifier',
                    'regression_model_class': 'sklearn.neighbors.KNeighborsRegressor'
                }
            }
        }

    @staticmethod
    def _get_default_random_forest_config() -> dict:
        """
        Returns the default configuration for a Random Forest model.

        Returns:
            dict: Configuration for Random Forest model.
        """
        return {
            'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
            'args': {
                'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
                'setup_args': {
                    'classification_model_class': 'sklearn.ensemble.RandomForestClassifier',
                    'regression_model_class': 'sklearn.ensemble.RandomForestRegressor'
                }
            }
        }

    @staticmethod
    def _get_default_gradient_boosting_config() -> dict:
        """
        Returns the default configuration for a Gradient Boosting Tree model.

        Returns:
            dict: Configuration for Gradient Boosting Tree model.
        """
        return {
            'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
            'args': {
                'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
                'setup_args': {
                    'classification_model_class': 'sklearn.ensemble.GradientBoostingClassifier',
                    'classification_model_args': {
                        'verbose': 0
                    },
                    'regression_model_class': 'sklearn.ensemble.GradientBoostingRegressor',
                    'regression_model_args': {
                        'verbose': 0
                    },
                }
            }
        }

    def summary(self) -> Optional[str]:
        """
        Returns the summary of the trained model. It is only implemented for DecisionTree models and returns
        the generated rules. For the other models, the name of the model class is used

        Returns:
            string that will contain the summary of each model. For DecisionTree it will contain the rules.
            For other models, just the model name
        """
        models_summary = ''
        for o_id, model in self.models.items():
            models_summary += f'OUTPUT: {o_id}\n'
            models_summary += f'MODEL: {str(model)}\n'
            if isinstance(model, DecisionTreeClassifier) or isinstance(model, DecisionTreeRegressor):
                models_summary += 'RULES: \n'
                rules = SKLearnModel._get_rules_from_model(tree_model=model,
                                                           input_transforms=self.input_transforms,
                                                           output_transform=self.output_transforms.get_transform(o_id))
                models_summary += '\n'.join(rules)
                models_summary += '\n'

            models_summary += f"{'-' * 25}\n\n"

        return models_summary

    def save_model(self, output_file: Union[str, IO]):
        """
        Method that saves sklearn models serialized in the given output_file
        This method expect to store as a dict all the information that will be needed to make predictions
        So in this case, we store a dict with the 'models' dict: {'models': self.models}
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        # Things to be saved
        to_store = {'models': self.models}
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
        return model_info

    # Note: This pylint directive is disabled because `sklearn` is not installed in GitHub Actions environments.
    # pylint: disable=c-extension-no-member
    @staticmethod
    def _get_rules_from_model(tree_model: Union[DecisionTreeRegressor, DecisionTreeClassifier],
                              input_transforms: ElementTransforms, output_transform: IndividualTransform) -> List:
        """
        Function that extracts the created rules using a DecisionTree model
        Note: this function is get from the internet
        Args:
            tree_model (Union[DecisionTreeRegressor, DecisionTreeClassifier]): tree model from where get rules
            input_transforms (ElementTransforms): transformations applied to input elements
            output_transform (ElementTransforms): transformations applied to output elements

        Returns:
            List with the rules extracted from tree model
        """
        feature_names = []
        for k, v in input_transforms.element_transform_map.items():
            output_info = v.get_transform_output_info()
            if output_info.num_features == 1:
                feature_names.append(k)
            else:
                for i in range(1, output_info.num_features):
                    feature_names.append(f'{k}_{i}')

        class_names = None
        if isinstance(tree_model, DecisionTreeClassifier):
            assert isinstance(output_transform, LabelEncoderTransform)
            class_names = output_transform.sklearn_transform.classes_

        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!' for i in tree_model.tree_.feature
        ]

        paths = []
        path = []

        def _recurse(node, path, paths):

            if tree_model.tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_model.tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f'({name} <= {np.round(threshold, 3)})']
                _recurse(tree_model.tree_.children_left[node], p1, paths)
                p2 += [f'({name} > {np.round(threshold, 3)})']
                _recurse(tree_model.tree_.children_right[node], p2, paths)
            else:
                path += [(tree_model.tree_.value[node], tree_model.tree_.n_node_samples[node])]
                paths += [path]

        _recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = 'if '

            for p in path[:-1]:
                if rule != 'if ':
                    rule += ' and '
                rule += str(p)
            rule += ' then '
            if class_names is None:
                rule += 'response: ' + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f'class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)'
            rule += f' | based on {path[-1][1]:,} samples'
            rules += [rule]

        return rules

    # pylint: enable=c-extension-no-member


def _from_class_name_to_constructor(module_and_class: str) -> Callable:
    """
    Function that returns a callable object creating it from the given module and class
    Args:
        module_and_class (str): the string that contains the module and class to be converted to callable

    Returns:
        Constructor of the given class after import it from the given module
    """
    # Split by '.'
    module_and_class = module_and_class.split('.')
    # All but the last part is the module
    module = '.'.join(module_and_class[:-1])
    # The las part is the class name
    class_name = module_and_class[-1]
    # Import module
    module = importlib.import_module(module)
    # Get class constructor from module
    return getattr(module, class_name)


def _setup_sklearn_from_config(schema: Schema,
                               input_transforms: ElementTransforms,
                               output_transforms: ElementTransforms,
                               classification_model_class: str,
                               regression_model_class: str,
                               classification_model_args: Dict = None,
                               regression_model_args: Dict = None) -> Dict:
    """
    Set up function for creating SKlearn like models
    Args:
        schema (Schema): the task schema
        input_transforms (ElementTransforms): transformations that are applied on each input element
        output_transforms (ElementTransforms): transformations that are applied on each output element
        classification_model_class (str): string indicating whole path to the class that be used for classification
        regression_model_class (str): string indicating whole path to the class that be used for regression
        classification_model_args (Dict): extra args used for creating the classification models
        regression_model_args (Dict): extra args used for creating the regression models

    Returns:
        Dict with one element per output containing the model to be used
    """
    # Get regression model callable object and set the given arguments
    regression_model = _from_class_name_to_constructor(regression_model_class)
    if regression_model_args is not None:
        regression_model = functools.partial(regression_model, **regression_model_args)

    # Get the classification model callable object and set the given arguments
    classification_model = _from_class_name_to_constructor(classification_model_class)
    if classification_model_args is not None:
        classification_model = functools.partial(classification_model, **classification_model_args)

    # Initialize models as empty dict
    models = {}

    # For each output
    for output in schema.outputs:
        # If type is float, create regression model
        if output['type'] == 'float':
            models[output['name']] = regression_model()
        elif output['type'] == 'category':
            # If type is categorical, create classification model
            models[output['name']] = classification_model()
        else:
            raise ValueError(f'Output "{output["name"]} is of type "{output["type"]}", that is not recognized')

    # Return models
    return models


def _setup_sklearn_grid_search_from_config(schema: Schema,
                                           input_transforms: ElementTransforms,
                                           output_transforms: ElementTransforms,
                                           classification_model_class: str,
                                           regression_model_class: str,
                                           classification_model_params: Union[Dict, List[Dict]],
                                           regression_model_params: Union[Dict, List[Dict]],
                                           n_jobs: int = None,
                                           cv: int = 5) -> Dict:
    """

    Args:
        schema (Schema): the task schema
        input_transforms (ElementTransforms): transformations that are applied on each input element
        output_transforms (ElementTransforms): transformations that are applied on each output element
        classification_model_class (str): string indicating whole path to the class that be used for classification
        regression_model_class (str): string indicating whole path to the class that be used for regression
        classification_model_params (Union[Dict, List[Dict]]): parameters to be searched for the classification model
        regression_model_params (Union[Dict, List[Dict]]): parameters to be searched for the regression model
        n_jobs (int): number if jobs used by GridSearch class for parallel computing
        cv (int): number of folds used on cross-validation for searching the parameters

    Returns:
        Dict with one element per output containing the model to be used
    """
    # Get regression and classification model constructors
    regression_model = _from_class_name_to_constructor(regression_model_class)
    classification_model = _from_class_name_to_constructor(classification_model_class)

    # To store the models
    models = {}

    # For each output
    for output in schema.outputs:
        # If type is float, create a GridSearch model giving the regression model
        if output['type'] == 'float':
            models[output['name']] = GridSearchCV(estimator=regression_model(),
                                                  param_grid=regression_model_params,
                                                  n_jobs=n_jobs,
                                                  cv=cv,
                                                  scoring='neg_mean_squared_error',
                                                  verbose=2)
        elif output['type'] == 'category':
            # If type is float, create a GridSearch model giving the classification model
            models[output['name']] = GridSearchCV(estimator=classification_model(),
                                                  param_grid=classification_model_params,
                                                  n_jobs=n_jobs,
                                                  cv=cv,
                                                  scoring='balanced_accuracy',
                                                  verbose=2)
        else:
            raise ValueError(f'Output "{output["name"]} is of type "{output["type"]}", that is not recognized')

    # Return models
    return models
