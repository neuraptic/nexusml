from abc import ABC
from collections import Counter
import copy
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from nexusml.engine.data.transforms.base import DataFrameTransform
from nexusml.engine.data.transforms.base import IndividualTransform
from nexusml.engine.data.transforms.base import TransformOutputInfo
from nexusml.engine.exceptions import DataError
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import MLProblemType


class SKLearnTransform(IndividualTransform, ABC):
    """
    Abstract transform for all SKLearn based transformation
    """

    def __init__(self, sklearn_transform: BaseEstimator, **kwargs):
        """
        Default constructor
        Args:
            sklearn_transform: SKLearn transform to be applied (with fit, transform and inverse_transform methods)
            **kwargs: other arguments
        """
        super(SKLearnTransform, self).__init__(**kwargs)
        # Store the sklearn transform
        self.sklearn_transform = sklearn_transform


class BasicTransform(SKLearnTransform, ABC):
    """
    Basic SKLearn transform.
    Applies the sklearn transform to the input data without modifying it
    """

    def __init__(self, sklearn_transform: BaseEstimator, **kwargs):
        """
        Default constructor
        Args:
            sklearn_transform: SKLearn transform to be applied (with fit, transform and inverse_transform methods)
            **kwargs: other arguments
        """
        super().__init__(sklearn_transform, **kwargs)

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        # Fit sklearn transform
        self.sklearn_transform.fit(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform given data
        Args:
            x (np.ndarray): data to transform

        Returns:
            np.ndarray with transformed data
        """
        # Transform data with sklearn transform
        return self.sklearn_transform.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply inverse transform to given data
        Args:
            x (np.ndarray): data to transform (inverse)

        Returns:
            np.ndarray after apply inverse transform
        """
        # Apply sklearn inverse transform
        return self.sklearn_transform.inverse_transform(x)


class ExpandDimTransform(SKLearnTransform, ABC):
    """
    Extension of SKLearnTransform but that add the "column" dimension to the input data
    Some SKLearn transformations have the requirement that data has two dimensions: (n_samples, n_features),
    but the input data is always applies to a single column with 1 dimension (n_samples, )
    For this reason, before apply transformation we extend the second dim
    """

    def __init__(self, sklearn_transform: BaseEstimator, **kwargs):
        """
        Default constructor
        Args:
            sklearn_transform: SKLearn transform to be applied (with fit, transform and inverse_transform methods)
            **kwargs: other arguments
        """
        super().__init__(sklearn_transform, **kwargs)

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        # Fit sklearn transform after add second dim
        self.sklearn_transform.fit(x[:, np.newaxis])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform given data
        Args:
            x (np.ndarray): data to transform

        Returns:
            np.ndarray with transformed data
        """
        # Transform data with sklearn transform after add second dim
        x = self.sklearn_transform.transform(x[:, np.newaxis])
        # If transformation returns a single columns, remove it to have the same format as input: (n_samples, )
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        # If transformed data in float64, transform to float32 so PyTorch neural networks can work with it
        if x.dtype == np.float64:
            x = x.astype(np.float32)
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply inverse transform to given data
        Args:
            x (np.ndarray): data to transform (inverse)

        Returns:
            np.ndarray after apply inverse transform
        """
        # Apply sklearn inverse transform. Add second dim if the array is of 1 dimension
        # Then return the single column to have (n_samples, ) shape
        return self.sklearn_transform.inverse_transform(x[:, np.newaxis] if x.ndim == 1 else x)[:, 0]


class StandardScalerTransform(ExpandDimTransform):
    """
    Applies the StandardScaler transform to data
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        Args:
            **kwargs: other arguments
        """
        # Call super with StandardScaler transformation
        super().__init__(StandardScaler(**kwargs), **kwargs)
        # Fill transform_output_info (type float, one feature, has no choices and init stats with None)
        # The output problem type will be REGRESSION if this transform is used as output
        self.transform_output_info = TransformOutputInfo(output_type='float',
                                                         num_features=1,
                                                         choices=None,
                                                         choice_counter=None,
                                                         stats=None,
                                                         output_problem_type=MLProblemType.REGRESSION)

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        # Fits the transform
        super().fit(x)
        # Get the stats
        self.transform_output_info.stats = (self.sklearn_transform.mean_[0], self.sklearn_transform.scale_[0])

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        Returns:
            TransformOutputInfo object filled
        """
        return self.transform_output_info


class MinMaxScalerTransform(ExpandDimTransform):
    """
    Applies the MinMaxScaler transform to data
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        Args:
            **kwargs: other arguments
        """
        # Call super with MinMaxScaler transformation
        super().__init__(MinMaxScaler(**kwargs), **kwargs)
        # Fill transform_output_info (type float, one feature, has no choices and init stats with None)
        # The problem type will be REGRESSION
        self.transform_output_info = TransformOutputInfo(output_type='float',
                                                         num_features=1,
                                                         choices=None,
                                                         choice_counter=None,
                                                         stats=None,
                                                         output_problem_type=MLProblemType.REGRESSION)

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        # Fits the transform
        super().fit(x)
        # Update stats, in this case with the feature range (min and max)
        self.transform_output_info.stats = (self.sklearn_transform.data_min_, self.sklearn_transform.data_max_)

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        Returns:
            TransformOutputInfo object filled
        """
        return self.transform_output_info


class OneHotEncoderTransform(ExpandDimTransform):
    """
    Applies the OneHotEncoder transform to data
    """

    def __init__(self, choices: List, **kwargs):
        """
        Default constructor
        Args:
            choices (List): list with the possible values than can take the element
            **kwargs: other arguments
        """
        # Call super with the OneHotEncoder transform, setting sparse to False and setting the choices
        # Note: `sparse` param is now `sparse_output`
        # pylint: disable-next=unexpected-keyword-arg
        super().__init__(OneHotEncoder(sparse=False, categories=choices, **kwargs), **kwargs)
        # Fill the output info. In this case, type int, one feature per choice, and set counter to None to fill later
        # We should avoid using this transformation for output. The OneHot process is done using cons-function
        # Most of the algorithms receives the label index, and they make this transform, so set it as None
        self.transform_output_info = TransformOutputInfo(output_type='int',
                                                         num_features=len(choices),
                                                         choices=choices,
                                                         choice_counter=None,
                                                         stats=None,
                                                         output_problem_type=None)

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        # Fit transformation
        super(OneHotEncoderTransform, self).fit(x)
        # Count choices (for example, to cost-sensitive learning)
        self.transform_output_info.choice_counter = Counter(x)

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        Returns:
            TransformOutputInfo object filled
        """
        return self.transform_output_info


class LabelEncoderTransform(BasicTransform):
    """
    Applies the LabelEncoder transform to data
    """

    def __init__(self, choices: List, **kwargs):
        """
        Default constructor
        Args:
            choices (List): list with the possible values than can take the element
            **kwargs: other arguments
        """
        # Create LabelEncoder and fit with choices, so we initialize the .classes_ values with choices
        super().__init__(LabelEncoder().fit(choices), **kwargs)
        # Set the transform output info (categorical type, 1 feature, possible values and init counter as None).
        # Problem type will be classification
        # If there are two choices, it will be BinaryClassification
        # If three or more, MultiClassClassification
        if len(choices) == 2:
            problem_type = MLProblemType.BINARY_CLASSIFICATION
        elif len(choices) > 2:
            problem_type = MLProblemType.MULTI_CLASS_CLASSIFICATION
        else:
            # There is only one choice, set is as None because it could be that this transform is applied over
            # input elements (although it has no sense because one input with always same value does not contribute)
            problem_type = None
        self.transform_output_info = TransformOutputInfo(output_type='category',
                                                         num_features=1,
                                                         choices=choices,
                                                         choice_counter=None,
                                                         stats=None,
                                                         output_problem_type=problem_type)
        # Store the fitted classes
        # If the model is fit with fewer classes than possibles, the output of the model will only have values for
        # those classes (the columns will not match). So, we store the classes and then, we can transform
        # the prediction putting 0s on those classes
        self._fitted_classes = None

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        # Assert that classes_ is initialized
        if not hasattr(self.sklearn_transform, 'classes_'):
            raise DataError('Missing \'classes_\' attribute on the \'sklearn_transform\'')
        # Get the counter of values
        self.transform_output_info.choice_counter = Counter(x)
        # Assert that all values in data are in classes
        if not np.isin(list(self.transform_output_info.choice_counter.keys()), self.sklearn_transform.classes_).all():
            raise DataError('Unexpected class value')
        # Store fitted classes
        self._fitted_classes = list(self.transform_output_info.choice_counter.keys())
        # if hasattr(self.sklearn_transform, "classes_"):
        #     assert np.all(np.unique(x) == self.sklearn_transform.classes_)
        # else:
        #     self.sklearn_transform.fit(x)
        # self.transform_output_info.choice_counter = Counter(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return super().transform(x).astype(np.int64)

    def inverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        """
        Apply inverse transform to given data
        This transformation is used to encode classes in classification
        In this case, the given data will be the prediction of the model for this output (score per class)
        We do not want to lose this information, so we return a DataFrame with this values
        setting the column name with the class name
        Args:
            x (np.ndarray): data to transform (inverse)

        Returns:
            DataFrame with the scores (probabilities) of each sample for each class
        """
        # Create a DataFrame with given data (probabilities) setting the class name as column name
        # Expand with 0 the not used classes if the prediction has fewer columns than classes
        if x.shape[1] < len(self.sklearn_transform.classes_):
            prediction = np.zeros((x.shape[0], len(self.sklearn_transform.classes_)), dtype=x.dtype)
            prediction[:, np.isin(self.sklearn_transform.classes_, self._fitted_classes)] = x
        else:
            if x.shape[1] > len(self.sklearn_transform.classes_):
                raise DataError('More number of classes predicted than fitted')
            prediction = x
        return pd.DataFrame(data=prediction, columns=self.sklearn_transform.classes_)

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        Returns:
            TransformOutputInfo object filled
        """
        return self.transform_output_info


class TfIdfTransform(BasicTransform):
    """
    Applies the TfidfVectorizer transform to data
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        Args:
            **kwargs: other arguments
        """
        # Call super with the TfidfVectorizer transform
        super().__init__(TfidfVectorizer(dtype=np.float32, **kwargs), **kwargs)
        # For now set output info as None because we have no information
        self.transform_output_info = None

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        # Call super fit with data, but transforming it to list (requirement of TfidfVectorizer transform)
        super(TfIdfTransform, self).fit(x.tolist())
        # Now we can get the information.
        # The output type is float, and we get the number of output features from get_feature_names_out method
        # This transform has no sense as output transform, so set it as None
        self.transform_output_info = TransformOutputInfo(output_type='float',
                                                         num_features=len(
                                                             self.sklearn_transform.get_feature_names_out()),
                                                         output_problem_type=None)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform given data
        Args:
            x (np.ndarray): data to transform

        Returns:
            np.ndarray with transformed data
        """
        # First convert data to list before transform. This result in sparse matrix so transform
        # it to numpy array and convert to float32 type
        return super(TfIdfTransform, self).transform(x.tolist()).toarray().astype(np.float32)

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        First ensure that transformation is fitted
        Returns:
            TransformOutputInfo object filled
        """
        # If we have no information, raise exception telling that the transformation is not fitted
        if self.transform_output_info is None:
            raise DataError('The transformation is not fitted yet')
        else:
            return self.transform_output_info


class OrdinalEncoderTransform(ExpandDimTransform):
    """
    Applies the OrdinalEncoder transform to data
    """

    def __init__(self, choices: List, **kwargs):
        """
        Default constructor
        Args:
            choices (List): list with the possible values than can take the element
            **kwargs: other arguments
        """
        # Note: the OrdinalEncoder class expect categories for each feature. For this reason,
        # we give the choices as a list (list of, list of choices).
        super().__init__(OrdinalEncoder(dtype='int64', categories=[choices], **kwargs), **kwargs)
        # Setup output info (1 categorical feature with choices)
        # This transform has no sense as output transform (the correct one will be LabelEncoder), so set it as None
        self.transform_output_info = TransformOutputInfo(output_type='category',
                                                         num_features=1,
                                                         choices=choices,
                                                         choice_counter=None,
                                                         stats=None,
                                                         output_problem_type=None)

    def fit(self, x: np.ndarray):
        """
        Fit the sklearn transform
        Args:
            x (np.ndarray): data to transform

        Returns:

        """
        # Fit transform calling super
        super(OrdinalEncoderTransform, self).fit(x)
        # Set counter
        self.transform_output_info.choice_counter = Counter(x)

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        Returns:
            TransformOutputInfo object filled
        """
        return self.transform_output_info


class KBinsDiscretizerTransform(ExpandDimTransform):
    """
    Applies the KBinsDiscretizer transform to data
    """

    def __init__(self, n_bins: int = 5, encode: str = 'ordinal', strategy: str = 'quantile', **kwargs):
        """
        Default constructor
        Args:
            n_bins (int): number of bins
            encode (str): encode type
            strategy (str): strategy used to define the widths of the bins.
            **kwargs:
        """
        # Note: for now only support 'ordinal' encoder, so it returns a single feature. if 'one-hot' is selected
        # it will return one feature per bin
        # ToDo: support other encode types
        # Call super with the created transformation
        super().__init__(KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, **kwargs), **kwargs)
        if encode != 'ordinal':
            raise DataError("'KBinsDiscretizerTransform' only supports 'ordinal' encode type")
        # Fill output: 1 categorical features with one choice per bin
        # Although this transformation changes a continuous attribute to categorical, after prediction
        # the 'inverse_transform' will be applied getting again a float type output
        # So, the problem type is regression
        self.transform_output_info = TransformOutputInfo(output_type='category',
                                                         num_features=1,
                                                         choices=list(range(n_bins)),
                                                         choice_counter=None,
                                                         stats=None,
                                                         output_problem_type=MLProblemType.REGRESSION)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform given data
        Args:
            x (np.ndarray): data to transform

        Returns:
            np.ndarray with transformed data
        """
        # Return the transformed data as int64 (PyTorch compatibility, 'long' type for embeddings)
        # ToDo: support other encode types (same as in __init__)
        return super(KBinsDiscretizerTransform, self).transform(x).astype(np.int64)

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        Returns:
            TransformOutputInfo object filled
        """
        return self.transform_output_info


class VarianceFeatureSelection(DataFrameTransform):
    """
    DataFrame transform that discards those features with low variance
    """

    def __init__(self, schema: Schema, categories: Categories = None, threshold: float = 0.95 * (1 - 0.95), **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            threshold (float): variance threshold to select features
            **kwargs: other arguments
        """
        super().__init__(schema=schema, categories=categories, **kwargs)
        # Store threshold
        self.threshold = threshold
        # Create transformation
        self.fe_transform = VarianceThreshold(threshold=threshold)

    def fit(self, x: pd.DataFrame):
        """
        Fits data to create the transformation
        Args:
            x (DataFrame): DataFrame with data to fit

        Returns:

        """
        # Get which features come in DataFrame
        self.input_features = x.columns.tolist()
        # Fit transformation with numpy data
        self.fe_transform.fit(x.to_numpy())
        # Do not remove output columns, so set a high value to output variances
        is_output = np.isin(x.columns, list(map(lambda y: y['name'], self.schema.outputs)))
        self.fe_transform.variances_[is_output] = self.threshold + 1
        # Get the columns that will be used (those whose variance is bigger than threshold)
        self.output_features = x.columns[self.fe_transform.variances_ > self.threshold].tolist()

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given data
        Args:
            x (DataFrame): DataFrame with data to be transformed

        Returns:
            DataFrame with transformed data
        """
        # Return a copy of the input DataFrame, but with the selected columns
        if not self.training:
            # We are evaluating, se we remove the output columns ig they are not given
            # Otherwise, an exception will be raised for trying to access a non-existent column
            output_features = copy.deepcopy(self.output_features)
            for i in self.schema.outputs:
                if i['name'] not in x.columns:
                    output_features.remove(i['name'])
        else:
            output_features = self.output_features
        return x[output_features].copy()


class ManualFeatureSelection(DataFrameTransform):
    """
    DataFrame transform that selects the given features
    """

    def __init__(self, schema: Schema, select_features: List[str], categories: Categories = None, **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            select_features (List): list of features to be selected
            **kwargs: other arguments
        """
        super().__init__(schema=schema, categories=categories, **kwargs)
        # Store the features to be selected
        self.select_features = select_features

    def fit(self, x: pd.DataFrame):
        """
        Fits data to create the transformation
        Args:
            x (DataFrame): DataFrame with data to fit

        Returns:

        """
        # Ensure that all features to select are in dataframe
        for i in self.select_features:
            assert i in x.columns
        # Store input features (those that come in DataFrame)
        self.input_features = x.columns.tolist()
        # Set output features as the selected features
        self.output_features = self.select_features

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given data
        Args:
            x (DataFrame): DataFrame with data to be transformed

        Returns:
            DataFrame with transformed data
        """
        # Return a copy of the input DataFrame, but with the selected columns
        if not self.training:
            # We are evaluating, se we remove the output columns ig they are not given
            # Otherwise, an exception will be raised for trying to access a non-existent column
            output_features = copy.deepcopy(self.select_features)
            for i in self.schema.outputs:
                if i['name'] not in x.columns and i['name'] in output_features:
                    output_features.remove(i['name'])
        else:
            output_features = self.output_features
        return x[output_features].copy()


class SelectInputOutputFeatures(ManualFeatureSelection):
    """
    DataFrame transform that selects the input and output features, and optionally the shapes
    """

    def __init__(self, schema: Schema, select_shapes: bool = False, categories: Categories = None, **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            select_shapes (bool): add "shapes" column to select list
            categories (Categories): the possible values for categorical features
            **kwargs: other arguments
        """
        select_features = list(map(lambda x: x['name'], schema.inputs + schema.outputs))
        if select_shapes:
            select_features.append('shapes')
        super().__init__(schema=schema, select_features=select_features, categories=categories, **kwargs)
        self.select_shapes = select_shapes


class SelectRequiredElements(ManualFeatureSelection):
    """
    DataFrame transform that selects the required elements.
    It also can select optional elements given them by name or type
    Finally, the shapes als can be selected optionally
    """

    def __init__(self,
                 schema: Schema,
                 select_shapes: bool = False,
                 exclusion: dict = None,
                 categories: Categories = None,
                 **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            select_shapes (bool): add "shapes" column to select list
            exclusion (dict): a dict that contains rules to also select some not required elements. Format:
                {
                    'inputs': {
                        'by_name': <list of input element names to select>,
                        'by_type': <list of input element types to select>
                    }
                    'outputs': {
                        'by_name': <list of output elements names to select>,
                        'by_type': <list of output elements types to select>
                    }
                }
            categories (Categories): the possible values for categorical features
            **kwargs: other arguments
        """
        select_features = []
        if exclusion is None:
            exclusion = {}
        for l, k in [(schema.inputs, 'inputs'), (schema.outputs, 'outputs')]:
            for i in l:
                if k in exclusion and exclusion[k] is not None:
                    if 'by_name' in exclusion[k] and i['name'] in exclusion[k]['by_name']:
                        select_features.append(i['name'])
                        continue
                    if 'by_type' in exclusion[k] and i['type'] in exclusion[k]['by_type']:
                        select_features.append(i['name'])
                        continue

                if i['required']:
                    select_features.append(i['name'])

        if select_shapes:
            select_features.append('shapes')
        super().__init__(schema=schema, select_features=select_features, categories=categories, **kwargs)
        self.select_shapes = select_shapes
        self.exclusion = exclusion


class DropNaNValues(DataFrameTransform):
    """
    DataFrame transform that deletes NaN values
    """

    def __init__(self, schema: Schema, categories: Categories = None, **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            **kwargs: other arguments
        """
        super().__init__(schema=schema, categories=categories, **kwargs)

    def fit(self, x: pd.DataFrame):
        """
        Fits data to create the transformation
        Args:
            x (pd.DataFrame): DataFrame with data to fit

        Returns:

        """
        # Get input features from input DataFrame
        self.input_features = x.columns.tolist()
        # Delete all NaN values
        x_no_nan = x.dropna()
        # Set the output features as the features that not have NaN values
        self.output_features = x_no_nan.columns.tolist()

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given data
        Args:
            x (DataFrame): DataFrame with data to be transformed

        Returns:
            DataFrame with transformed data
        """
        if self.training:
            # When training we remove all NaN values. If too much are removed we should discard the features that
            # have a lot of NaN values. In this case, is supposed that input DataFrame is the same as in fit
            # so the output will have the selected features (output_features) and only removes rows with NaNs
            return x[self.output_features].dropna().copy()
        else:
            # We are evaluating, se we remove the output columns ig they are not given
            # Otherwise, an exception will be raised for trying to access a non-existent column
            output_features = copy.deepcopy(self.output_features)
            for i in self.schema.outputs:
                if i['name'] not in x.columns and i['name'] in output_features:
                    output_features.remove(i['name'])
            # When testing, we should not remove any row (the model should return
            # one prediction per example)
            return x[output_features].copy()


class SimpleMissingValueImputation(DataFrameTransform):
    """
    DataFrame transform that set a value to NaN values
    For float features, the mean is used
    For categorical, the most common value is used
    """

    def __init__(self, schema: Schema, categories: Categories = None, **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            **kwargs: other arguments
        """
        super().__init__(schema=schema, categories=categories, **kwargs)
        # To store which value use for each element (column)
        self.impute_value_per_element = dict()

    def fit(self, x: pd.DataFrame):
        """
        Fits data to create the transformation
        Args:
            x (DataFrame): DataFrame with data to fit

        Returns:

        """
        # In this case, the input features are the same as output features
        # Get from DataFrame and store them
        self.input_features = x.columns.tolist()
        self.output_features = x.columns.tolist()
        # For each input feature (do not impute output values to avoid bad teaching to model)
        for i in self.input_features:
            # Assert that not all values are NaN
            if x[i].isna().all():
                raise DataError(f'All values are NaN for {i}, so \'MissingValueImputation\' cannot be applied')
            # Get from inputs the elements that match the id
            elements = self.schema.get_inputs_by_name(i)
            # We could get one unique element (is a match) or 0 (because 'i' is an output element)
            # Only fill NaNs on input elements
            if len(elements) >= 1:
                assert len(elements) == 1
                assert len(self.schema.get_outputs_by_name(i)) == 0
                element_type = elements[0]['type']
                if element_type == 'float':
                    self.impute_value_per_element[i] = x[i].dropna().mean()
                else:
                    # It could have multiple values, but we get the first one
                    self.impute_value_per_element[i] = x[i].dropna().mode().iloc[0]
            else:
                # The element should be an output element. Check that but do nothing
                elements = self.schema.get_outputs_by_name(i)
                assert len(elements) == 1

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given data
        Args:
            x (DataFrame): DataFrame with data to be transformed

        Returns:
            DataFrame with transformed data
        """
        # Fill NaN values with computed elements
        # The columns that are not in the impute_value_per_element dict are not filled
        # That is, the output column are not changed
        return x.fillna(self.impute_value_per_element).copy()


class JoinStringColumns(DataFrameTransform):
    """
    DataFrame transform to join several string columns in one column with given sep (default ' ').
    The result is stored in the first column of the given list
    """

    def __init__(self,
                 schema: Schema,
                 join_columns: List[str],
                 sep: str = ' ',
                 categories: Categories = None,
                 **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            join_columns (List): list of columns to be joined
            sep (str): separator to join columns
            **kwargs: other arguments
        """
        super().__init__(schema=schema, categories=categories, **kwargs)
        # Store join columns and separator
        self.join_columns = join_columns
        self.sep = sep

    def fit(self, x: pd.DataFrame):
        """
        Fits data to create the transformation
        Args:
            x (DataFrame): DataFrame with data to fit

        Returns:

        """
        # Ensure that all join columns are present in input DataFrame
        for i in self.join_columns:
            if i not in x.columns:
                raise DataError(f'Trying to join column {i}, but it is not in given data')
        # Store columns as input and output features (they will be the same)
        self.input_features = x.columns.tolist()
        self.output_features = x.columns.tolist()

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given data
        Args:
            x (DataFrame): DataFrame with data to be transformed

        Returns:
            DataFrame with transformed data
        """

        def _join_row(r: pd.Series) -> pd.Series:
            """
            Function that makes the join for a single row
            Args:
                r (Series): a single row of the DataFrame

            Returns:
                Series: the transformed row
            """
            # Create a copy
            r = r.copy()
            # Set the value of the series in the first "column" to be joined the result after joining them
            r[self.join_columns[0]] = self.sep.join(r[self.join_columns])
            return r

        # Apply the _join_row function to each row and return a copy of the DataFrame
        return x.apply(_join_row, axis=1).copy()


class CorrelationFeatureSelection(DataFrameTransform):
    """
    DataFrame transform that discards those features with high correlation with other ones
    """

    def __init__(self, schema: Schema, categories: Categories = None, threshold: float = 0.9, **kwargs):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            categories (Categories): the possible values for categorical features
            threshold (float): correlation threshold to select features
            **kwargs: other arguments
        """
        super().__init__(schema=schema, categories=categories, **kwargs)
        # Store threshold
        self.threshold = threshold

    def fit(self, x: pd.DataFrame):
        """
        Fits data to create the transformation
        Args:
            x (DataFrame): DataFrame with data to fit

        Returns:

        """
        # Get which features come in DataFrame
        self.input_features = x.columns.tolist()
        # Fit transformation with numpy data
        remove_features = []
        inputs = [f for f in self.input_features if f in [i['name'] for i in self.schema.inputs]]

        df_copy = x.copy()

        # Encode categorical outputs for getting correlations with them
        for out in self.schema.outputs:
            if out['type'] == 'category':
                df_copy[out['name']] = LabelEncoder().fit_transform(df_copy[out['name']])

        df_corr = df_copy.corr()

        # Get features with correlation higher than threshold
        for i, j in np.argwhere(np.abs(df_corr.loc[inputs, inputs].to_numpy()) > self.threshold):
            if i != j and not any(k in remove_features for k in [df_corr[inputs].index[i], df_corr[inputs].columns[j]]):
                votes_i = 0
                votes_j = 0
                # Remove feature with lower correlation with outputs
                for o in self.schema.outputs:
                    out = o['name']
                    if df_corr.loc[df_corr[inputs].index[i], out] > df_corr.loc[df_corr[inputs].columns[j], out]:
                        votes_i += 1
                    elif df_corr.loc[df_corr[inputs].index[i], out] < df_corr.loc[df_corr[inputs].columns[j], out]:
                        votes_j += 1
                remove_features.append(df_corr[inputs].index[i] if votes_i < votes_j else df_corr[inputs].columns[j])

        # Get the columns that will be used (those whose variance is bigger than threshold)
        self.output_features = [c for c in x.columns if c not in remove_features]

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given data
        Args:
            x (DataFrame): DataFrame with data to be transformed

        Returns:
            DataFrame with transformed data
        """
        # Return a copy of the input DataFrame, but with the selected columns
        if not self.training:
            # We are evaluating, se we remove the output columns ig they are not given
            # Otherwise, an exception will be raised for trying to access a non-existent column
            output_features = copy.deepcopy(self.output_features)
            for i in self.schema.outputs:
                if i['name'] not in x.columns:
                    output_features.remove(i['name'])
        else:
            output_features = self.output_features
        return x[output_features].copy()
