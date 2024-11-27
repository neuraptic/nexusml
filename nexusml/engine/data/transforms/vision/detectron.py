import json

import cv2
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.mask import encode

from nexusml.engine.data.transforms.base import DataFrameTransform
from nexusml.engine.data.transforms.base import Transform
from nexusml.engine.data.transforms.base import TransformOutputInfo
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import MLProblemType


def get_object_detection_dataset(data: pd.DataFrame, categories: Categories, schema: Schema, training=True) -> list:
    """
    Get the object detection dataset from the given data.

    Args:
        data (pd.DataFrame): the data
        categories (Categories): the categories
        schema (Schema): the schema
        training (bool): if it is training or not

    Returns:
        list: the list of examples
    """

    schema_inputs = schema.inputs
    schema_outputs = schema.categorical_outputs()

    output_categories = categories.get_categories(schema_outputs[0]['name'])

    all_examples = []

    for ex_idx in range(len(data)):
        example = data.iloc[ex_idx]
        image_dict = {}
        image_dict['file_name'] = example[schema_inputs[0]['name']]

        if training:
            image_dict['annotations'] = []
            shape_ids = example[schema.shape_type_outputs()[0]['name']]
            if not isinstance(shape_ids, list):
                shape_ids = [shape_ids]
            for shape_id in shape_ids:
                for shape in json.loads(example['shapes']):
                    if shape['id'] == shape_id:
                        annotation = {}
                        polygon_x = [coord['x'] for coord in shape['polygon']]
                        polygon_y = [coord['y'] for coord in shape['polygon']]
                        annotation['bbox'] = [min(polygon_x), min(polygon_y), max(polygon_x), max(polygon_y)]
                        annotation['bbox_mode'] = BoxMode.XYXY_ABS
                        annotation['category_id'] = output_categories.index(shape['outputs'][0]['value'])
                        annotation['segmentation'] = []
                        image_dict['annotations'].append(annotation)

        all_examples.append(image_dict)

    return all_examples


def get_object_segmentation_dataset(data: pd.DataFrame, categories: Categories, schema: Schema, training=True) -> list:
    """
    Get the object segmentation dataset from the given data.

    Args:
        data (pd.DataFrame): the data
        categories (Categories): the categories
        schema (Schema): the schema
        training (bool): if it is training or not

    Returns:
        list: the list of examples
    """

    schema_inputs = schema.inputs
    schema_outputs = schema.categorical_outputs()

    output_categories = categories.get_categories(schema_outputs[0]['name'])

    all_examples = []

    for ex_idx in range(len(data)):
        example = data.iloc[ex_idx]
        image_dict = {}
        image_dict['file_name'] = example[schema_inputs[0]['name']]

        if training:
            height, width = Image.open(image_dict['file_name']).size

            image_dict['annotations'] = []
            shape_ids = example[schema.shape_type_outputs()[0]['name']]
            if not isinstance(shape_ids, list):
                shape_ids = [shape_ids]
            for shape_id in shape_ids:
                for shape in json.loads(example['shapes']):
                    if shape['id'] == shape_id:
                        annotation = {}
                        polygon_x = [coord['x'] for coord in shape['polygon']]
                        polygon_y = [coord['y'] for coord in shape['polygon']]
                        annotation['bbox'] = [min(polygon_x), min(polygon_y), max(polygon_x), max(polygon_y)]
                        annotation['bbox_mode'] = BoxMode.XYXY_ABS
                        annotation['category_id'] = output_categories.index(shape['outputs'][0]['value'])
                        mask = np.zeros((width, height), dtype=np.uint8)
                        # Segmentation shapes will be defined by a polygon's vertices
                        # Get the polygon's vertices, draw a polygon in a white image and get pixels
                        pts = np.array([[polygon_x[i], polygon_y[i]] for i in range(len(polygon_x))], np.int32)
                        polygon_image = cv2.fillPoly(mask, [pts], 1)

                        annotation['segmentation'] = encode(np.asarray(polygon_image, order='F'))
                        image_dict['annotations'].append(annotation)

        all_examples.append(image_dict)

    return all_examples


class RegisterDetectionDatasetTransform(DataFrameTransform):
    """
    DataFrame transform that registers a dataset in Detectron with the given features
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
        self.name = 'train'
        self.categories = categories

    def fit(self, x: pd.DataFrame):
        """
        Fits data to create the transformation
        Args:
            x (DataFrame): DataFrame with data to fit

        Returns:

        """
        pass

    def transform(self, data: pd.DataFrame) -> str:
        """
        Transform the given data
        Args:
            data (DataFrame): DataFrame with data to be transformed

        Returns:
            str with the name of the registered dataset
        """
        if self.name in DatasetCatalog.values()._mapping.data.keys():
            DatasetCatalog.remove(self.name)

        DatasetCatalog.register(name=self.name,
                                func=lambda x=data, schema=self.schema, cat=self.categories, training=True
                                if self.name == 'train' else False: get_object_detection_dataset(
                                    data=x, categories=cat, schema=schema, training=training))

        MetadataCatalog.get(self.name).thing_classes = (self.categories.get_categories(
            self.schema.categorical_outputs()[0]['name']))

        return self.name

    def train(self):
        """
        Train method. Sets training to True
        """
        self.name = 'train'

    def eval(self):
        """
        Test method. Sets training to False
        """
        self.name = 'test'


class RegisterSegmentationDatasetTransform(DataFrameTransform):
    """
    DataFrame transform that registers a dataset in Detectron with the given features
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

        self.name = 'train'

    def fit(self, x: np.ndarray):
        """
        Fits data to create the transformation
        Args:
            x (DataFrame): DataFrame with data to fit

        Returns:

        """
        pass

    def transform(self, data: pd.DataFrame) -> str:
        """
        Transform the given data
        Args:
            x (DataFrame): DataFrame with data to be transformed

        Returns:
            str with the name of the registered dataset
        """
        if self.name in DatasetCatalog.values()._mapping.data.keys():
            DatasetCatalog.remove(self.name)

        DatasetCatalog.register(name=self.name,
                                func=lambda x=data, schema=self.schema, cat=self.categories, training=True
                                if self.name == 'train' else False: get_object_segmentation_dataset(
                                    data=x, categories=cat, schema=schema, training=training))

        MetadataCatalog.get(self.name).thing_classes = (self.categories.get_categories(
            self.schema.categorical_outputs()[0]['name']))

        return self.name

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def train(self):
        """
        Train method. Sets training to True
        """
        self.name = 'train'

    def eval(self):
        """
        Test method. Sets training to False
        """
        self.name = 'test'


class OutputIdentityTransform(Transform):
    """
    Applies no transform to the output data
    """

    def __init__(self, problem_type: int, **kwargs):
        """
        Default constructor
        Args:
            problem_type (int): the type of the problem
            **kwargs: other arguments
        """
        super().__init__(**kwargs)
        self.transform_output_info = TransformOutputInfo(output_type='category',
                                                         num_features=1,
                                                         choice_counter=None,
                                                         stats=None,
                                                         output_problem_type=MLProblemType(problem_type))

    def fit(self, x: np.ndarray):
        pass

    def transform(self, data: pd.DataFrame):
        pass

    def inverse_transform(self, x: np.ndarray):
        pass

    def get_transform_output_info(self) -> TransformOutputInfo:
        """
        Return the transform output info
        Returns:
            TransformOutputInfo object filled
        """
        return self.transform_output_info


class IdentityImageTransform(Transform):
    """
    Applies no transform to the input data
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        pass
