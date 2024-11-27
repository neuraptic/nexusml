import os
import pickle
from typing import Any, Dict, IO, List, Optional, Union

import cv2
from detectron2.config import CfgNode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo
from matplotlib import pyplot as plt
import pandas as pd
import torch

from nexusml.engine.data.transforms.base import DataFrameTransforms
from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.utils import predictions_to_example_format
from nexusml.engine.models.base import Model
from nexusml.engine.models.base import TrainingOutputInfo
from nexusml.engine.models.utils import smooth
from nexusml.engine.models.vision.utils import detectron_preds_to_shapes
from nexusml.engine.schema.base import Schema
from nexusml.engine.schema.categories import Categories
from nexusml.enums import MLProblemType
from nexusml.enums import TaskType


class DetectronObjectDetectionModel(Model):
    """
    Model class specialization for Detectron object detection models
    """

    def __init__(self,
                 schema: Schema,
                 categories: Categories,
                 dataframe_transforms: DataFrameTransforms,
                 model_config: Dict,
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
        self.cfg = None
        self.model_config = model_config
        self.predictor = None
        self.pytorch_model = None
        self.train_args = None

    def fit(self,
            train_data: Union[pd.DataFrame, dict, List[dict]],
            valid_data: Union[pd.DataFrame, dict, List[dict]] = None,
            train_args: Dict = None) -> TrainingOutputInfo:
        """
        Function called to train the model
        Args:
            train_data (Union[DataFrame, dict, List[dict]]): train data that could be a DataFrame, a single example
                                                            as dict, or a list of dict examples
            valid_data (Union[DataFrame, dict, List[dict]]): validation data that could be a DataFrame, a single example
                                                            as dict, or a list of dict examples
            train_args (Dict): dict with extra arguments for training like number of epochs.
                            Required keys: 'batch_size' and 'epochs'

        Returns:
            TrainingOutputInfo filled with the train history figures for each output
        """
        if isinstance(train_data, dict) or isinstance(train_data, list):
            train_data = Model.examples_to_dataframe(train_data)

        if isinstance(valid_data, dict) or isinstance(valid_data, list):
            valid_data = Model.examples_to_dataframe(valid_data)

        self.dataframe_transforms.fit(train_data)
        self.dataframe_transforms.train()
        dataset_name = self.dataframe_transforms.transform(train_data)

        # Fit the transformed data to both, input and output transforms
        # Note: the data is not transformed here, the dataset class does it
        self.input_transforms.fit(train_data)
        self.output_transforms.fit(train_data)

        # Put them on train mode
        self.input_transforms.train()
        self.output_transforms.train()

        # If given train_args is None, get the saved args
        if train_args is None:
            train_args = self.train_args
        else:
            # We have new training args, save them
            self.train_args = train_args

        # Get training device. 'cuda' if GPU is available. 'cpu' otherwise
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tr_metadata = MetadataCatalog.get(dataset_name)

        self.cfg = create_detectron_model_config(checkpoint_url=self.model_config['setup_args']['checkpoint_url'],
                                                 lr=train_args['lr'],
                                                 epochs=train_args['epochs'],
                                                 batch_size=train_args['batch_size'],
                                                 num_workers=train_args['num_workers'],
                                                 num_classes=len(tr_metadata.get('thing_classes', None)),
                                                 dataset_name=dataset_name,
                                                 device=device)

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        self.pytorch_model = trainer

        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR,
                                              'model_final.pth')  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1  # Non-max supression threshold

        self.predictor = DefaultPredictor(self.cfg)

        idx = 0
        for i, output in enumerate(self.schema.outputs):
            if output['type'] == 'shape':
                idx = i

        train_hist = {self.schema.outputs[idx]['name']: [x[0] for x in trainer.storage.history('total_loss').values()]}

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

        self.dataframe_transforms.eval()
        dataset_name = self.dataframe_transforms.transform(data)

        # Get the device. If not initialized, set device as 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if train_args is None:
            train_args = self.train_args

        tr_dicts = DatasetCatalog.get(dataset_name)

        bs = 1 if train_args is None or 'batch_size' not in train_args else train_args['batch_size']
        self.predictor.cfg.SOLVER.IMS_PER_BATCH = bs
        self.predictor.cfg.MODEL.DEVICE = device

        print('[+] Making predictions')

        predictions = []
        for d in tr_dicts:
            im = cv2.imread(d['file_name'])
            outputs = self.predictor(
                im
            )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

            predictions.append(outputs)

        predictions = detectron_preds_to_shapes(predictions=predictions,
                                                outputs=self.schema.outputs,
                                                inputs=self.schema.inputs,
                                                dataframe_transforms=self.dataframe_transforms)

        predictions = {self.schema.shape_type_outputs()[0]['name']: predictions}
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
        if schema.task_type != TaskType.OBJECT_DETECTION:
            return False
        # Single input
        if len(schema.inputs) != 1:
            return False
        # Image file and required
        if schema.inputs[0]['type'] != 'image_file' or not schema.inputs[0]['required']:
            return False
        # Two outputs: shape and not required category
        if len(schema.outputs) != 2:
            return False
        # The first is a required shape
        if schema.outputs[0]['type'] == 'shape':
            if not schema.outputs[0]['required']:
                return False
            # The second must be a not required category
            if schema.outputs[1]['type'] != 'category' or schema.outputs[1]['required']:
                return False
        elif schema.outputs[0]['type'] == 'category':
            # The first is a not required category
            if schema.outputs[0]['required']:
                return False
            # The second must be a required shape
            if schema.outputs[1]['type'] != 'shape' or not schema.outputs[1]['required']:
                return False
        else:
            return False
        # If all checks passed, return True
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
                    'exclusion': None,
                    'select_shapes': True
                },
            }, {
                'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
                'args': None
            }, {
                'class': 'nexusml.engine.data.transforms.vision.detectron.RegisterDetectionDatasetTransform',
                'args': None,
            }],
            'model': {
                'args': {
                    'setup_args': {
                        'checkpoint_url': 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'
                    },
                    'setup_function': 'nexusml.engine.models.vision.detectron.create_detection_model'
                },
                'class': 'nexusml.engine.models.vision.detectron.DetectronObjectDetectionModel'
            },
            'training': {
                'batch_size': 8,
                'epochs': 300,
                'loss_function': {
                    'args': {},
                    'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction'
                },
                'lr': 0.005,
                'num_workers': 4
            },
            'transforms': {
                'input_transforms': {
                    'global': {
                        'image_file': {
                            'args': [],
                            'class': 'nexusml.engine.data.transforms.vision.detectron.IdentityImageTransform'
                        }
                    },
                    'specific': None
                },
                'output_transforms': {
                    'global': {
                        'category': {
                            'args': {
                                'problem_type': MLProblemType.OBJECT_DETECTION
                            },
                            'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform'
                        },
                        'shape': {
                            'args': {
                                'problem_type': MLProblemType.OBJECT_DETECTION
                            },
                            'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform'
                        }
                    },
                    'specific': None
                }
            }
        }]

    def save_model(self, output_file: Union[str, IO]) -> None:
        """
        Method that saves all the information needed to create the Detectron model serialized in the given output_file
        In this case, we will store the information needed to create the model (all information
        used inside _setup_model function). Then, when the model is created, we have to load the weights.
        So we store the cfg of the model
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        with open(os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth'), 'rb') as f:
            model_weights = f.read()

        # Things to be saved
        to_store = {'config': self.cfg, 'model_weights': model_weights, 'train_args': self.train_args}

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

        os.makedirs(model_info['config'].OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(model_info['config'].OUTPUT_DIR, 'model_final.pth'), 'wb') as f:
            f.write(model_info['model_weights'])
        # Create the model
        pytorch_model = DefaultPredictor(model_info['config'])
        os.remove(os.path.join(model_info['config'].OUTPUT_DIR, 'model_final.pth'))

        # Return pytorch model as dict
        return {'predictor': pytorch_model, 'train_args': model_info['train_args']}

    def summary(self) -> Optional[str]:
        """
        Returns the summary of the trained model. In this case, just the string representation of the
        PyTorch model is returned

        Returns:
            string that will contain the summary of the PyTorch model (just string representation)
        """
        return str(self.pytorch_model.model)


class DetectronObjectSegmentationModel(Model):
    """
    Model class specialization for Detectron object segmentation models
    """

    def __init__(self,
                 schema: Schema,
                 dataframe_transforms: DataFrameTransforms,
                 model_config: Dict,
                 categories: Categories,
                 input_transforms: ElementTransforms,
                 output_transforms: ElementTransforms,
                 inference_mode: bool = False):
        """
        Default constructor
        Args:
            schema (Schema): the task schema
            model_config (Dict): the configuration to be used for model construction
            categories (Categories): the possible values for categorical features
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
        self.cfg = None
        self.model_config = model_config
        self.pytorch_model = None
        self.predictor = None
        self.train_args = None

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

        self.dataframe_transforms.fit(train_data)
        self.dataframe_transforms.train()
        dataset_name = self.dataframe_transforms.transform(train_data)

        # If given train_args is None, get the saved args
        if train_args is None:
            train_args = self.train_args
        else:
            # We have new training args, save them
            self.train_args = train_args

        # Get training device. 'cuda' if GPU is available. 'cpu' otherwise
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tr_metadata = MetadataCatalog.get(dataset_name)

        self.cfg = create_detectron_model_config(checkpoint_url=self.model_config['setup_args']['checkpoint_url'],
                                                 lr=train_args['lr'],
                                                 epochs=train_args['epochs'],
                                                 batch_size=train_args['batch_size'],
                                                 num_workers=train_args['num_workers'],
                                                 num_classes=len(tr_metadata.get('thing_classes', None)),
                                                 dataset_name=dataset_name,
                                                 device=device)

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        self.pytorch_model = trainer

        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR,
                                              'model_final.pth')  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set a custom testing threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1  # Non-max supression threshold

        self.predictor = DefaultPredictor(self.cfg)

        idx = 0
        for i, output in enumerate(self.schema.outputs):
            if output['type'] == 'shape':
                idx = i
        train_hist = {self.schema.outputs[idx]['name']: [x[0] for x in trainer.storage.history('total_loss').values()]}

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

        self.dataframe_transforms.eval()
        dataset_name = self.dataframe_transforms.transform(data)

        # Get the device. If not initialized, set device as 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if train_args is None:
            train_args = self.train_args

        tr_dicts = DatasetCatalog.get(dataset_name)

        bs = 1 if train_args is None or 'batch_size' not in train_args else train_args['batch_size']
        self.predictor.cfg.SOLVER.IMS_PER_BATCH = bs
        self.predictor.cfg.MODEL.DEVICE = device

        print('[+] Making predictions')

        predictions = []
        for d in tr_dicts:
            im = cv2.imread(d['file_name'])
            outputs = self.predictor(
                im
            )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

            predictions.append(outputs)

        predictions = detectron_preds_to_shapes(predictions=predictions,
                                                outputs=self.schema.outputs,
                                                inputs=self.schema.inputs,
                                                dataframe_transforms=self.dataframe_transforms)

        predictions = {self.schema.shape_type_outputs()[0]['name']: predictions}
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
        if schema.task_type != TaskType.OBJECT_SEGMENTATION:
            return False
        # Single input
        if len(schema.inputs) != 1:
            return False
        # Image file and required
        if schema.inputs[0]['type'] != 'image_file' or not schema.inputs[0]['required']:
            return False
        # Two outputs: shape and not required category
        if len(schema.outputs) != 2:
            return False
        # The first is a required shape
        if schema.outputs[0]['type'] == 'shape':
            if not schema.outputs[0]['required']:
                return False
            # The second must be a not required category
            if schema.outputs[1]['type'] != 'category' or schema.outputs[1]['required']:
                return False
        elif schema.outputs[0]['type'] == 'category':
            # The first is a not required category
            if schema.outputs[0]['required']:
                return False
            # The second must be a required shape
            if schema.outputs[1]['type'] != 'shape' or not schema.outputs[1]['required']:
                return False
        else:
            return False
        # If all checks passed, return True
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
                    'exclusion': None,
                    'select_shapes': True
                },
            }, {
                'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
                'args': None
            }, {
                'class': 'nexusml.engine.data.transforms.vision.detectron.RegisterSegmentationDatasetTransform',
                'args': None,
            }],
            'model': {
                'args': {
                    'setup_args': {
                        'checkpoint_url': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'
                    },
                    'setup_function': 'nexusml.engine.models.vision.detectron.create_segmentation_model'
                },
                'class': 'nexusml.engine.models.vision.detectron.DetectronObjectSegmentationModel'
            },
            'training': {
                'batch_size': 8,
                'epochs': 300,
                'loss_function': {
                    'args': {},
                    'class': 'nexusml.engine.models.common.pytorch.BasicLossFunction'
                },
                'lr': 0.005,
                'num_workers': 4
            },
            'transforms': {
                'input_transforms': {
                    'global': {
                        'image_file': {
                            'args': [],
                            'class': 'nexusml.engine.data.transforms.vision.detectron.IdentityImageTransform'
                        }
                    },
                    'specific': None
                },
                'output_transforms': {
                    'global': {
                        'category': {
                            'args': {
                                'problem_type': MLProblemType.OBJECT_SEGMENTATION
                            },
                            'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform'
                        },
                        'shape': {
                            'args': {
                                'problem_type': MLProblemType.OBJECT_SEGMENTATION
                            },
                            'class': 'nexusml.engine.data.transforms.vision.detectron.OutputIdentityTransform'
                        }
                    },
                    'specific': None
                }
            }
        }]

    def save_model(self, output_file: Union[str, IO]):
        """
        Method that saves all the information needed to create the Detectron model serialized in the given output_file
        In this case, we will store the information needed to create the model (all information
        used inside _setup_model function). Then, when the model is created, we have to load the weights.
        So we store the cfg of the model
        If the given output file is string, it will be the path where store the object
        If is not a string, it will be a file descriptor (opened file, IO buffer, etc.) where write the object
        Args:
            output_file (Union[str, IO]): output file path or output buffer/descriptor where store object

        Returns:

        """
        with open(os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth'), 'rb') as f:
            model_weights = f.read()

        # Things to be saved
        to_store = {'config': self.cfg, 'model_weights': model_weights, 'train_args': self.train_args}

        # If the given output file is a string, open the file and write the object (serialized with pickle)
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                pickle.dump(to_store, f)
        else:
            # If is not a string, write the object there
            pickle.dump(to_store, output_file)

    @classmethod
    def load_model(cls, input_file: Union[str, IO], schema: Schema, input_transforms: ElementTransforms,
                   output_transforms: ElementTransforms,
                   dataframe_transforms: DataFrameTransforms) -> Dict[str, DefaultPredictor]:
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

        os.makedirs(model_info['config'].OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(model_info['config'].OUTPUT_DIR, 'model_final.pth'), 'wb') as f:
            f.write(model_info['model_weights'])

        # Create the model
        pytorch_model = DefaultPredictor(model_info['config'])
        os.remove(os.path.join(model_info['config'].OUTPUT_DIR, 'model_final.pth'))

        # Return pytorch model as dict
        return {'predictor': pytorch_model, 'train_args': model_info['train_args']}

    def summary(self) -> Any:
        """
        Returns the summary of the trained model. In this case, just the string representation of the
        PyTorch model is returned

        Returns:
            string that will contain the summary of the PyTorch model (just string representation)
        """
        return str(self.pytorch_model.model)


def create_detectron_model_config(checkpoint_url: str, lr: float, epochs: int, batch_size: int, num_workers: int,
                                  num_classes: int, dataset_name: str, device: str) -> CfgNode:
    '''
    Function that creates the Detectron model configuration
    Args:
        checkpoint_url (str): checkpoint file from which load weights
        lr (float): learning rate for training
        epochs (int): epochs for training
        batch_size (int): batch size for training
        num_workers (int): number of workers for training
        num_classes (int): number of classes for training
        dataset_name (str): name of the Detectron dataset
        device (str): the device to be used for training

    Returns:
        CfgNode with the configuration object for the model
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(checkpoint_url))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.SOLVER.MAX_ITER = epochs
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MIN_SIZE_TRAIN = 512
    cfg.INPUT.MAX_SIZE_TRAIN = 768
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 768
    cfg.INPUT.RANDOM_FLIP = 'horizontal'

    return cfg
