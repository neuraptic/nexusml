# TODO: Try to make this module independent from `nexusml.api`

import abc
import copy
from datetime import datetime
import json
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple
import zipfile

import psutil
from sklearn.model_selection import train_test_split
from sqlalchemy import and_ as sql_and
import torch
import yaml

from nexusml.api.resources.ai import AIModel
from nexusml.api.resources.base import dump
from nexusml.api.resources.examples import Example
from nexusml.api.resources.files import TaskFile
from nexusml.api.resources.tasks import InputCategory
from nexusml.api.resources.tasks import InputElement
from nexusml.api.resources.tasks import OutputCategory
from nexusml.api.resources.tasks import OutputElement
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.ai import AIModelRequest
from nexusml.api.schemas.files import TaskFileRequest
from nexusml.api.schemas.services import MonitoringServiceTemplatesSchema
from nexusml.api.utils import get_file_storage_backend
from nexusml.constants import DATETIME_FORMAT
from nexusml.database.ai import AIModelDB
from nexusml.database.core import db_commit
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.files import TaskFileDB
from nexusml.database.services import Service
from nexusml.database.services import ServiceType
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.engine.exceptions import DataError
from nexusml.engine.exceptions import ExperimentError
from nexusml.engine.exceptions import SchemaError
from nexusml.engine.experiments.run import retrain_model
from nexusml.engine.experiments.run import run_experiment_from_config_file
from nexusml.engine.experiments.tracking.mlflow import get_best_model
from nexusml.engine.models.base import Model
from nexusml.engine.models.utils import discover_models
from nexusml.engine.schema.base import Schema
from nexusml.engine.services.monitoring import MonitoringService
from nexusml.enums import AIEnvironment
from nexusml.enums import ElementType
from nexusml.enums import EngineType
from nexusml.enums import FileStorageBackend
from nexusml.enums import LabelingStatus
from nexusml.statuses import AL_WAITING_STATUS_CODE
from nexusml.statuses import CL_DATA_ERROR_STATUS_CODE
from nexusml.statuses import CL_INITIALIZING_TRAINING_STATUS_CODE
from nexusml.statuses import CL_SCHEMA_ERROR_STATUS_CODE
from nexusml.statuses import CL_TRAINING_STATUS_CODE
from nexusml.statuses import CL_UNKNOWN_ERROR_STATUS_CODE
from nexusml.statuses import CL_WAITING_STATUS_CODE
from nexusml.statuses import INFERENCE_WAITING_STATUS_CODE
from nexusml.statuses import MONITORING_WAITING_STATUS_CODE
from nexusml.statuses import Status
from nexusml.statuses import TASK_ACTIVE_STATUS_CODE
from nexusml.statuses import TESTING_WAITING_STATUS_CODE
from nexusml.utils import get_s3_config
from nexusml.utils import s3_client


class EngineWorker(abc.ABC):
    """
    Abstract base class that defines the interface for an AI Engine. The `EngineWorker` class handles the
    logic for version management, downloading data and files, preparing the environment for training
    or inference tasks, and abstract methods for specific task interactions such as model deployment
    and predictions. Subclasses should implement all abstract methods.

    The `EngineWorker` class contains utility methods to verify version formats, retrieve the largest
    available model version, and generate the next valid version by incrementing the version number.
    It also handles downloading task-related data, files, and model versions, which can be overridden
    in subclasses to account for specific storage or deployment strategies.
    """

    def __init__(self, task_uuid: str):
        """
        Initialize the EngineWorker instance with a specific task UUID.

        Args:
            task_uuid (str): The unique identifier of the task.
        """
        self._task_uuid = task_uuid

    @staticmethod
    def _verify_monitoring_templates(monitoring_templates: dict, task_id: int):
        """
        Verifies the integrity of the monitoring templates using a schema validator
        and ensures the templates are suitable for the provided task.

        Steps:
        1. Load the monitoring templates using the `MonitoringServiceTemplatesSchema` loader.
        2. Use the `MonitoringService` class to verify that the templates are correctly associated
           with the given task.

        Args:
            monitoring_templates (dict): Dictionary containing the monitoring templates.
            task_id (int): Identifier of the task to which the monitoring templates should be associated.
        """
        monitoring_templates = MonitoringServiceTemplatesSchema().load(monitoring_templates)
        MonitoringService.verify_templates(templates=monitoring_templates, task_id=task_id)

    @staticmethod
    def random_split(all_examples: list, test_size: float = 0.3, seed: int = None) -> Tuple[list, list]:
        """
        Function used for splitting randomly examples into train and test
        Args:
            all_examples (list): list with all examples
            test_size (float): the percentage [0, 1] to use as test
            seed (int): seed for random numbers

        Returns:
            Tuple of:
                - list of training examples
                - list of test examples
        """
        train_ex, test_ex = train_test_split(all_examples, test_size=test_size, random_state=seed)
        return train_ex, test_ex

    @staticmethod
    def version_is_bigger(a: Tuple[str, str, str], b: Tuple[str, str, str]) -> bool:
        """
        Compares two versions represented as tuples of strings. Determines if version 'a' is greater than version 'b'.
        The comparison is done lexicographically for each segment of the version tuple.

        Steps:
        1. Validate the format of the provided version tuples (must be in the format x.x.x).
        2. Convert version components from strings to integers.
        3. Perform lexicographic comparison for each component of the version tuple (major, minor, patch).

        Args:
            a (Tuple[str, str, str]): Version to be compared.
            b (Tuple[str, str, str]): Version to compare against.

        Returns:
            bool: True if version 'a' is greater than version 'b', False otherwise.
        """

        def _check_version(v):
            if not isinstance(v, tuple):
                raise ValueError('Excepted version is a tuple of 3 strings elements. i.e. ("0", "1", "0") for v0.1.0')
            if len(v) != 3:
                raise ValueError('Excepted version is a tuple of 3 strings elements. i.e. ("0", "1", "0") for v0.1.0')
            if not all(list(map(lambda x: isinstance(x, str), v))):
                raise ValueError('Excepted version is a tuple of 3 strings elements. i.e. ("0", "1", "0") for v0.1.0')

        # Check that the given versions are in the correct format
        _check_version(a)
        _check_version(b)
        # Convert to integers
        try:
            a = tuple(map(int, a))
            b = tuple(map(int, b))
        except ValueError:
            raise ValueError('Excepted version is a tuple of 3 strings elements. i.e. ("0", "1", "0") for v0.1.0')

        # Compare numbers
        # The first digit
        if a[0] > b[0]:
            # If the digit is bigger, the version is bigger
            return True
        elif a[0] < b[0]:
            # Is the digit is smaller, the version is smaller
            return False
        else:
            # If there are equal, go to next digit
            if a[1] > b[1]:
                # If the digit is bigger, the version is bigger
                return True
            elif a[1] < b[1]:
                # Is the digit is smaller, the version is smaller
                return False
            else:
                # If there are equal, go to next digit
                # And the version will be bigger, only if the last digit is bigger
                return a[2] > b[2]

    def get_biggest_model_version(self) -> Optional[Tuple[str, str, str]]:
        """
        Retrieves the highest valid version from all available models. Version format must match x.x.x.

        Steps:
        1. Retrieve all models.
        2. Use a regular expression to filter models with valid version tags (x.x.x).
        3. Compare model versions to find the largest one.

        Returns:
            Optional[Tuple[str, str, str]]: The largest version found, or None if no valid version exists.
        """
        # Get all models
        models = self.get_models()
        # If there is no model yet, return None
        if len(models) == 0:
            return None
        else:
            # Get the biggest version (tag). Must match x.x.x (like 0.1.1)
            regexp = re.compile('^(\d+)\.(\d+)\.(\d+)$')
            # Init as None
            max_version = None
            # For each model
            for i in models:
                # Get the tag (version)
                m_tag = i['version']
                # Match with regexp
                match = regexp.match(m_tag)
                # If there is a matching, get the 3 version values
                if match:
                    v = match.groups()
                    # If the previous max version is None, update it to the current model version
                    if max_version is None:
                        max_version = v
                    else:
                        # Otherwise, update max version with the current model version only if the last one is bigger
                        max_version = v if EngineWorker.version_is_bigger(v, max_version) else max_version
            return max_version

    def get_next_model_version(self) -> str:
        """
        Generates the next model version by incrementing the patch version of the largest existing model version.
        If no model exists, returns the first version '0.1.0'.

        Steps:
        1. Retrieve the largest existing model version.
        2. Increment the patch version (third component) by 1.
        3. If no models exist, return the initial version '0.1.0'.

        Returns:
            str: The next model version in the format x.x.x.
        """
        # Get the current biggest version
        current_biggest_version = self.get_biggest_model_version()
        if current_biggest_version is None:
            # If there is no model (or no matching version), return the first one (0.1.0)
            return '0.1.0'
        else:
            # Sum 1 to the last number
            return '.'.join(current_biggest_version[:2] + (str(int(current_biggest_version[2]) + 1),))

    def train_initial_model(self, working_dir: str, data_dir: str, paths: dict) -> Tuple[str, str]:
        """
        Performs the initial training of a model based on the task schema. This function starts by retrieving
        the task schema and then filtering candidate models that can support it. If no compatible models are found,
        the function raises a ValueError indicating that the schema is not trainable from scratch.

        For each compatible model, the function generates default configurations, modifies them to include data paths,
        experiment details, schema paths, and optional categories if provided. These configurations are saved in the
        working directory for each model. The function then sequentially runs experiments using the generated
        configuration files, while updating the progress of the training process.

        Once all experiments are completed, the function identifies and retrieves the best model based on the experiment
        results, returning both the path to the best model and its configuration file.

        Args:
            working_dir (str): Path to the working directory where configuration files will be stored.
            data_dir (str): Path to the directory where experiment data is located.
            paths (dict): Dictionary containing paths to schema, train data, test data, categories, and
                          other necessary files.

        Returns:
            tuple: A tuple containing the path to the best model and the best model's configuration file.

        Raises:
            ValueError: If the schema is not trainable from scratch due to no compatible models.
            Exception: If any errors occur during experiment execution, these will be captured and logged.
        """
        # Get Task Schema
        task_schema = Schema.create_schema_from_dict(self.get_task_schema())

        # Get candidate models
        candidate_models = list(filter(lambda x: x.supports_schema(schema=task_schema), discover_models()))

        # If there are no candidate models, the schema is not trainable from scratch
        if len(candidate_models) == 0:
            raise SchemaError('The schema is not trainable from scratch')

        mlflow_uri = 'file:///' + os.path.join(data_dir, 'mlruns').replace('\\', '/')

        # Generate the default configs for the candidate models
        config_file_paths = []
        idx = 1
        for model in candidate_models:
            for default_config in model.get_default_configs():
                config = copy.deepcopy(default_config)
                # Add data, experiment, schema and seed
                config['data'] = {'train_data': paths['train_data_path'], 'test_data': paths['test_data_path']}
                config['experiment'] = {
                    'mlflow_uri': mlflow_uri,
                    'name': 'exp1',
                    'save_figures': False,
                    'save_predictions': False
                }
                config['schema'] = paths['schema_path']
                config['seed'] = 0
                # Add categories path if given
                if 'categories_path' in paths and paths['categories_path'] is not None:
                    config['categories'] = paths['categories_path']
                # Output file name
                output_file_name = f'conf_{idx}.yaml'
                output_file = os.path.join(working_dir, output_file_name)
                with open(output_file, 'w') as f:
                    yaml.dump(config, f)
                # Store config file path in list
                config_file_paths.append(output_file)
                # Add 1 to idx
                idx += 1

        # Run all experiments with the generated config files
        for idx, conf_file_name in enumerate(config_file_paths):
            try:
                run_experiment_from_config_file(config_file_path=os.path.join(working_dir, conf_file_name))
            except Exception as e:
                # An experiment could fail, let the others finish
                pass
            finally:
                # Always update the progress
                progress = (idx + 1) / len(config_file_paths)
                progress = int(round(progress * 100))
                progress = min(progress, 100)
                self.update_continual_learning_service_status(code=CL_TRAINING_STATUS_CODE,
                                                              details={'progress_percentage': progress})

        # Get best model
        try:
            best_model_path, best_model_config_path = get_best_model(mlflow_uri=mlflow_uri, experiment_name='exp1')
            return best_model_path, best_model_config_path
        except Exception as e:
            raise ExperimentError('Error retrieving the best model for the experiment')

    def train(self, model_uuid: Optional[str] = None, **kwargs):
        """
        Handles the training or retraining process of a model. If no model_uuid is provided,
        it initiates the first training process. If a model_uuid is provided, the model is retrieved,
        and the retraining process is triggered using the entire dataset.

        This function creates temporary directories for storing configuration files and data,
        retrieves or prepares the data, handles model training, compresses the resulting model,
        and finally updates the CL status.

        Steps:
        1. Start the training timer.
        2. Create temporary working and data directories.
        3. Prepare the training data.
        4. Perform the first training or load the existing model for retraining.
        5. Retrain the model using the whole dataset.
        6. Compress the model and its configuration.
        7. Perform post-training tasks such as computing monitoring templates, and updating task usage.

        Args:
            model_uuid (Optional[str]): UUID of the model to be retrained. If None, performs first training.
            **kwargs: Additional arguments for future extensions or subclass-specific behavior.

        Returns:
            None
        """
        try:
            # To measure time
            training_start_time = time.time()

            # Working dir for confing files
            working_dir = tempfile.TemporaryDirectory()
            working_dir_name = working_dir.name

            # Directory to store schema, data, and experiments
            data_dir = tempfile.TemporaryDirectory()
            data_dir_name = data_dir.name

            # Set status as initializing
            self.update_continual_learning_service_status(code=CL_INITIALIZING_TRAINING_STATUS_CODE)

            # Prepare data
            try:
                paths = self.prepare(output_path=data_dir_name, test_size=0.2, seed=0)
            except Exception as e:
                raise DataError('Error preparing the training data')

            # If model_uuid is None, make the first training
            if model_uuid is None:
                best_model_path, best_model_config_path = self.train_initial_model(working_dir=working_dir_name,
                                                                                   data_dir=data_dir_name,
                                                                                   paths=paths)
            else:
                # Download model
                # Get the file
                self.get_model_file(model_uuid=model_uuid, output_path=data_dir_name)

                # Zip file
                out_zip_file = os.path.join(data_dir_name, model_uuid)

                # Open the zip file and extract the specific file
                with zipfile.ZipFile(out_zip_file, 'r') as zipf:
                    # Check if the file is in the zip archive
                    if 'model.pkl' in zipf.namelist():
                        zipf.extract('model.pkl', data_dir_name)
                        zipf.extract('config.yaml', data_dir_name)

                best_model_config_path = os.path.join(data_dir_name, 'config.yaml')

            # Retrain with whole data
            retrain_model(base_config_file=best_model_config_path,
                          schema=paths['schema_path'],
                          categories=paths['categories_path'],
                          train_data=paths['data_path'],
                          output_path=data_dir_name)
            best_model_path = os.path.join(data_dir_name, 'model.pkl')
            best_model_config_path = os.path.join(data_dir_name, 'config.yaml')

            # Create a zip with model and config
            zip_file_name = os.path.join(data_dir_name, 'model.zip')
            with zipfile.ZipFile(zip_file_name, 'w') as zipf:
                # Add the files to the zip file
                zipf.write(best_model_path, os.path.basename(best_model_path))
                zipf.write(best_model_config_path, os.path.basename(best_model_config_path))

            self.update_continual_learning_service_status(code=CL_TRAINING_STATUS_CODE,)

            # Get monitoring templates
            with open(paths['data_path'], 'r') as f:
                data = json.load(f)
            m = Model.load(input_file=best_model_path)
            monitoring_templates = m.compute_templates(data=data, output_file_path=None)

            # Remove model from memory
            del m

            # Get training time
            training_end_time = time.time()

            # Get training device
            training_device = 'gpu' if torch.cuda.is_available() else 'cpu'

            # Create model
            self.create_ai_model(model_path=zip_file_name,
                                 training_device=training_device,
                                 training_time=(training_end_time - training_start_time),
                                 monitoring_templates=monitoring_templates)

            self.update_continual_learning_service_status(code=CL_WAITING_STATUS_CODE)

            # Get the total amount of time
            total_time = time.time() - training_end_time

            # Update task usage
            self.update_task_usage(cpu_hours=total_time,
                                   gpu_hours=(training_end_time -
                                              training_start_time) if training_device == 'gpu' else 0)

            # Set task status as active
            self.update_task_status(code=TASK_ACTIVE_STATUS_CODE)

            # If services are enabled, set them as active in a waiting status
            service_settings = self.get_services_settings()
            if service_settings.get('inference', {}).get('enabled', False):
                self.update_inference_service_status(code=INFERENCE_WAITING_STATUS_CODE)
            if service_settings.get('active_learning', {}).get('enabled', False):
                self.update_active_learning_service_status(code=AL_WAITING_STATUS_CODE)
            if service_settings.get('monitoring', {}).get('enabled', False):
                self.update_monitoring_service_status(code=MONITORING_WAITING_STATUS_CODE)
            if service_settings.get('testing', {}).get('enabled', False):
                self.update_testing_service_status(code=TESTING_WAITING_STATUS_CODE)
        except SchemaError:
            self.update_continual_learning_service_status(code=CL_SCHEMA_ERROR_STATUS_CODE)
        except DataError:
            self.update_continual_learning_service_status(code=CL_DATA_ERROR_STATUS_CODE)
        except Exception as e:
            self.update_continual_learning_service_status(code=CL_UNKNOWN_ERROR_STATUS_CODE)
        finally:
            try:
                # Clean temp dirs
                working_dir.cleanup()
                data_dir.cleanup()
            except Exception as e:
                pass

    @abc.abstractmethod
    def get_task_schema(self) -> dict:
        """
        Abstract method to retrieve the task schema, which includes input/output element details, metadata, and more.
        Must be implemented by subclasses.

        Returns:
            dict: Task schema information.
        """
        pass

    @abc.abstractmethod
    def get_element_categories(self, task_schema: dict) -> dict:
        """
        Abstract method to retrieve element categories based on the provided task schema. Element categories
        correspond to inputs, outputs, or metadata that are categorized.

        Args:
            task_schema (dict): The schema of the task containing inputs, outputs, and metadata.

        Returns:
            dict: Dictionary with element categories.
        """
        pass

    @abc.abstractmethod
    def get_labeled_examples(self) -> list:
        """
        Abstract method to retrieve labeled examples associated with a specific task. Examples consist of labeled data
        required for training or testing machine learning models.

        Returns:
            list: A list of labeled examples.
        """
        pass

    @abc.abstractmethod
    def get_file(self, file_uuid: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Abstract method to retrieve a file by its UUID. Optionally, the file can be saved to an output path.
        If no output path is provided, the file should be loaded into memory.

        Args:
            file_uuid (str): UUID of the file to retrieve.
            output_path (Optional[str]): Path to save the file. If None, the file is loaded into memory.

        Returns:
            bytes or None: Returns file content if output_path is None, otherwise saves the file to the specified path.
        """
        pass

    def download_data_files(self, task_schema: dict, data: List[dict], output_path: str):
        """
        Downloads the task-related files specified in the task schema and stores them in the output path.

        Iterates through the task schema to find elements marked as file types and proceeds to download
        files for each matching data element in the input or output values of the task data.

        Important steps:
        1. Identify elements marked as file types in the schema.
        2. Create a directory for files under the specified output path.
        3. Download the files and update the data structure with the new file paths.

        Args:
            task_schema (dict): A dictionary containing task schema details, specifically inputs and outputs.
            data (List[dict]): A list of dictionaries containing task data, including file UUIDs to download.
            output_path (str): The directory path where files will be downloaded.

        Assertions:
            - The `task_schema` contains file type elements.
            - Files are successfully downloaded to the specified output path.
        """
        file_type_elements = []
        for el in task_schema.get('inputs', []) + task_schema.get('outputs', []):
            if '_file' in el['type']:
                file_type_elements.append(el['name'])

        # Only download if there are file type elements
        if len(file_type_elements) > 0:
            os.makedirs(os.path.join(output_path, 'files'), exist_ok=True)
            for ex in data:
                for el in ex.get('values', []) + ex.get('inputs', []) + ex.get('outputs', []):
                    if el['element'] in file_type_elements:
                        file_id = el['value']
                        self.get_file(file_uuid=file_id, output_path=os.path.join(output_path, 'files'))
                        # Update the value with the file path
                        el['value'] = os.path.join(output_path, 'files', file_id)

    def prepare(self, output_path: str, test_size: float = 0.2, seed: int = 0) -> dict:
        """
        Prepares the environment by downloading schema, categories, and labeled examples.
        Files associated with the task elements are also downloaded, and data is split into training
        and test sets.

        Steps:
        1. Download and save the task schema to the output path.
        2. Download element categories and save them.
        3. Retrieve labeled examples.
        4. Download file-type elements and save them locally.
        5. Split the labeled examples into training and test sets.
        6. Save the training and test sets to the output path.

        Args:
            output_path (str): Directory path where data and files will be saved.
            test_size (float): Fraction of the data to be used as the test set.
            seed (int): Random seed for splitting data into train/test sets.

        Returns:
            dict: Dictionary containing paths to the saved schema, data, train/test sets, and categories.
        """
        # Download schema
        task_schema = self.get_task_schema()
        with open(os.path.join(output_path, 'schema.json'), 'w') as f:
            json.dump(task_schema, f)

        # Download categories
        categories = self.get_element_categories(task_schema)
        with open(os.path.join(output_path, 'categories.json'), 'w') as f:
            json.dump(categories, f)

        # Download examples
        data = self.get_labeled_examples()

        # Download example files
        self.download_data_files(task_schema=task_schema, data=data, output_path=output_path)

        # Split train/test
        train_data, test_data = EngineWorker.random_split(all_examples=data, test_size=test_size, seed=seed)

        # Store data
        with open(os.path.join(output_path, 'data.json'), 'w') as f:
            json.dump(data, f)

        with open(os.path.join(output_path, 'train_data.json'), 'w') as f:
            json.dump(train_data, f)

        with open(os.path.join(output_path, 'test_data.json'), 'w') as f:
            json.dump(test_data, f)

        return {
            'schema_path': os.path.join(output_path, 'schema.json'),
            'data_path': os.path.join(output_path, 'data.json'),
            'train_data_path': os.path.join(output_path, 'train_data.json'),
            'test_data_path': os.path.join(output_path, 'test_data.json'),
            'categories_path': os.path.join(output_path, 'categories.json') if len(categories) > 0 else None
        }

    @abc.abstractmethod
    def get_models(self) -> list:
        """
        Abstract method to retrieve a list of all models associated with the task. The models contain
        metadata such as version, file information, and other relevant details.

        Returns:
            list: A list of models associated with the task.
        """
        pass

    @abc.abstractmethod
    def get_model(self, model_uuid: str) -> dict:
        """
        Abstract method to retrieve metadata for a specific model by its UUID. The model contains
        details such as its version, associated files, and deployment information.

        Args:
            model_uuid (str): UUID of the model to retrieve.

        Returns:
            dict: Dictionary containing the model metadata.
        """
        pass

    def get_model_file(self, model_uuid: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Retrieves a model file by its UUID and optionally saves it to the specified output path.
        If no output path is provided, the file content is loaded into memory.

        Steps:
        1. Retrieve the model metadata.
        2. Retrieve the associated file UUID and download it.
        3. Optionally, rename the file after downloading.

        Args:
            model_uuid (str): UUID of the model file to retrieve.
            output_path (Optional[str]): Path to save the file. If None, the file is loaded into memory.

        Returns:
            bytes or None: Returns file content if output_path is None, otherwise saves the file to the specified path.
        """
        model = self.get_model(model_uuid)
        file_uuid = model['file']['uuid']
        content = self.get_file(file_uuid, output_path=output_path)
        if content is None:
            os.rename(os.path.join(output_path, file_uuid), os.path.join(output_path, model_uuid))
        else:
            return content

    @abc.abstractmethod
    def load_model(self, model_uuid: str) -> Model:
        """
        Abstract method that defines the interface for downloading, extracting, and loading a model
        based on a unique identifier.

        Args:
            model_uuid (str): The unique identifier for the model to be loaded.

        Returns:
            Model: The loaded model instance.
        """
        pass

    @abc.abstractmethod
    def create_ai_model(self,
                        model_path: str,
                        training_device: str,
                        training_time: float,
                        monitoring_templates: Optional[dict] = None):
        """
        Abstract method to create and register a new AI model with the task. The model is created from
        the provided path, and relevant metadata such as the training device, time, and monitoring
        templates can be supplied.

        Args:
            model_path (str): Path to the model file.
            training_device (str): Device on which the model was trained (e.g., 'GPU', 'CPU').
            training_time (float): Duration of the model training process.
            monitoring_templates (Optional[dict]): Optional monitoring templates for the model.
        """
        pass

    @abc.abstractmethod
    def update_task_usage(self, cpu_hours: Optional[float] = None, gpu_hours: Optional[float] = None):
        """
        Abstract method to update the task resource usage. The resource usage is updated for both CPU
        and GPU hours based on the provided values.

        Args:
            cpu_hours (Optional[float]): Amount of CPU hours used for the task.
            gpu_hours (Optional[float]): Amount of GPU hours used for the task.
        """
        pass

    @abc.abstractmethod
    def update_monitoring_templates(self, model_uuid: str, monitoring_templates: dict):
        """
        Abstract method to update the monitoring templates for a given model. This method validates the
        templates and stores them in the associated monitoring service for the task.

        Args:
            model_uuid (str): UUID of the model for which to update the monitoring templates.
            monitoring_templates (dict): Dictionary containing the monitoring templates.
        """
        pass

    @abc.abstractmethod
    def update_task_status(self, code: str):
        """
        Abstract method that defines the interface for updating the status of a task based on a
        provided status code.

        Args:
            code (str): The status code that represents the new state or action for the task.
        """
        pass

    @abc.abstractmethod
    def get_services_settings(self) -> Dict[str, dict]:
        """
        Abstract method that defines the interface for retrieving the settings of the services
        associated with the current task.

        Implementations of this method should return a dictionary where each key is the service name and the value
        is another dictionary containing the service's specific settings.

        Returns:
            Dict[str, dict]: A dictionary mapping service names to their respective settings.
        """
        pass

    def update_service_status(self, service_type: ServiceType, code: str, details: Optional[dict] = None):
        """
        Updates the status of a specific service based on the provided service type. This method acts as a dispatcher
        that routes the status update to the corresponding service update method depending on the service type.
        Each service type has a dedicated update method that handles its specific status update logic.
        If the service type is not recognized, it raises a ValueError.

        Steps:
        1. Check the service_type and invoke the corresponding update method with the provided code and details.
        2. If service_type is not valid, raise a ValueError indicating an invalid service type.

        Args:
            service_type (ServiceType): The type of service whose status needs to be updated (e.g., inference, continual
                                        learning, active learning, monitoring, or testing).
            code (str): The status code representing the current status of the service.
            details (Optional[dict]): Additional optional details related to the status update.

        Raises:
            ValueError: If an invalid service type is provided.
        """
        if service_type == ServiceType.INFERENCE:
            self.update_inference_service_status(code=code, details=details)
        elif service_type == ServiceType.CONTINUAL_LEARNING:
            self.update_continual_learning_service_status(code=code, details=details)
        elif service_type == ServiceType.ACTIVE_LEARNING:
            self.update_active_learning_service_status(code=code, details=details)
        elif service_type == ServiceType.MONITORING:
            self.update_monitoring_service_status(code=code, details=details)
        elif service_type == ServiceType.TESTING:
            self.update_monitoring_service_status(code=code, details=details)
        else:
            raise ValueError(f'Invalid service type {service_type}')

    @abc.abstractmethod
    def update_inference_service_status(self, code: str, details: Optional[dict] = None):
        """
        Abstract method to update the status of the inference service. This method should be implemented in
        the subclasses to handle the specific status update logic for inference services.

        Args:
            code (str): The status code representing the current state of the inference service.
            details (Optional[dict]): Additional optional details relevant to the status update.
        """
        pass

    @abc.abstractmethod
    def update_continual_learning_service_status(self, code: str, details: Optional[dict] = None):
        """
        Abstract method to update the status of the continual learning service. This method should be implemented in
        the subclasses to handle the specific status update logic for continual learning services.

        Args:
            code (str): The status code representing the current state of the continual learning service.
            details (Optional[dict]): Additional optional details relevant to the status update.
        """
        pass

    @abc.abstractmethod
    def update_active_learning_service_status(self, code: str, details: Optional[dict] = None):
        """
       Abstract method to update the status of the active learning service. This method should be implemented in
       the subclasses to handle the specific status update logic for active learning services.

       Args:
           code (str): The status code representing the current state of the active learning service.
           details (Optional[dict]): Additional optional details relevant to the status update.
       """
        pass

    @abc.abstractmethod
    def update_monitoring_service_status(self, code: str, details: Optional[dict] = None):
        """
        Abstract method to update the status of the monitoring service. This method should be implemented in
        the subclasses to handle the specific status update logic for monitoring services.

        Args:
            code (str): The status code representing the current state of the monitoring service.
            details (Optional[dict]): Additional optional details relevant to the status update.
        """
        pass

    @abc.abstractmethod
    def update_testing_service_status(self, code: str, details: Optional[dict] = None):
        """
        Abstract method to update the status of the testing service. This method should be implemented in
        the subclasses to handle the specific status update logic for testing services.

        Args:
            code (str): The status code representing the current state of the testing service.
            details (Optional[dict]): Additional optional details relevant to the status update.
        """
        pass

    @abc.abstractmethod
    def deploy_model(self, model_uuid: str, environment: AIEnvironment):
        """
        Abstract method to deploy a model to the specified environment (production, testing, etc.). The
        deployment process involves updating the environment with the new model version and ensuring that
        the task is in an active state before deployment.

        Args:
            model_uuid (str): UUID of the model to be deployed.
            environment (AIEnvironment): The environment to which the model will be deployed.
        """
        pass

    @abc.abstractmethod
    def predict(self, environment: AIEnvironment, data: List[dict]) -> List[dict]:
        """
        Abstract method that defines the interface for making predictions based on provided data
        within a specified AI environment.

        Args:
            environment (AIEnvironment): The environment in which the prediction will be made.
            data (List[dict]): A list of input data, where each dictionary represents an example to be predicted.

        Returns:
            List[dict]: A list of dictionaries representing the predicted results for each input data.
        """
        pass


class LocalEngineWorker(EngineWorker):
    """
    A concrete implementation of the `EngineWorker` abstract class. `LocalEngineWorker` is responsible for managing
    tasks, models, and files in a local or on-premises environment. It interacts with the task database
    and manages the lifecycle of AI models, including versioning, training, and deployment.

    This class also handles file storage (local or S3) and provides methods to update resource usage,
    service statuses, and monitoring templates. It performs task schema retrieval, labeled example
    extraction, and category downloading.

    The class uses a continual learning service and the task database for operations like file
    downloading, task updates, and model deployments. The engine assumes the presence of a task and
    continual learning service for handling these actions.
    """

    class _LocalModelManager:
        """
        Singleton class that manages the loading, caching, and memory handling of AI models in a local environment.
        Ensures only necessary models are loaded, and handles memory constraints by unloading less-used models when
        needed.

        This class also manages local task environments and the model cache to optimize performance in terms of
        memory usage and access times.
        """
        _instance = None
        _instance_lock = threading.Lock()

        def __init__(self, max_memory_usage=0.8):
            """
            Initializes the LocalModelManager with a specified maximum memory usage threshold. Manages models and their
            usage times, ensuring memory constraints are respected.

            Args:
                max_memory_usage (float): Maximum memory usage allowed before models are unloaded (default is 80%).
            """
            self._models = {}  # {(task_id, AIEnvironment): model_instance}
            self._models_usage = {}  # {(task_id, AIEnvironment): last_access_time}
            self._max_memory_usage = max_memory_usage  # e.g., 80% of RAM
            self._lock = threading.Lock()

        @classmethod
        def get_instance(cls):
            """
            Singleton method to get the instance of the LocalModelManager. Ensures only one instance is created,
            providing thread-safe access to models.

            Returns:
                _LocalModelManager: The singleton instance of the model manager.
            """
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
            return cls._instance

        def load_model(self,
                       task_uuid: str,
                       environment: AIEnvironment,
                       model_uuid: Optional[str] = None) -> Optional[Model]:
            """
            Loads a model into memory, either by retrieving it from the model cache or loading it from
            the task database. Ensures memory availability before loading a new model. If a model is already loaded,
            it updates the usage timestamp to indicate recent access.

            Args:
                task_uuid (str): The unique identifier for the task associated with the model.
                environment (AIEnvironment): The deployment environment (e.g., production, testing).
                model_uuid (Optional[str]): The unique identifier for the model
                                            (if None, it will retrieve from the database).

            Returns:
                Optional[Model]: The loaded model instance or None if the model could not be retrieved.

            Assertions:
                - Memory availability is checked before loading the model.
            """
            with self._lock:
                key = (task_uuid, environment)

                if key in self._models:
                    # If a model is given, replace the current one.
                    # Otherwise, update usage and return
                    if model_uuid is None:
                        # Update usage
                        self._models_usage[key] = time.time()
                        # Model already loaded
                        return self._models[key]
                    else:
                        del self._models[key]

                # If there is no model to be loaded, get it from DB
                if model_uuid is None:
                    # Get the task
                    task_db_obj = TaskDB.get_from_id(id_value=task_uuid)
                    # Get the model_id from the given deployment environment
                    if environment == AIEnvironment.PRODUCTION:
                        model_id = task_db_obj.prod_model_id
                    elif environment == AIEnvironment.TESTING:
                        model_id = task_db_obj.test_model_id
                    else:
                        raise ValueError(f'Unknown environment type: {environment}')

                    # If model_id is None, the model is not deployed in the environment yet
                    # Otherwise, get the model_uuid
                    if model_id:
                        model_uuid = AIModelDB.get(model_id=model_id).uuid
                    else:
                        return None

                # Check memory before loading
                self._ensure_memory_available()

                # Load and store model
                engine = LocalEngineWorker(task_uuid=task_uuid)
                model = engine.load_model(model_uuid=model_uuid)
                self._models[key] = model

                # Update usage
                self._models_usage[key] = time.time()

                return model

        def predict(self, task_uuid: str, environment: AIEnvironment, data: List[dict]) -> List[dict]:
            """
            Performs a prediction using the loaded model for the specified task and environment. The method loads the
            model if it is not already in memory, then uses the model to predict based on the input data.

            Args:
                task_uuid (str): The unique identifier for the task.
                environment (AIEnvironment): The environment (e.g., production, testing) in which the task operates.
                data (List[dict]): A list of input data, where each dictionary represents an example to be predicted.

            Returns:
                List[dict]: The predicted output from the model.

            Assertions:
                - Ensures that the model is successfully loaded before making predictions.
            """
            model = self.load_model(task_uuid, environment)
            assert model is not None
            return model.predict(data=data)

        def _ensure_memory_available(self):
            """
            Ensures that enough memory is available to load a new model by checking memory usage. If memory usage is too
            high, it frees memory by unloading least recently used models.

            Raises:
                MemoryError: If memory cannot be freed enough to load the new model.
            """
            if not self._has_enough_memory():
                self._free_memory()

        def _has_enough_memory(self):
            """
            Checks if there is enough available memory (both system and GPU) to load a new model.

            Returns:
                bool: True if enough memory is available, False otherwise.
            """
            # Check both system and GPU memory
            mem = psutil.virtual_memory()
            if mem.percent >= self._max_memory_usage * 100:
                return False
            # ToDo: think about GPU memory
            # if torch.cuda.is_available():
            #     gpu_mem_free = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            #     if gpu_mem_free < self.required_gpu_memory:
            #         return False
            return True

        def _free_memory(self):
            """
            Frees memory by unloading the least recently used models until enough memory is available. Unloads models
            in order of least recent usage.

            Raises:
                MemoryError: If memory cannot be freed enough to load the new model.
            """
            # Unload least recently used models until there's enough memory
            sorted_models = sorted(self._models_usage.items(), key=lambda item: item[1])
            for task_id, environment in sorted_models:
                self._unload_model(task_id, environment)
                if self._has_enough_memory():
                    break
            if not self._has_enough_memory():
                raise MemoryError('Not enough memory to load the new model.')

        def _unload_model(self, task_uuid: str, environment: AIEnvironment):
            """
            Unloads a model from memory based on the task UUID and environment, freeing up system memory. The model's
            entry is removed from both the model cache and the usage log.

            Args:
                task_uuid (str): The unique identifier of the task associated with the model.
                environment (AIEnvironment): The environment in which the model was deployed.
            """
            key = (task_uuid, environment)
            model = self._models.pop(key, None)
            self._models_usage.pop(key, None)
            if model is not None:
                del model
                torch.cuda.empty_cache()

    def __init__(self, task_uuid: str):
        """
        Initializes the LocalEngineWorker with the given task UUID. This includes fetching the task from the
        database and ensuring the associated continual learning service is available.

        Args:
            task_uuid (str): The unique identifier for the task.
        """
        super().__init__(task_uuid=task_uuid)
        self._task_db_obj = TaskDB.get_from_id(id_value=self._task_uuid)
        self._cl_service = Service.filter_by_task_and_type(task_id=self._task_db_obj.task_id,
                                                           type_=ServiceType.CONTINUAL_LEARNING)
        assert self._cl_service is not None
        self._task = Task.get(agent=self._cl_service.client, db_object_or_id=self._task_db_obj)

    def get_task_schema(self) -> dict:
        """
        Retrieves the task schema for the current task from the continual learning service.
        The task schema includes information about the input and output elements, metadata,
        and additional task-specific configurations.

        Returns:
            dict: The task schema.
        """
        return self._task.dump_task_schema()

    def get_element_categories(self, task_schema: dict) -> dict:
        """
        Retrieves element categories based on the task schema. Element categories correspond to
        inputs, outputs, or metadata in the schema that are categorized. These categories are retrieved
        by querying the database and interacting with the continual learning service.

        Args:
            task_schema (dict): The schema of the task containing inputs, outputs, and metadata.

        Returns:
            dict: A dictionary containing categories for the elements, organized by element name.
        """
        categories = {}
        elements = task_schema.get('inputs', []) + task_schema.get('outputs', []) + task_schema.get('metadata', [])
        for el in elements:
            if el['type'] == 'category':
                element_db_obj = ElementDB.get_from_id(id_value=el['uuid'])
                element_resource_class = InputElement if element_db_obj.element_type == ElementType.INPUT \
                    else OutputElement
                element = element_resource_class.get(agent=self._cl_service.client,
                                                     db_object_or_id=element_db_obj,
                                                     parents=[self._task])
                category_db_objects = CategoryDB.query().filter_by(element_id=element_db_obj.element_id).all()
                category_resource_class = InputCategory if element_db_obj.element_type == ElementType.INPUT \
                    else OutputCategory
                categories[el['name']] = [
                    category_resource_class.get(agent=self._cl_service.client,
                                                db_object_or_id=x,
                                                parents=[self._task, element]).dump() for x in category_db_objects
                ]
        return categories

    def get_labeled_examples(self) -> list:
        """
        Retrieves a list of labeled examples for the current task from the database.
        The examples are required for training and evaluation tasks.

        Returns:
            list: A list of labeled examples associated with the task.
        """
        example_db_objects = ExampleDB.query().filter_by(task_id=self._task_db_obj.task_id,
                                                         labeling_status=LabelingStatus.LABELED).all()
        examples = Example.dump_batch(examples=example_db_objects, task=self._task)
        return examples

    def get_file(self, file_uuid: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Downloads a file associated with the task using its UUID. If an output path is provided,
        the file is saved there; otherwise, the file is loaded into memory. The method supports both
        local and S3-based storage backends.

        Steps:
        1. Fetch the file metadata from the database.
        2. If the file storage backend is S3, the file is downloaded from S3.
        3. Otherwise, the file is copied from the local storage.
        4. If no output path is specified, the file is loaded into memory and returned as bytes.

        Args:
            file_uuid (str): UUID of the file to retrieve.
            output_path (Optional[str]): Path to save the file. If None, the file is loaded into memory.

        Returns:
            bytes or None: Returns file content if no output path is provided, otherwise None.
        """
        if output_path is None:
            output_path_temp = tempfile.TemporaryDirectory()
            output_path = output_path_temp.name
        else:
            output_path_temp = None
        output_file = os.path.join(output_path, file_uuid)
        file_db_obj = TaskFileDB.get_from_id(id_value=file_uuid)
        file = TaskFile.get(agent=self._cl_service.client, db_object_or_id=file_db_obj, parents=[self._task])
        if get_file_storage_backend() == FileStorageBackend.S3:
            s3_config = get_s3_config()
            object_key = file.path()
            # Check whether the file exists in S3
            if 'Contents' not in s3_client().list_objects_v2(Bucket=s3_config['bucket'], Prefix=object_key):
                raise Exception(f'"{object_key}" not found in S3')
            s3_client().download_file(Bucket=s3_config['bucket'], Key=object_key, Filename=output_file)
        else:
            file_path = file.path()
            shutil.copyfile(file_path, output_file)

        if output_path_temp is not None:
            with open(output_file, 'rb') as f:
                byte_array = f.read()
            output_path_temp.cleanup()
            return byte_array

    def get_models(self) -> list:
        """
        Retrieves all AI models associated with the task from the database.
        Each model is fetched via the continual learning service.

        Returns:
            list: A list of models in serialized (dumped) format.
        """
        ai_model_db_objects = AIModelDB.query().filter_by(task_id=self._task_db_obj.task_id).all()
        ai_models = [
            AIModel.get(agent=self._cl_service.client, db_object_or_id=x, parents=[self._task], check_parents=False)
            for x in ai_model_db_objects
        ]

        return dump(ai_models)

    def get_model(self, model_uuid: str) -> dict:
        """
        Retrieves the details of a specific AI model by its UUID. The model is fetched
        from the database and returned in serialized format.

        Args:
            model_uuid (str): The UUID of the model to retrieve.

        Returns:
            dict: The model details in a dictionary.
        """
        ai_model_db_obj = AIModelDB.get_from_id(id_value=model_uuid)
        ai_model = AIModel.get(agent=self._cl_service.client,
                               db_object_or_id=ai_model_db_obj,
                               parents=[self._task],
                               check_parents=False)
        return ai_model.dump()

    def load_model(self, model_uuid: str) -> Model:
        """
        Downloads, extracts, and loads a model based on a unique identifier, and returns the loaded model instance.

        The function initiates by creating a temporary directory where the downloaded model will be stored.
        It then calls the `get_model_file` method to download the model into the temporary location. Afterward,
        the function attempts to extract the `model.pkl` file from the downloaded zip archive. If the file is found,
        it extracts the model, loads it using the `Model.load()` method, and finally cleans up the temporary directory.
        The function then returns the loaded model.

        Steps:
        1. Create a temporary directory to store the downloaded model file.
        2. Download the model into the temporary directory using the provided model UUID.
        3. Extract the 'model.pkl' file from the downloaded zip file.
        4. Load the extracted model using the `Model.load` method.
        5. Clean up the temporary directory and return the model instance.

        Args:
            model_uuid (str): The unique identifier for the model to be loaded.

        Returns:
            Model: The loaded model instance.
        """
        # Temp dir for the model
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_name = temp_dir.name

        # Get the file
        self.get_model_file(model_uuid=model_uuid, output_path=temp_dir_name)

        # Zip file
        out_zip_file = os.path.join(temp_dir_name, model_uuid)

        # Open the zip file and extract the specific file
        with zipfile.ZipFile(out_zip_file, 'r') as zipf:
            # Check if the file is in the zip archive
            if 'model.pkl' in zipf.namelist():
                zipf.extract('model.pkl', temp_dir_name)

        m = Model.load(os.path.join(temp_dir_name, 'model.pkl'))
        temp_dir.cleanup()
        return m

    def create_ai_model(self,
                        model_path: str,
                        training_device: str,
                        training_time: float,
                        monitoring_templates: Optional[dict] = None):
        """
        Creates a new AI model and registers it with the task. The method uploads the model file,
        generates a new version, and stores model metadata in the database. Optionally, it also
        updates the monitoring templates for the model.

        Steps:
        1. Upload the model file.
        2. Generate the next model version.
        3. Create the model entry in the database with its metadata.
        4. Optionally, update monitoring templates if provided.

        Args:
            model_path (str): Path to the model file to upload.
            training_device (str): The device (e.g., GPU, CPU) used to train the model.
            training_time (float): The time taken to train the model.
            monitoring_templates (Optional[dict]): Monitoring templates for the model, if any.
        """
        # Create file
        file_data = {'filename': model_path, 'size': os.stat(model_path).st_size, 'use_for': 'ai_model', 'type': None}
        file_data = TaskFileRequest().load(data=file_data)
        task_file = TaskFile.post(agent=self._cl_service.client, data=file_data, parents=[self._task])

        # Upload file
        if get_file_storage_backend() == FileStorageBackend.S3:
            s3_config = get_s3_config()
            s3_client().upload_file(Filename=model_path, Bucket=s3_config['bucket'], Key=task_file.path())
        else:
            os.makedirs(os.path.dirname(task_file.path()), exist_ok=True)
            shutil.copyfile(model_path, task_file.path())

        # Get next version
        version = self.get_next_model_version()

        # Create AI Model
        model_info = {
            'file': task_file.uuid(),
            'version': version,
            'training_device': training_device,
            'training_time': training_time
        }
        model_info = AIModelRequest().load(data=model_info)
        model = AIModel.post(agent=self._cl_service.client, data=model_info, parents=[self._task])

        # Update monitoring templates
        # ToDo: only if the AI model is put on production
        if monitoring_templates:
            # Add the AI model UUID first
            monitoring_templates['ai_model'] = model.uuid()
            self.update_monitoring_templates(model_uuid=model.uuid(), monitoring_templates=monitoring_templates)

    def update_task_usage(self, cpu_hours: Optional[float] = None, gpu_hours: Optional[float] = None):
        """
        Updates the resource usage for the task, such as CPU or GPU hours. If provided,
        the usage values are added to the task's resource usage quota.

        Args:
            cpu_hours (Optional[float]): The number of CPU hours to add.
            gpu_hours (Optional[float]): The number of GPU hours to add.
        """
        if cpu_hours is not None and cpu_hours > 0:
            self._task.update_quota_usage(name='cpu', delta=cpu_hours)

        if gpu_hours is not None and gpu_hours > 0:
            self._task.update_quota_usage(name='gpu', delta=gpu_hours)

    def update_monitoring_templates(self, model_uuid: str, monitoring_templates: dict):
        """
        Updates the monitoring templates for a specific AI model. The templates are verified and
        saved in the associated monitoring service.

        Args:
            model_uuid (str): The UUID of the model for which to update the monitoring templates.
            monitoring_templates (dict): The new monitoring templates to associate with the model.
        """
        try:
            self._verify_monitoring_templates(monitoring_templates=monitoring_templates,
                                              task_id=self._task_db_obj.task_id)
        except ValueError as e:
            return None
        # Save templates
        monitoring_service = Service.filter_by_task_and_type(task_id=self._task_db_obj.task_id,
                                                             type_=ServiceType.MONITORING)
        monitoring_service.data = monitoring_templates
        save_to_db(monitoring_service)

    @staticmethod
    def _update_service_status(service: Service, status: dict):
        """
        Updates the status of a given service by creating a new `Status` object from the provided dictionary
        and setting the service's status accordingly.

        Args:
            service (Service): The service whose status is being updated.
            status (dict): A dictionary representing the new status to be applied.
        """
        new_status = Status.from_dict(status)
        service.set_status(status=new_status)

    def _get_service(self, service_type: ServiceType) -> Service:
        """
        Retrieves a service associated with the current task and of the specified service type.

        Args:
            service_type (ServiceType): The type of service to retrieve (e.g., continual learning, monitoring).

        Returns:
            Service: The service that matches the task ID and service type.

        """
        return Service.filter_by_task_and_type(task_id=self._task_db_obj.task_id, type_=service_type)

    def update_task_status(self, code: str):
        """
        Updates the status of a task based on the provided status code.

        The method creates a new status object using the provided `code` by converting it
        into a dictionary format and passing it to the `Status.from_dict` method.
        The new status is then set for the task using the internal database object (`_task_db_obj`).

        Args:
            code (str): The status code that represents the new state or action for the task.
        """
        new_status = Status.from_dict({'code': code})
        self._task_db_obj.set_status(new_status)

    def get_services_settings(self) -> Dict[str, dict]:
        """
        Retrieves the settings of the services associated with the current task.

        This method queries the `Service` model to filter services by the task's ID. It then constructs
        a dictionary where the keys are the lowercase names of the service types, and the values are
        the respective settings of each service obtained by converting the service objects to a dictionary
        format.

        Returns:
            Dict[str, dict]: A dictionary mapping service names to their respective settings.
        """
        services = Service.filter_by_task(task_id=self._task_db_obj.task_id)
        service_settings = {service.type_.name.lower(): service.to_dict()['settings'] for service in services}
        return service_settings

    def update_inference_service_status(self, code: str, details: Optional[dict] = None):
        """
        Updates the status of the inference service associated with the current task.

        Args:
            code (str): The status code to be applied to the inference service.
            details (Optional[dict]): Optional additional information to include in the status update.
        """
        status = {'code': code}
        if details is not None:
            status['details'] = details
        LocalEngineWorker._update_service_status(service=self._get_service(ServiceType.INFERENCE), status=status)

    def update_continual_learning_service_status(self, code: str, details: Optional[dict] = None):
        """
        Updates the status of the continual learning service. Handles special cases for session start and end
        by logging the corresponding start or end datetime and updating trained examples.

        Important steps:
        1. Checks if the status code has changed and performs specific actions for training session start/end.
        2. Updates the service's status.

        Args:
            code (str): The new status code for the continual learning service.
            details (Optional[dict]): Optional additional information for the status update.
        """
        if code != self._cl_service.status['code']:
            # If finishing training session, register session end datetime and mark trained examples
            if self._cl_service.status['code'] == CL_TRAINING_STATUS_CODE and code.startswith('02'):
                self._cl_service.data = {
                    'last_end': datetime.utcnow().strftime(DATETIME_FORMAT),
                    **self._cl_service.data
                }
                # Mark all examples with `labeling_status="labeled" and activity_at < last_start` as `trained=True`
                last_start = datetime.strptime(self._cl_service.data['last_start'], DATETIME_FORMAT)
                trained_examples = (ExampleDB.query().filter(
                    sql_and(ExampleDB.task_id == self._task_db_obj.task_id,
                            ExampleDB.labeling_status == LabelingStatus.LABELED, ExampleDB.activity_at < last_start)))
                trained_examples.update({'trained': True})
                db_commit()
            # If starting training session, register session start datetime
            elif code == CL_TRAINING_STATUS_CODE:
                self._cl_service.data = {
                    'last_start': datetime.utcnow().strftime(DATETIME_FORMAT),
                    **self._cl_service.data
                }
                db_commit()

        status = {'code': code}
        if details is not None:
            status['details'] = details
        LocalEngineWorker._update_service_status(service=self._cl_service, status=status)

    def update_active_learning_service_status(self, code: str, details: Optional[dict] = None):
        """
        Updates the status of the active learning service associated with the current task.

        Args:
            code (str): The status code to be applied to the active learning service.
            details (Optional[dict]): Optional additional information to include in the status update.
        """
        status = {'code': code}
        if details is not None:
            status['details'] = details
        LocalEngineWorker._update_service_status(service=self._get_service(ServiceType.ACTIVE_LEARNING), status=status)

    def update_monitoring_service_status(self, code: str, details: Optional[dict] = None):
        """
        Updates the status of the monitoring service associated with the current task.

        Args:
            code (str): The status code to be applied to the monitoring service.
            details (Optional[dict]): Optional additional information to include in the status update.
        """
        status = {'code': code}
        if details is not None:
            status['details'] = details
        LocalEngineWorker._update_service_status(service=self._get_service(ServiceType.MONITORING), status=status)

    def update_testing_service_status(self, code: str, details: Optional[dict] = None):
        """
        Updates the status of the testing service associated with the current task.

        Args:
            code (str): The status code to be applied to the testing service.
            details (Optional[dict]): Optional additional information to include in the status update.
        """
        status = {'code': code}
        if details is not None:
            status['details'] = details
        LocalEngineWorker._update_service_status(service=self._get_service(ServiceType.TESTING), status=status)

    def deploy_model(self, model_uuid: str, environment: AIEnvironment):
        """
        Deploys a model to the specified environment by loading it using the local model manager.

        Important steps:
        1. Retrieve the instance of the local model manager.
        2. Load the specified model into the target environment using the manager.

        Args:
            model_uuid (str): The unique identifier for the model to deploy.
            environment (AIEnvironment): The environment where the model should be deployed.
        """
        model_manager = LocalEngineWorker._LocalModelManager.get_instance()
        model_manager.load_model(task_uuid=self._task_uuid, environment=environment, model_uuid=model_uuid)

    def predict(self, environment: AIEnvironment, data: List[dict]) -> List[dict]:
        """
        Makes predictions based on provided data within a specified AI environment.

        The method first prepares the data by downloading the necessary files into a temporary
        directory using the `download_data_files` method. Then, it retrieves the model manager
        instance from `LocalEngineWorker._LocalModelManager` and performs the predictions.
        After predictions are made, the temporary directory is cleaned up, and the predictions
        are returned.

        Steps:
        1. Create a temporary directory to store any required data files.
        2. Download the necessary data files using the `download_data_files` method.
        3. Retrieve the model manager instance to handle the prediction process.
        4. Use the model manager to predict the output based on the task UUID, environment, and data.
        5. Clean up the temporary directory and return the prediction results.

        Args:
            environment (AIEnvironment): The environment in which the prediction will be made.
            data (List[dict]): A list of input data, where each dictionary represents an example to be predicted.

        Returns:
            List[dict]: A list of dictionaries representing the predicted results for each input data.
        """
        # Prepare data (get files)
        tempdir = tempfile.TemporaryDirectory()
        self.download_data_files(task_schema=self.get_task_schema(), data=data, output_path=tempdir.name)

        # Get model manager and predict
        model_manager = LocalEngineWorker._LocalModelManager.get_instance()
        predictions = model_manager.predict(task_uuid=self._task_uuid, environment=environment, data=data)

        # Clean up
        tempdir.cleanup()

        # Return predictions
        return predictions


def get_engine(engine_type: EngineType, task_uuid: str) -> EngineWorker:
    """
    Returns an engine instance based on the provided engine type.

    This function takes an engine type and a task UUID to determine which
    engine to instantiate.

    Args:
        engine_type (EngineType): The type of engine to instantiate.
        task_uuid (str): The UUID associated with the task for which the engine is being instantiated.

    Returns:
        EngineWorker: An instance of the corresponding engine.

    Raises:
        NotImplementedError: If the engine type is not implemented yet.
        ValueError: If the engine type is not recognized.
    """
    if engine_type == EngineType.LOCAL:
        return LocalEngineWorker(task_uuid=task_uuid)
    elif engine_type == EngineType.CLOUD:
        raise NotImplementedError('Cloud engine not implemented yet.')
    elif engine_type == EngineType.EDGE:
        raise NotImplementedError('Edge engine not implemented yet.')
    else:
        raise ValueError(f'Unknown engine type: {engine_type}')
