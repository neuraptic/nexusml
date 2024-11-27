# TODO: Try to make this module independent from `nexusml.api`

import copy
from typing import Dict, List, TYPE_CHECKING
import uuid

import numpy as np
import pandas as pd

from nexusml.api.resources import ResourceNotFoundError
from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.services import MonitoringServiceTemplatesSchema
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredictionDB
from nexusml.database.buffers import MonBufferItemDB
from nexusml.database.organizations import ClientDB
from nexusml.database.services import Service as ServiceDB
from nexusml.database.tasks import ElementDB
from nexusml.engine.services.base import Service
from nexusml.engine.services.utils import send_email_notification
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import ServiceType

if TYPE_CHECKING:
    from nexusml.engine.buffers import MonBuffer


class MonitoringService(Service):
    """
    Monitoring Service.

    Implemented metrics:
        - Out-of-distribution (OOD) predictions
    """
    OOD_CAT_MIN_TH = 1  # TODO: set minimum anomaly score threshold for categorical outputs
    OOD_CAT_MAX_TH = 3  # TODO: set maximum anomaly score threshold for categorical outputs
    OOD_NUM_MIN_TH = 2  # minimum anomaly score threshold for numerical outputs
    OOD_NUM_MAX_TH = 3.5  # maximum anomaly score threshold for numerical outputs

    def __init__(self, buffer: 'MonBuffer', refresh_interval: int, ood_min_sample: int, ood_sensitivity: float,
                 ood_smoothing: float):
        """
        Constructor.

        Args:
            buffer (MonBuffer): buffer from which predictions will be read
            refresh_interval (int): interval in which metrics are refreshed (in number of predictions). Setting it to 1
                                    forces metrics to be refreshed each time a new prediction is saved to the buffer.
            ood_min_sample (int): minimum number of predictions required for OOD detection
            ood_sensitivity (float): sensitivity to OOD predictions (value between 0 and 1)
            ood_smoothing (float): smoothing factor for OOD detection (value between 0 and 1). Low values result in less
                                   smoothing and thus a high responsiveness to variations in predictions.
        """
        super().__init__(buffer=buffer, service_type=ServiceType.MONITORING)
        self._refresh_interval = refresh_interval
        self._ood_min_sample = ood_min_sample
        assert 0 <= ood_sensitivity <= 1
        self._ood_sensitivity = ood_sensitivity
        assert 0 <= ood_smoothing <= 1
        self._ood_smoothing = ood_smoothing
        self._templates = dict()  # posterior distribution template of each output
        self.service: ServiceDB = ServiceDB.filter_by_task_and_type(task_id=self.task().task_id,
                                                                    type_=ServiceType.MONITORING)
        self.client_db_agent: ClientDB = ClientDB.get(client_id=self.service.client_id)

    def service_name(self) -> str:
        """
        Returns the name of the service, 'monitoring'.

        Returns:
            str: The service name.
        """
        return 'monitoring'

    def check_if_enough_samples(self, items: List[MonBufferItemDB]) -> bool:
        """
        Checks if there are enough samples in the buffer to perform OOD detection based on the set refresh interval.

        Args:
            items (List[MonBufferItemDB]): List of items in the buffer.

        Returns:
            bool: True if there are enough samples, False otherwise.
        """
        return len(items) >= self._refresh_interval

    ##################################################
    # Out-of-distribution (OOD) prediction detection #
    ##################################################

    def detect_ood_predictions(self) -> Dict[str, float]:
        """
        Detects out-of-distribution predictions from the buffer. It checks the anomaly scores of both categorical
        and numerical outputs, using different methods to calculate these scores. If the anomaly score exceeds a
        threshold, an alert is sent.

        Steps:
        1. Load and verify the templates for the output elements.
        2. Retrieve predictions from the buffer and calculate their anomaly scores.
        3. If the scores exceed a threshold, an email notification is sent.

        Returns:
            Dict[str, float]: Dictionary with detected out-of-distribution outputs and their average anomaly scores,
                              with the element UUID as the key and the score as the value.
        Raises:
            ValueError: If the templates do not match the current task schema or the running AI model.
        """
        # Check templates if available. Otherwise, download them.
        if self._templates:
            # Check templates' integrity (must match current task schema)
            try:
                self.verify_templates(templates=self._templates, task_id=self.task().task_id)
            except ValueError:
                self.refresh_templates()
        else:
            # If templates were not downloaded, do it now
            self.refresh_templates()

        # Check there is enough predictions
        buffer_items: list = self.buffer().read()
        if len(buffer_items) < self._ood_min_sample:
            return dict()

        # Load AI model metadata
        ai_model_db_obj: AIModelDB = AIModelDB.get(model_id=self.task().prod_model_id)
        if ai_model_db_obj is None:
            raise ResourceNotFoundError('No deployed AI model found')

        if self._templates['ai_model'] != ai_model_db_obj.uuid:
            raise ValueError('Templates do not correspond to the AI model running in production')

        # Run detection
        ood_numerical_outputs = self._detect_ood_numbers()
        ood_categorical_outputs = self._detect_ood_categories()
        if not set(ood_numerical_outputs.keys()).intersection(set(ood_categorical_outputs.keys())) == set():
            raise Exception('OOD numerical output and OOD categorical outputs intersection is not empty')

        # Clear buffer
        self.buffer().clear()

        return {**ood_numerical_outputs, **ood_categorical_outputs}

    def _detect_ood_categories(self) -> Dict[str, float]:
        """
        Detects out-of-distribution predictions for categorical outputs using the KL Divergence method. The method
        compares the predicted category scores to a pre-built posterior distribution template.

        Returns:
            Dict[str, float]: Dictionary with detected OOD categorical outputs and their average anomaly scores,
                              with the element UUID as the key and the score as the value.

        Raises:
            TypeError: If the template format is invalid.
        """
        # Prepare templates
        monitoring_template: dict = MonitoringServiceTemplatesSchema().load(self.service.data)
        elements: list = [ElementDB.get_from_id(element['element']) for element in monitoring_template['outputs']]

        categorical_outputs: set = {x.uuid for x in elements if x.value_type == ElementValueType.CATEGORY}

        templates: dict = dict()
        for elem_template in self._templates['outputs']:
            elem_uuid: str = elem_template['element']
            if elem_uuid not in categorical_outputs:
                continue
            template = elem_template['template']
            if not isinstance(template, list):
                raise TypeError('Template is not a list')
            cats_templates = dict()
            for cat_templates in template:
                cat_uuid: str = cat_templates['category']
                cat_template = cat_templates['template']
                cats_templates[cat_uuid] = dict()
                for cat_means in cat_template:
                    cats_templates[cat_uuid][cat_means['category']] = cat_means['mean']
            sorted_cat_templates: list = []
            for _, cat_template in sorted(cats_templates.items()):
                sorted_cat_templates.append([mean for _, mean in sorted(cat_template.items())])
            templates[elem_uuid] = np.array(sorted_cat_templates)

        # Get predicted scores for each category
        buffer_items: list = self.buffer().read()
        pred_scores = dict()
        elem_categories = dict()
        for buffer_item in buffer_items:
            prediction_db_obj: PredictionDB = PredictionDB.query().filter_by(
                prediction_id=buffer_item.prediction_id).first()

            prediction: PredictionLog = PredictionLog.get(agent=self.client_db_agent,
                                                          db_object_or_id=prediction_db_obj,
                                                          check_permissions=False)

            prediction_data: dict = prediction.dump(serialize=False)

            for pred in prediction_data['outputs']:
                # Get the element to which the prediction refers
                element: ElementDB = ElementDB.get_from_id(id_value=pred['element'], parent=self.task())
                if element.value_type != ElementValueType.CATEGORY:
                    continue
                # Ensure all predictions refer to the same categories
                pred_cat_uuids = set(pred['value']['scores'].keys())
                if pred['element'] not in elem_categories:
                    elem_categories[pred['element']] = pred_cat_uuids
                if pred_cat_uuids != elem_categories[pred['element']]:
                    raise ValueError("Predicted categories don't match")
                # Get all category scores and convert dictionaries to numpy arrays
                # Note: in Python 3.6+, dictionary order is guaranteed to be insertion order
                if element.uuid not in pred_scores:
                    pred_scores[element.uuid] = []
                scores = np.array([x[1] for x in sorted(pred['value']['scores'].items())])
                pred_scores[element.uuid].append(scores)

        # Normalize scores if they don't sum to 1
        norm_scores = {elem: self.softmax(np.array(scores)) for elem, scores in pred_scores.items()}

        # Compute the anomaly score of each prediction for each output
        # (KL Divergence between predicted scores and the corresponding posterior distribution template)
        elem_anomaly_scores: dict = dict()
        for elem, scores in norm_scores.items():
            pdt = templates[elem]
            elem_anomaly_scores[elem] = np.sum(scores * np.log(scores / pdt[np.argmax(scores, axis=1)]), axis=1)

        # Process anomaly scores
        threshold = (self.OOD_CAT_MAX_TH - self.OOD_CAT_MIN_TH) * self._ood_sensitivity + self.OOD_CAT_MIN_TH
        return self._process_anomaly_scores(anomaly_scores=elem_anomaly_scores, threshold=threshold)

    def _detect_ood_numbers(self) -> Dict[str, float]:
        """
        Detects out-of-distribution predictions for numerical outputs using the z-score method. The method compares
        the predicted values to the mean and standard deviation defined in the template.

        Returns:
            Dict[str, float]: Dictionary with detected OOD numerical outputs and their average anomaly scores,
                              with the element UUID as the key and the score as the value.

        Raises:
            TypeError: If the template format is invalid.
        """
        # Prepare templates
        monitoring_template: MonitoringServiceTemplatesSchema = MonitoringServiceTemplatesSchema().load(
            self.service.data)
        elem_list: list = [ElementDB.get_from_id(element['element']) for element in monitoring_template['outputs']]
        numerical_outputs: set = {
            elem.uuid for elem in elem_list if elem.value_type in [ElementValueType.INTEGER, ElementValueType.FLOAT]
        }
        templates: dict = dict()
        for elem_template in self._templates['outputs']:
            elem_uuid: str = elem_template['element']
            if elem_uuid not in numerical_outputs:
                continue
            template: dict = elem_template['template']
            if not isinstance(template, dict) or not set(template.keys()) == {'mean', 'std'}:
                raise TypeError(f'Template is not dict or have more keys than mean and std - template: {template}')
            templates[elem_uuid] = template

        # Get predicted values
        pred_values: dict = dict()
        buffer_items: list = self.buffer().read()
        for buffer_item in buffer_items:
            prediction_db_obj: PredictionDB = PredictionDB.query().filter_by(
                prediction_id=buffer_item.prediction_id).first()
            prediction: PredictionLog = PredictionLog.get(agent=self.client_db_agent,
                                                          db_object_or_id=prediction_db_obj,
                                                          check_permissions=False)  # TODO: Check this
            prediction_data: dict = prediction.dump(serialize=False)

            for pred in prediction_data['outputs']:
                # Get the element to which the prediction refers
                element: ElementDB = ElementDB.get_from_id(id_value=pred['element'], parent=self.task())
                if (element.element_type != ElementType.OUTPUT or
                        element.value_type not in [ElementValueType.INTEGER, ElementValueType.FLOAT]):
                    continue
                # Save predicted value
                element_uuid: str = element.uuid
                if element_uuid not in pred_values:
                    pred_values[element_uuid] = []
                assert isinstance(pred['value'], float)
                pred_values[element_uuid].append(pred['value'])
        pred_values = {elem: np.array(values) for elem, values in pred_values.items()}
        # Compute the anomaly score (z-score) of each predicted value
        elem_anomaly_scores = {
            elem: (values - templates[elem]['mean']) / templates[elem]['std'] for elem, values in pred_values.items()
        }

        # Process anomaly scores
        threshold = (self.OOD_NUM_MAX_TH - self.OOD_NUM_MIN_TH) * self._ood_sensitivity + self.OOD_NUM_MIN_TH
        return self._process_anomaly_scores(anomaly_scores=elem_anomaly_scores, threshold=threshold)

    def _process_anomaly_scores(self, anomaly_scores: Dict[str, np.ndarray], threshold: float) -> Dict[str, float]:
        """
        Processes the anomaly scores and sends a notification if any scores exceed the given threshold.

        Args:
            anomaly_scores (Dict[str, np.ndarray]): Dictionary of anomaly scores for each output element.
            threshold (float): Anomaly score threshold.

        Returns:
            Dict[str, float]: Dictionary with detected out-of-distribution outputs and their average anomaly scores,
                              with the element UUID as the key and the score as the value.
        """
        avg_anomaly_scores = {elem: self.ema(data=scores) for elem, scores in anomaly_scores.items()}

        # Filter the outputs having an average anomaly score greater than the specified threshold
        ood = {elem: avg_score for elem, avg_score in avg_anomaly_scores.items() if avg_score >= threshold}
        if ood:
            msg = ''
            for elem_uuid in ood.keys():
                elem: ElementDB = ElementDB.get_from_id(id_value=elem_uuid, parent=self.task())
                msg += f'<li>{elem.name} ({elem.uuid})</li>'
            notification = {'message': f'Anomalies detected in the following outputs:<ul>{msg}</ul>'}

            task: Task = Task.get(agent=self.client_db_agent, db_object_or_id=self.task())

            send_email_notification(task=task, service_type=ServiceType.MONITORING, payload=notification)

        return ood

    def refresh_templates(self):
        """
        Downloads and verifies the templates used for out-of-distribution detection. The templates define the
        distribution of values predicted by the model during training for each output element.
        """
        # TODO: Check why this is not getting service data
        service: ServiceDB = ServiceDB.filter_by_task_and_type(task_id=self.task().task_id,
                                                               type_=ServiceType.MONITORING)
        service_data: dict = service.data
        templates = MonitoringServiceTemplatesSchema().load(service_data)
        self.verify_templates(templates=templates, task_id=self.task().task_id)
        self._templates = copy.deepcopy(templates)

    @staticmethod
    def verify_templates(templates: dict, task_id: int):
        """
        Verifies the format and consistency of the templates used for out-of-distribution detection. Ensures that
        the templates contain valid values for numerical and categorical outputs.

        Args:
            templates (dict): The dictionary of templates for each output.
            task_id (int): The task ID for which the templates are being verified.

        Raises:
            ValueError: If the template format is invalid or inconsistent.

        Expected format:

        ```
        {
            "ai_model": <model_uuid>,
            "outputs": [
                {
                    "element": <element_uuid>,
                    "template": {
                        "mean": <float>,
                        "std": <float>,
                    }
                },
                {
                    "element": <element_uuid>,
                    "template": [
                        {
                            "category": <category_uuid>,
                            "template": [
                                {
                                    "category": <category_uuid>,
                                    "mean": <float>,
                                },
                                {
                                    "category": <category_uuid>,
                                    "mean": <float>,
                                },
                                ...
                            ]
                        },
                        {
                            "category": <category_uuid>,
                            "template": [
                                ...
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
        ```

        Note: in categorical outputs, each category has its own template consisting of the mean softmax probability
              for each category (called "posterior distribution template"). The template of a category should be
              computed taking only the examples in which the model predicted that category (i.e., its score was the
              highest one among all the categories).
        """

        if not templates:
            raise ValueError('No templates found')
        if not (isinstance(templates, dict) and isinstance(templates['outputs'], list)):
            raise ValueError('Invalid templates')

        numerical_outputs: set = set()
        categorical_outputs: set = set()
        for element in ElementDB.query().filter_by(task_id=task_id, element_type=ElementType.OUTPUT).all():
            if element.value_type in [ElementValueType.INTEGER, ElementValueType.FLOAT]:
                numerical_outputs.add(element.uuid)
            elif element.value_type == ElementValueType.CATEGORY:
                categorical_outputs.add(element.uuid)

        all_outputs = numerical_outputs.union(categorical_outputs)

        errors = []
        for elem_template in templates['outputs']:  # for the moment, only output elements' templates are supported
            if not isinstance(elem_template, dict) or set(elem_template.keys()) != {'element', 'template'}:
                errors.append(f'Invalid template format: "{elem_template}"')
            # Verify element
            elem_uuid = elem_template['element']
            if elem_uuid not in all_outputs:
                errors.append(f'Output element not found: "{elem_uuid}"')
            # Verify template
            template = elem_template['template']
            try:
                # Numerical output
                if elem_uuid in numerical_outputs:
                    if not isinstance(template, dict) or set(template.keys()) != {'mean', 'std'}:
                        raise ValueError()
                    if not isinstance(template['mean'], float):
                        raise ValueError()
                    if not isinstance(template['std'], float):
                        raise ValueError()
                # Categorical output
                else:
                    assert elem_uuid in categorical_outputs
                    if not isinstance(template, list):
                        raise ValueError()
                    # Each category has its own template
                    for cat_template in template:
                        if not isinstance(cat_template, dict) or set(cat_template.keys()) != {'category', 'template'}:
                            raise ValueError()
                        uuid.UUID(cat_template['category'])  # check UUID format
                        pass  # TODO: verify the category exists
                        # The template consists of the mean softmax probability for each category
                        # ("posterior distribution template"), otherwise the KL Divergence couldn't be computed.
                        mean_sum = 0
                        for cat_mean in cat_template['template']:
                            if not isinstance(cat_mean, dict) or set(cat_mean.keys()) != {'category', 'mean'}:
                                raise ValueError()
                            uuid.UUID(cat_mean['category'])  # check UUID format
                            pass  # TODO: verify the category exists
                            if not isinstance(cat_mean['mean'], float):
                                raise ValueError()
                            mean_sum += cat_mean['mean']
                        if not np.isclose(mean_sum, 1):
                            raise ValueError()
            except ValueError:
                errors.append(f'Wrong value for element "{elem_uuid}": "{template}"')

        if errors:
            e = '\n\t- '.join(errors)
            raise ValueError(f'Errors found in templates:{e}')

    @staticmethod
    def softmax(x: np.array) -> np.array:
        """
        Computes the softmax of a given 2-D array, ensuring that all rows sum to 1. Softmax is used to normalize
        predicted scores for categorical outputs.

        Args:
            x (np.array): A 2-D array where rows represent observations and columns represent the predicted scores
                          for each category.

        Returns:
            np.array: A 2-D array where all rows sum to 1.
        """
        if np.allclose(x.sum(axis=1), 1):
            return x
        x_max = np.amax(x, axis=1, keepdims=True)
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)

    def ema(self, data: np.array) -> float:
        """
        Computes the Exponential Moving Average (EMA) of the given data array. EMA is used to smooth the anomaly
        scores and give more weight to recent predictions.

        Args:
            data (np.array): Array of anomaly scores.

        Returns:
            float: The Exponential Moving Average of the anomaly scores.
        """
        return pd.DataFrame(data).ewm(alpha=1 - self._ood_smoothing).mean().mean().mean()
