from unittest.mock import MagicMock

import numpy as np
import pytest

from nexusml.api.buffers import MonBuffer
from nexusml.database.organizations import ClientDB
from nexusml.database.services import Service
from nexusml.engine.services.monitoring import ElementDB
from nexusml.engine.services.monitoring import MonitoringService
from nexusml.engine.services.monitoring import PredictionLog
from nexusml.engine.services.monitoring import ResourceNotFoundError
from nexusml.engine.services.monitoring import Task


class TestMonitoringService:
    """
    Test methods cover various aspects of the MonitoringService's functionality,
    including detecting Out-Of-Distribution (OOD) predictions for both numeric and
    categorical outputs, processing anomaly scores, and refreshing templates.
    """

    @pytest.fixture
    def monitoring_service_instance(self, mocker) -> MonitoringService:
        """
        Fixture to create a mock instance of MonitoringService.

        Returns:
            MonitoringService: An instance of MonitoringService configured with mock dependencies.
        """
        # Create a mock instance of MonitoringService
        refresh_interval = 10
        ood_min_sample = 1
        ood_sensitivity = 0.5
        ood_smoothing = 0.3

        mock_buffer = mocker.MagicMock(spec=MonBuffer)
        mock_service_db = mocker.MagicMock(spec=Service)
        mock_client_db_agent = mocker.MagicMock(spec=ClientDB)

        mocker.patch('nexusml.engine.services.monitoring.ServiceDB.filter_by_task_and_type',
                     return_value=mock_service_db)
        mocker.patch('nexusml.engine.services.monitoring.ClientDB.get', return_value=mock_client_db_agent)

        service = MonitoringService(buffer=mock_buffer,
                                    refresh_interval=refresh_interval,
                                    ood_min_sample=ood_min_sample,
                                    ood_sensitivity=ood_sensitivity,
                                    ood_smoothing=ood_smoothing)

        return service

    @pytest.fixture
    def general_template(self) -> dict:
        general_template: dict = {
            'outputs': [{
                'element':
                    'elem_uuid_1',
                'template': [{
                    'category': 'cat_uuid_1',
                    'template': [{
                        'category': 'cat_uuid_1',
                        'mean': 0.1
                    }, {
                        'category': 'cat_uuid_2',
                        'mean': 0.9
                    }]
                }, {
                    'category': 'cat_uuid_2',
                    'template': [{
                        'category': 'cat_uuid_1',
                        'mean': 0.8
                    }, {
                        'category': 'cat_uuid_2',
                        'mean': 0.2
                    }]
                }]
            }]
        }

        return general_template

    @pytest.fixture()
    def mock_monitoring_service_template(self, mocker):
        mock_service_schema: MagicMock = MagicMock()
        mock_service_schema.load.return_value = MagicMock()
        return mocker.patch('nexusml.engine.services.monitoring.MonitoringServiceTemplatesSchema',
                            return_value=mock_service_schema)

    @pytest.fixture
    def mock_prediction_log(self, mocker):
        return mocker.patch.object(PredictionLog, 'get', return_value=MagicMock())

    class TestDetectOODPredictions:
        """
        Tests cover scenarios where predictions are detected successfully, templates
        need refreshing, no AI model is found, and mismatched templates raise errors.
        """

        @pytest.fixture
        def mock_ai_model_db(self, mocker):
            return mocker.patch('nexusml.engine.services.monitoring.AIModelDB')

        @pytest.fixture
        def mock_task(self, mocker):
            return mocker.patch.object(MonitoringService, 'task')

        @pytest.fixture
        def mock_verify_templates(self, mocker):
            return mocker.patch.object(MonitoringService, 'verify_templates')

        @pytest.fixture
        def mock_refresh_templates(self, mocker):
            return mocker.patch.object(MonitoringService, 'refresh_templates')

        @pytest.fixture
        def mock_detect_ood_numbers(self, mocker):
            return mocker.patch.object(MonitoringService, '_detect_ood_numbers')

        @pytest.fixture
        def mock_detect_ood_categories(self, mocker):
            return mocker.patch.object(MonitoringService, '_detect_ood_categories')

        @pytest.fixture
        def mock_buffer(self, mocker):
            mock_buffer_ = mocker.patch.object(MonitoringService, 'buffer')
            buffer_instance = MagicMock()
            buffer_instance.read.return_value: list = [MagicMock()]
            mock_buffer_.return_value = buffer_instance
            return mock_buffer_

        def test_detect_ood_predictions(self, mock_ai_model_db, mock_task, mock_verify_templates,
                                        mock_refresh_templates, mock_detect_ood_numbers, mock_detect_ood_categories,
                                        mock_buffer, monitoring_service_instance):
            """
            Tests the detection of out-of-distribution predictions. Verifies that number and category
            predictions are successfully detected, and buffer is cleared afterwards.

            Test Steps:
            1. Mock necessary objects and methods including AI model, task instance, and detection functions.
            2. Set up mock return values for detection functions and method calls.
            3. Call the method `detect_ood_predictions` on `monitoring_service_instance`.

            Assertions:
            - `mock_refresh_templates` should not be called.
            - `mock_detect_ood_numbers` should be called exactly once.
            - `mock_detect_ood_categories` should be called exactly once.
            - `buffer_instance.clear` should be called exactly once.
            - The returned result should match `{'num_output_uuid': 0.8, 'cat_output_uuid': 0.9}`.

            """
            # Mock objects and methods
            monitoring_service_instance._templates = {'ai_model': 'mock_model_uuid'}
            task_instance = MagicMock()
            mock_task.return_value = task_instance
            task_instance.schema = {'mock': 'schema'}
            task_instance.uuid = 'task_uuid'
            task_instance.db_object.return_value.prod_model_id = 'prod_model_id'
            ai_model_db_obj = MagicMock()
            ai_model_db_obj.uuid = 'mock_model_uuid'
            mock_ai_model_db.get.return_value = ai_model_db_obj

            mock_detect_ood_numbers.return_value = {'num_output_uuid': 0.8}
            mock_detect_ood_categories.return_value = {'cat_output_uuid': 0.9}

            buffer_instance = mock_buffer.return_value

            # Call the method
            result = monitoring_service_instance.detect_ood_predictions()

            # Assertions
            mock_refresh_templates.assert_not_called()
            mock_detect_ood_numbers.assert_called_once()
            mock_detect_ood_categories.assert_called_once()
            buffer_instance.clear.assert_called_once()

            assert result == {'num_output_uuid': 0.8, 'cat_output_uuid': 0.9}

        def test_detect_ood_predictions_refresh_templates(self, mock_ai_model_db, mock_task, mock_verify_templates,
                                                          mock_refresh_templates, mock_detect_ood_numbers,
                                                          mock_detect_ood_categories, mock_buffer,
                                                          monitoring_service_instance):
            """
            Tests the detection of out-of-distribution predictions when templates need refreshing.

            Test Steps:
            1. Mock necessary objects and methods including AI model, task instance, and detection functions.
            2. Set up mock return values for detection functions and method calls.
            3. Call the method `detect_ood_predictions` on `monitoring_service_instance`.

            Assertions:
            - `mock_refresh_templates` should be called exactly once.
            - `mock_verify_templates` should not be called.
            - `mock_detect_ood_numbers` should be called exactly once.
            - `mock_detect_ood_categories` should be called exactly once.
            - `buffer_instance.clear` should be called exactly once.
            - The returned result should match `{'num_output_uuid': 0.8, 'cat_output_uuid': 0.9}`.

            """
            # Mock objects and methods
            monitoring_service_instance._templates = None
            new_template: dict = {'ai_model': 'mock_model_uuid'}
            mock_refresh_templates.side_effect = lambda: setattr(monitoring_service_instance, '_templates', new_template
                                                                )
            task_instance = MagicMock()
            mock_task.return_value = task_instance
            task_instance.schema = {'mock': 'schema'}
            task_instance.uuid = 'task_uuid'
            task_instance.db_object.return_value.prod_model_id = 'prod_model_id'

            ai_model_db_obj = MagicMock()
            ai_model_db_obj.uuid = 'mock_model_uuid'
            mock_ai_model_db.get.return_value = ai_model_db_obj

            mock_detect_ood_numbers.return_value = {'num_output_uuid': 0.8}
            mock_detect_ood_categories.return_value = {'cat_output_uuid': 0.9}

            # Call the method
            result = monitoring_service_instance.detect_ood_predictions()

            # Assertions
            mock_refresh_templates.assert_called_once()
            mock_verify_templates.assert_not_called()
            mock_detect_ood_numbers.assert_called_once()
            mock_detect_ood_categories.assert_called_once()
            mock_buffer.return_value.clear.assert_called_once()

            assert result == {'num_output_uuid': 0.8, 'cat_output_uuid': 0.9}

        def test_detect_ood_predictions_no_ai_model(self, mock_ai_model_db, mock_task, mock_verify_templates,
                                                    mock_refresh_templates, mock_detect_ood_numbers,
                                                    mock_detect_ood_categories, mock_buffer,
                                                    monitoring_service_instance):
            """
                Tests the detection of out-of-distribution predictions when no AI model is found.

                Test Steps:
                1. Mock necessary objects and methods including task instance and AI model database.
                2. Set up mock return value for AI model as None.
                3. Call the method `detect_ood_predictions` on `monitoring_service_instance`.

                Assertions:
                - A `ResourceNotFoundError` exception should be raised during the method call.
            """
            # Mock objects and methods
            monitoring_service_instance._templates = {'ai_model': 'mock_model_uuid'}
            task_instance = MagicMock()
            mock_task.return_value = task_instance
            task_instance.schema = {'mock': 'schema'}
            task_instance.uuid = 'task_uuid'
            task_instance.db_object.return_value.prod_model_id = 'prod_model_id'

            mock_ai_model_db.get.return_value = None

            # Call the method and assert exception
            with pytest.raises(ResourceNotFoundError):
                monitoring_service_instance.detect_ood_predictions()

        def test_detect_ood_predictions_templates_mismatch(self, mock_ai_model_db, mock_task, mock_verify_templates,
                                                           mock_refresh_templates, mock_detect_ood_numbers,
                                                           mock_detect_ood_categories, mock_buffer,
                                                           monitoring_service_instance):
            """
            Tests the detection of out-of-distribution predictions when templates do not match the AI model.

            Test Steps:
            1. Mock necessary objects and methods including AI model, task instance, and detection functions.
            2. Set up mock return values for AI model and method calls.
            3. Set the `_templates` attribute of `monitoring_service_instance` to a mismatched model.
            4. Call the method `detect_ood_predictions` on `monitoring_service_instance`.

            Assertions:
            - A `ValueError` exception with message 'Templates do not correspond to the AI model running in production'
              should be raised during the method call.
            """
            # Mock objects and methods
            monitoring_service_instance._templates = {'ai_model': 'different_model_uuid'}
            task_instance = MagicMock()
            mock_task.return_value = task_instance
            task_instance.schema = {'mock': 'schema'}
            task_instance.uuid = 'task_uuid'
            task_instance.db_object.return_value.prod_model_id = 'prod_model_id'

            ai_model_db_obj = MagicMock()
            ai_model_db_obj.uuid = 'mock_model_uuid'
            mock_ai_model_db.get.return_value = ai_model_db_obj

            # Call the method and assert exception
            with pytest.raises(ValueError, match='Templates do not correspond to the AI model running in production'):
                monitoring_service_instance.detect_ood_predictions()

    class TestDetectOODCategories:
        """
        Tests cover scenarios where categories are detected successfully, insufficient
        samples trigger early returns, and mismatches in predicted categories raise errors.
        """

        @pytest.fixture
        def mock_task(self, mocker):
            mock_task_ = mocker.patch.object(MonitoringService, 'task')
            mock_task_.return_value.get_element.return_value = {'type': 'category', 'uuid': 'elem_uuid_1'}
            return mock_task_

        @pytest.fixture
        def mock_buffer(self, mocker):
            return mocker.patch.object(MonitoringService, 'buffer')

        @pytest.fixture
        def mock_softmax(self, mocker):
            return mocker.patch.object(MonitoringService, 'softmax')

        @pytest.fixture
        def mock_process_anomaly_scores(self, mocker):
            return mocker.patch.object(MonitoringService, '_process_anomaly_scores')

        def test_detect_ood_categories(self, mock_task, mock_buffer, mock_softmax, mock_process_anomaly_scores,
                                       monitoring_service_instance, mock_monitoring_service_template,
                                       mock_prediction_log):
            # Mock objects and methods
            monitoring_service_instance._templates = {
                'outputs': [{
                    'element':
                        'elem_uuid_1',
                    'template': [{
                        'category': 'cat_uuid_1',
                        'template': [{
                            'category': 'cat_uuid_1',
                            'mean': 0.1
                        }, {
                            'category': 'cat_uuid_2',
                            'mean': 0.9
                        }]
                    }, {
                        'category': 'cat_uuid_2',
                        'template': [{
                            'category': 'cat_uuid_1',
                            'mean': 0.8
                        }, {
                            'category': 'cat_uuid_2',
                            'mean': 0.2
                        }]
                    }]
                }]
            }
            monitoring_service_instance.OOD_CAT_MAX_TH = 0.7
            monitoring_service_instance.OOD_CAT_MIN_TH = 0.3
            monitoring_service_instance._ood_sensitivity = 0.5

            task_schema: dict = {
                'outputs': [{
                    'uuid': 'elem_uuid_1',
                    'type': 'category'
                }, {
                    'uuid': 'elem_uuid_2',
                    'type': 'numeric'
                }]
            }
            mock_task.return_value.schema = task_schema

            buffer_items = [
                MagicMock(data={
                    'outputs': [{
                        'element': 'elem_uuid_1',
                        'value': {
                            'scores': {
                                'cat_uuid_1': 0.2,
                                'cat_uuid_2': 0.8
                            }
                        }
                    }]
                })
            ]
            mock_buffer.return_value.read.return_value = buffer_items

            mock_softmax.side_effect = lambda x: x / np.sum(x, axis=1, keepdims=True)

            mock_process_anomaly_scores.return_value = {'elem_uuid_1': 0.5}

            # Call the method
            result = monitoring_service_instance._detect_ood_categories()

            # Assertions
            mock_process_anomaly_scores.assert_called_once()

            assert result == {'elem_uuid_1': 0.5}

    class TestDetectOODNumbers:

        @pytest.fixture
        def mock_task(self, mocker):
            return mocker.patch.object(MonitoringService, 'task')

        @pytest.fixture
        def mock_buffer(self, mocker):
            return mocker.patch.object(MonitoringService, 'buffer')

        @pytest.fixture
        def mock_process_anomaly_scores(self, mocker):
            return mocker.patch.object(MonitoringService, '_process_anomaly_scores')

        def test_detect_ood_numbers(self, mock_task, mock_buffer, mock_process_anomaly_scores,
                                    monitoring_service_instance, mock_monitoring_service_template, mock_prediction_log):
            """
            Tests the detection of out-of-distribution categories.

            Test Steps:
            1. Mock necessary objects and methods including task, buffer, softmax function,
            and anomaly score processing.
            2. Set up mock templates, task schema, buffer items, and expected softmax behavior.
            3. Call the method `_detect_ood_categories` on `monitoring_service_instance`.

            Assertions:
            - `mock_softmax` should be called exactly once.
            - `mock_process_anomaly_scores` should be called exactly once.
            - The returned result should match `{'elem_uuid_1': 0.5}`.
            """
            # Setup
            mock_task_instance = MagicMock()
            mock_task.return_value = mock_task_instance
            mock_task_instance.schema = {
                'outputs': [{
                    'uuid': 'elem_uuid_1',
                    'type': 'float'
                }, {
                    'uuid': 'elem_uuid_2',
                    'type': 'integer'
                }]
            }
            mock_task_instance.get_element.side_effect = lambda element_id, collection: next(
                elem for elem in mock_task_instance.schema['outputs'] if elem['uuid'] == element_id)

            mock_buffer.return_value.read.return_value = [
                MagicMock(data={
                    'outputs': [{
                        'element': 'elem_uuid_1',
                        'value': 3.0
                    }, {
                        'element': 'elem_uuid_2',
                        'value': 5.0
                    }]
                })
            ]

            monitoring_service_instance._templates = {
                'outputs': [{
                    'element': 'elem_uuid_1',
                    'template': {
                        'mean': 2.0,
                        'std': 1.0
                    }
                }, {
                    'element': 'elem_uuid_2',
                    'template': {
                        'mean': 4.0,
                        'std': 0.5
                    }
                }]
            }

            monitoring_service_instance._ood_sensitivity = 0.5

            expected_anomaly_scores = {'elem_uuid_1': np.array([3.0]), 'elem_uuid_2': np.array([4.0])}
            mock_process_anomaly_scores.return_value = expected_anomaly_scores

            # Act
            result = monitoring_service_instance._detect_ood_numbers()

            # Assert
            mock_process_anomaly_scores.assert_called_once_with(anomaly_scores={}, threshold=2.75)
            assert result == expected_anomaly_scores

        def test_detect_ood_numbers_no_numerical_outputs(self, mock_task, mock_buffer, monitoring_service_instance,
                                                         general_template, mock_monitoring_service_template,
                                                         mock_prediction_log):
            """
            Tests the scenario where mismatches in predicted categories raise a `ValueError`.

            Test Steps:
            1. Mock necessary objects and methods including task, buffer, softmax function,
            and anomaly score processing.
            2. Set up mock templates, task schema, and buffer items with mismatched predicted categories.
            3. Call the method `_detect_ood_categories` on `monitoring_service_instance`.
            4. Assert that a `ValueError` exception is raised with the message indicating category mismatch.

            Assertions:
            - A `ValueError` exception with message "Predicted categories don't match"
            should be raised during the method call.
            """
            # Setup
            mock_task_instance = MagicMock()
            mock_task.return_value = mock_task_instance
            mock_task_instance.schema = {'outputs': [{'uuid': 'elem_uuid_3', 'type': 'category'}]}

            mock_buffer.return_value.read.return_value = [
                MagicMock(data={'outputs': [{
                    'element': 'elem_uuid_3',
                    'value': {
                        'category': 'cat_uuid'
                    }
                }]})
            ]

            monitoring_service_instance._templates = general_template

            # Act
            result = monitoring_service_instance._detect_ood_numbers()

            # Assert
            assert result == {}

    class TestProcessAnomalyScores:

        @pytest.fixture
        def mock_task(self, mocker):
            mock_task_ = mocker.patch.object(MonitoringService, 'task')
            mock_task_instance = MagicMock()
            mock_task_.return_value = mock_task_instance
            return mock_task_

        @pytest.fixture
        def mock_send_email_notification(self, mocker):
            return mocker.patch('nexusml.engine.services.monitoring.send_email_notification')

        @pytest.fixture
        def mock_ema(self, mocker):
            return mocker.patch.object(MonitoringService, 'ema')

        @pytest.fixture
        def mock_element_db(self, mocker):
            return mocker.patch.object(ElementDB, 'get_from_id')

        @pytest.fixture
        def mock_task_resource(self, mocker):
            return mocker.patch.object(Task, 'get')

        def test_process_anomaly_scores(self, mock_task, mock_send_email_notification, mock_ema,
                                        monitoring_service_instance, mock_element_db, mock_task_resource):
            """
            Tests the processing of anomaly scores and verifies email notifications are sent
            for elements exceeding the threshold.

            Test Steps:
            1. Mock necessary objects and methods including task, email notification function, and EMA function.
            2. Set up mock anomaly scores and threshold.
            3. Mock task instance and its methods.
            4. Call the method `_process_anomaly_scores` on `monitoring_service_instance`.

            Assertions:
            - The returned result should match `{'elem_uuid_1': 2.0}`.
            - `send_email_notification` should be called exactly once.
            """

            anomaly_scores: dict = {'elem_uuid_1': np.array([1.0, 2.0, 3.0]), 'elem_uuid_2': np.array([0.5, 0.7, 0.9])}
            threshold: float = 1.5

            # Mocks
            mock_task_instance = MagicMock()
            mock_task.return_value = mock_task_instance
            mock_task_instance.uuid = 'task_uuid'
            mock_task_instance.get_element.side_effect = lambda element_id, collection: {
                'name': f'Element {element_id}',
                'uuid': element_id
            }

            mock_ema.side_effect = lambda data: np.mean(a=data)

            # Act
            result = monitoring_service_instance._process_anomaly_scores(anomaly_scores=anomaly_scores,
                                                                         threshold=threshold)

            # Assert
            assert result == {'elem_uuid_1': 2.0}
            mock_send_email_notification.assert_called_once()

        def test_process_anomaly_scores_no_ood(self, mock_task, mock_send_email_notification, mock_ema,
                                               monitoring_service_instance, mock_element_db, mock_task_resource):
            """
            Tests the processing of anomaly scores when no elements exceed the threshold,
            ensuring no email notifications are sent.

            Test Steps:
            1. Mock necessary objects and methods including task, email notification function, and EMA function.
            2. Set up mock anomaly scores and a higher threshold.
            3. Mock task instance and its methods.
            4. Call the method `_process_anomaly_scores` on `monitoring_service_instance`.

            Assertions:
            - The returned result should be an empty dictionary (`{}`).
            - `send_email_notification` should not be called.
            """
            # Setup
            mock_task_instance = MagicMock()
            mock_task.return_value = mock_task_instance

            anomaly_scores = {'elem_uuid_1': np.array([1.0, 1.2, 1.1]), 'elem_uuid_2': np.array([0.5, 0.7, 0.9])}
            threshold = 2.0

            mock_ema.side_effect = lambda data: np.mean(a=data)

            # Act
            result = monitoring_service_instance._process_anomaly_scores(anomaly_scores, threshold)

            # Assert
            assert result == dict()
            mock_send_email_notification.assert_not_called()

    class TestRefreshTemplates:

        @pytest.fixture
        def mock_task(self, mocker):
            return mocker.patch.object(MonitoringService, 'task')

        @pytest.fixture
        def mock_service_db(self, mocker):
            return mocker.patch('nexusml.engine.services.monitoring.ServiceDB.filter_by_task_and_type')

        @pytest.fixture
        def mock_monitoring_service_templates_schema(self, mocker):
            return mocker.patch('nexusml.engine.services.monitoring.MonitoringServiceTemplatesSchema')

        @pytest.fixture
        def mock_verify_templates(self, mocker):
            return mocker.patch.object(MonitoringService, 'verify_templates', autospec=True)

        def test_refresh_templates(self, mock_task, mock_service_db, mock_monitoring_service_templates_schema,
                                   mock_verify_templates, monitoring_service_instance):
            """
            Tests the refresh_templates method in MonitoringService.

            Test Steps:
            1. Mock necessary objects and methods including task, service database, schema,
               MonitoringServiceTemplatesSchema, and verify_templates method.
            2. Set up mock task instance, service data, and loaded templates.
            3. Call the refresh_templates method on monitoring_service_instance.

            Assertions:
            - verify_templates should be called once with loaded_templates and task schema.
            - The _templates attribute of monitoring_service_instance should be updated to loaded_templates.
            """
            # Mocks
            mock_task_instance = MagicMock()
            mock_task_instance.uuid = 'task_uuid'
            mock_task_instance.schema = {'some': 'schema'}
            mock_task_instance.db_object.return_value.task_id = 'task_id'
            mock_task.return_value = mock_task_instance

            service_data = {'some': 'data'}
            mock_service_db.return_value.data = service_data

            loaded_templates = {'loaded': 'templates'}
            mock_monitoring_service_templates_schema_instance = mock_monitoring_service_templates_schema.return_value
            mock_monitoring_service_templates_schema_instance.load.return_value = loaded_templates

            monitoring_service_instance._templates = None

            # Act
            monitoring_service_instance.refresh_templates()

            # Assert
            mock_verify_templates.assert_called_once()
            assert monitoring_service_instance._templates == loaded_templates
