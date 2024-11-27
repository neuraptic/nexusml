import datetime
from unittest.mock import MagicMock

import pytest

from nexusml.database.tasks import ElementDB
from nexusml.engine.services.active_learning import ActiveLearningService
from nexusml.engine.services.active_learning import Comment
from nexusml.engine.services.active_learning import Example
from nexusml.engine.services.active_learning import PredictionDB
from nexusml.engine.services.active_learning import PredictionLog
from nexusml.engine.services.active_learning import ServiceDB
from nexusml.engine.services.active_learning import Task


class TestActiveLearningService:

    class TestQuery:

        @pytest.fixture
        def mock_al_service_instance(self, mocker):
            return ActiveLearningService(buffer=mocker.MagicMock(), query_interval=1, max_examples_per_query=1)

        @pytest.fixture
        def mock_ac_task(self, mocker):
            return mocker.patch.object(target=ActiveLearningService, attribute='task')

        @pytest.fixture
        def mock_buffer(self, mocker):
            return mocker.patch.object(target=ActiveLearningService, attribute='buffer')

        @pytest.fixture
        def mock_update_service_status(self, mocker):
            return mocker.patch.object(target=ActiveLearningService, attribute='update_service_status')

        @pytest.fixture
        def mock_save_to_db(self, mocker):
            return mocker.patch(target='nexusml.engine.services.active_learning.save_to_db', autospec=True)

        @pytest.fixture
        def mock_example(self, mocker):
            mock_example_pb = mocker.patch.object(target=Example, attribute='post_batch', return_value=MagicMock())
            mock_example_get = mocker.patch.object(target=Example,
                                                   attribute='get',
                                                   return_value=MagicMock(spec=Example))
            return mock_example_pb, mock_example_get

        @pytest.fixture
        def mock_service_db(self, mocker):
            mock_service_db_obj: MagicMock = MagicMock(spec=ServiceDB)
            mock_service_db_ = mocker.patch.object(target=ServiceDB,
                                                   attribute='filter_by_task_and_type',
                                                   return_value=mock_service_db_obj)
            return mock_service_db_

        @pytest.fixture
        def mock_prediction_log(self, mocker):
            expected_result = {'inputs': [], 'outputs': [{'element': 'elem_uuid', 'value': 'file1'}]}
            prediction_log_instance: MagicMock = MagicMock(spec=PredictionLog)
            prediction_log_instance.dump.return_value: dict = expected_result
            mock_prediction_log_ = mocker.patch.object(target=PredictionLog,
                                                       attribute='get',
                                                       return_value=prediction_log_instance)
            return mock_prediction_log_

        @pytest.fixture
        def mock_prediction_db(self, mocker):
            mock_example_dict: dict = {'output': MagicMock()}
            mock_prediction_db_query: MagicMock = MagicMock()
            (mock_prediction_db_query.with_entities.return_value.filter.return_value.all).return_value = [[
                mock_example_dict
            ]]
            mock_prediction_db_query = mocker.patch.object(target=PredictionDB,
                                                           attribute='query',
                                                           return_value=mock_prediction_db_query)

            mock_prediction_db_id = mocker.patch.object(target=PredictionDB,
                                                        attribute='get_from_id',
                                                        return_value=MagicMock(spec=PredictionDB))
            return mock_prediction_db_query, mock_prediction_db_id

        @pytest.fixture()
        def mock_element_db(self, mocker):
            mock_element_db_ = mocker.patch.object(target=ElementDB,
                                                   attribute='get_from_id',
                                                   return_value=MagicMock(spec=ElementDB))
            return mock_element_db_

        @pytest.fixture()
        def mock_task(self, mocker):
            mock_task_db_obj: MagicMock = MagicMock(spec=Task)
            mock_task_ = mocker.patch.object(target=Task, attribute='get', return_value=mock_task_db_obj)
            return mock_task_

        @pytest.fixture()
        def mock_comment(self, mocker):
            mock_comment_ = mocker.patch.object(target=Comment, attribute='post')
            return mock_comment_

        def test_query(self, mock_ac_task, mock_buffer, mock_update_service_status, mock_save_to_db, mock_example,
                       mock_service_db, mock_al_service_instance, mock_prediction_log, mock_prediction_db,
                       mock_element_db, mock_task, mock_comment):
            """
            Test the query method when new examples are available.

            Test Steps:
            1. Setup mock instances for the task, buffer, and service methods.
            2. Configure the mocks to return predefined values simulating the query process.
            3. Call the query method on the ActiveLearningService instance.

            Assertions:
            - Verify that the update_service_status method is called with specific arguments.
            - Ensure that process_post_or_put_request and save_to_db methods are called once.
            - Check that the returned result matches the expected structure and values.
            """

            # Setup
            mock_task_instance = MagicMock()
            mock_task_instance.uuid = 'task_uuid'
            mock_task_instance.agent.return_value = 'agent'
            mock_task_instance.last_al_update = None
            mock_ac_task.return_value = mock_task_instance

            mock_buffer_read_response: list = [
                MagicMock(data={
                    'outputs': [{
                        'element': 'elem_uuid',
                        'value': 'file1'
                    }],
                    'inputs': list()
                })
            ]
            mock_buffer.return_value.read.return_value = mock_buffer_read_response

            mock_buffer_item_files_response: dict = {
                'metadata': ['file1', 'file2'],
                'input': ['input1', 'input2'],
                'output': ['output1']
            }
            mock_buffer.return_value.buffer_io.return_value.item_files.return_value = mock_buffer_item_files_response

            # Act
            result = mock_al_service_instance.query()

            # Assert
            mock_save_to_db.assert_called_once()

            expected_result = [{'inputs': [], 'outputs': [{'element': 'elem_uuid', 'value': 'file1'}]}]
            assert result == expected_result

        def test_query_no_update_needed(self, mock_ac_task, mock_buffer, mock_save_to_db, mock_service_db,
                                        mock_al_service_instance):
            """
            Test the query method when no update is needed.

            Test Steps:
            1. Setup mock instances for the task and configure last_al_update to be recent.
            2. Call the query method on the ActiveLearningService instance.

            Assertions:
            - Verify that the buffer read method is not called.
            - Check that the returned result is an empty list.
            """

            # Setup
            mock_task_instance: MagicMock = MagicMock()
            mock_task_instance.last_al_update = datetime.datetime.utcnow() - datetime.timedelta(days=0.5)
            mock_ac_task.return_value = mock_task_instance

            # Act
            result = mock_al_service_instance.query()

            # Assert
            assert result == []

        def test_query_no_new_examples(self, mock_ac_task, mock_buffer, mock_save_to_db, mock_update_service_status,
                                       mock_service_db, mock_al_service_instance):
            """
            Test the query method when no new examples are available.

            Test Steps:
            1. Setup mock instances for the task with no recent last_al_update.
            2. Configure the buffer to return an empty list.
            3. Call the query method on the ActiveLearningService instance.

            Assertions:
            - Check that the returned result is an empty list.
            """
            # Setup
            mock_task_instance: MagicMock = MagicMock()
            mock_task_instance.last_al_update = None
            mock_ac_task.return_value = mock_task_instance

            mock_buffer_value = MagicMock(read=list)
            mock_buffer.return_value = mock_buffer_value

            # Act
            result = mock_al_service_instance.query()

            # Assert
            assert result == []
