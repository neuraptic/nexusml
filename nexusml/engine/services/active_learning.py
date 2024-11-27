# TODO: Try to make this module independent from `nexusml.api`

import copy
from datetime import datetime
from datetime import timedelta
from typing import List

from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.examples import Comment
from nexusml.api.resources.examples import Example
from nexusml.api.resources.tasks import Task
from nexusml.constants import API_NAME
from nexusml.database import save_to_db
from nexusml.database.ai import PredictionDB
from nexusml.database.organizations import ClientDB
from nexusml.database.services import Service as ServiceDB
from nexusml.database.tasks import ElementDB
from nexusml.engine.buffers import ALBuffer
from nexusml.engine.services.base import Service
from nexusml.enums import ElementValueType
from nexusml.enums import ServiceType
from nexusml.statuses import AL_ANALYZING_STATUS_CODE
from nexusml.statuses import AL_WAITING_STATUS_CODE


class ActiveLearningService(Service):
    """ Active Learning Service. """
    comment_message: str = f'This is {API_NAME}. Can you give me a hand with this example? :-)'

    def __init__(self, buffer: ALBuffer, query_interval: int, max_examples_per_query: int):
        """
        Active Learning Service.

        Steps:
            1. Initialize the service with the given buffer and service type.
            2. Retrieve the corresponding service from the database.
            3. Fetch the client database agent using the service's client ID.

        Args:
            buffer (ALBuffer): Buffer to store predictions
            query_interval (int): Interval in days between queries
            max_examples_per_query (int): Maximum number of examples to query


        """
        super().__init__(buffer=buffer, service_type=ServiceType.ACTIVE_LEARNING)
        self._query_interval = query_interval
        self._max_examples_per_query = max_examples_per_query
        self.service_db: ServiceDB = ServiceDB.filter_by_task_and_type(task_id=self.task().task_id,
                                                                       type_=ServiceType.ACTIVE_LEARNING)
        self.client_db_agent: ClientDB = ClientDB.get(client_id=self.service_db.client_id)

    def service_name(self) -> str:
        return 'active-learning'

    def query(self) -> List[dict]:
        """
        Queries the buffer for predictions and processes them into examples for active learning.
        The function checks whether the required query interval has passed before proceeding.
        It selects predictions based on entropy and processes them into examples, which are
        then posted for review. The service and task statuses are updated during the process.

        Steps:
        1. Check if the required interval since the last update has passed.
        2. Retrieve and process predictions ordered by entropy.
        3. Post selected predictions as examples for review.
        4. Update the service status to 'analyzing' during processing and 'waiting' once completed.
        5. Clear the buffer and update the last query time in the database.

        Returns:
            List[dict]: A list of queried predictions, capped by the maximum examples per query.

        Raises:
            None
        """
        # Check interval
        now: datetime = datetime.utcnow()

        if self.task().last_al_update is None:
            self.task().last_al_update = datetime.min

        if now < self.task().last_al_update + timedelta(days=self._query_interval):
            return []

        # Chose predictions with the highest entropy
        # Note: buffer's `read()` returns predictions ordered by entropy
        buffered_predictions: list = list()
        buffer_prediction_ids: list = [buffer_item_db.prediction_id for buffer_item_db in self.buffer().read()]
        for prediction_id in buffer_prediction_ids:
            prediction_db_obj: PredictionDB = PredictionDB.query().filter_by(prediction_id=prediction_id).first()
            prediction_log: PredictionLog = PredictionLog.get(agent=self.client_db_agent,
                                                              db_object_or_id=prediction_db_obj,
                                                              check_permissions=False)
            prediction_data: dict = prediction_log.dump(serialize=False)
            buffered_predictions.append(prediction_data)

        queried_predictions: list = buffered_predictions[:self._max_examples_per_query]

        # Create examples
        new_examples: list = copy.deepcopy(queried_predictions)
        if new_examples:
            self.update_service_status(code=AL_ANALYZING_STATUS_CODE, ignore_errors=True)

            for raw_example_data in new_examples:
                selected_example_data: dict = dict()
                # Remove scores from categorical values
                for target in raw_example_data['outputs']:
                    output_type: ElementDB = ElementDB.get_from_id(id_value=target['element'],
                                                                   parent=self.task()).value_type
                    if output_type != ElementValueType.CATEGORY:
                        continue
                    target['value'] = target['value']['category']

                # Create example
                # Join inputs, outputs and metadata in single 'values' list
                selected_example_data['values'] = raw_example_data['inputs']
                selected_example_data['values'] += raw_example_data['outputs'] if 'outputs' in raw_example_data else []
                selected_example_data[
                    'values'] += raw_example_data['metadata'] if 'metadata' in raw_example_data else []
                selected_example_data['labeling_status'] = 'pending_review'

                task: Task = Task.get(agent=self.client_db_agent, db_object_or_id=self.task())

                examples: list[Example] = Example.post_batch(data=[selected_example_data], task=task)
                for example in examples:
                    Comment.post(agent=self.client_db_agent,
                                 data={'message': self.comment_message},
                                 parents=[task, example])

            self.update_service_status(code=AL_WAITING_STATUS_CODE, ignore_errors=True)

        # Clear buffer
        self.buffer().clear()

        # Keep track of the last update datetime
        self.task().last_al_update = now
        save_to_db(objects=self.task())

        return queried_predictions  # TODO: Return `new_examples` instead (requires adapting tests)
