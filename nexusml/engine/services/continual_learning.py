# TODO: Try to make this module independent from `nexusml.api`

from datetime import datetime
from datetime import timedelta
import math
from typing import Tuple

from nexusml.engine.buffers import CLBuffer
from nexusml.engine.services.base import Service
from nexusml.enums import ServiceType


class ContinualLearningService(Service):
    """ Continual Learning Service. """

    def __init__(self, buffer: CLBuffer):
        """
        Initializes the Continual Learning Service.

        The service is set up with a buffer for predictions, and it is categorized
        as a 'continual_learning' service type. This service will handle the logic
        for determining when to retrain the AI model based on resource usage and
        the state of examples in the buffer.

        Args:
            buffer (CLBuffer): The buffer from which predictions and example data will be retrieved.
        """
        super().__init__(buffer=buffer, service_type=ServiceType.CONTINUAL_LEARNING)

    def service_name(self) -> str:
        """
        Returns the name of the service.

        This is used to identify the service as 'continual-learning'.

        Returns:
            str: The name of the service.
        """
        return 'continual-learning'

    @staticmethod
    def should_train(eol_dt: datetime, min_days: int, max_days: int, min_sample: float, max_examples: int,
                     min_cpu_quota: int, min_gpu_quota: int, max_cpu_quota: int, max_gpu_quota: int, cpu_usage: float,
                     gpu_usage: float, last_dt: datetime, last_len: float, last_dev: str, num_trained_examples: dict,
                     num_untrained_examples: dict) -> bool:
        """
        Determines whether the AI model should be retrained based on various criteria.

        This function evaluates whether the AI model requires retraining, taking into
        account constraints such as the availability of examples, the usage of CPU/GPU
        quotas, and the time since the last training session. It also checks for certain
        conditions like exceeding the number of allowed examples, hitting the minimum
        number of examples for retraining, and quota limits. The function ensures that
        retraining only occurs when necessary to optimize resource usage and model performance.

        Steps:
        1. Validates input parameters, such as ensuring `min_sample` is between 0 and 1,
           CPU/GPU usage is non-zero, and `last_dt` is valid.
        2. Calculates the ratio of untrained to trained examples to estimate the expected
           training session length.
        3. Checks if the current quota usage and future expected usage exceeds the allowed quota.
        4. Ensures retraining does not occur too frequently by enforcing a minimum time
           between training sessions.
        5. Determines the next valid retraining time window and checks if retraining should
           be scheduled within the model's life span.
        6. Checks for the addition of new categories in untrained examples and determines
           if they trigger a retraining session.

        Args:
            eol_dt (datetime): The end-of-life date and time (usually marking the end of subscription).
            min_days (int): The minimum number of days between retraining sessions.
            max_days (int): The maximum allowable number of days between retraining sessions.
            min_sample (float): The minimum sample ratio to trigger retraining, represented as a value between 0 and 1.
            max_examples (int): The maximum number of examples allowed.
            min_cpu_quota (int): The minimum CPU quota (in hours) required for training.
            min_gpu_quota (int): The minimum GPU quota (in hours) required for training.
            max_cpu_quota (int): The maximum CPU quota (in hours) allowed for training.
            max_gpu_quota (int): The maximum GPU quota (in hours) allowed for training.
            cpu_usage (float): The current usage of CPU resources (in hours).
            gpu_usage (float): The current usage of GPU resources (in hours).
            last_dt (datetime): The date and time of the last training session.
            last_len (float): The length of the last training session (in hours).
            last_dev (str): The device used for the last training session, either 'cpu' or 'gpu'.
            num_trained_examples (dict): A dictionary containing the number of trained examples,
                                          including a 'total' key for the total number of trained examples.
            num_untrained_examples (dict): A dictionary containing the number of untrained examples,
                                           including a 'total' key for the total number of untrained examples.

        Returns:
            bool: True if retraining should be performed, False otherwise.

        Raises:
            AssertionError: If any validation on input parameters fails, such as invalid `min_sample`,
                            exceeded `max_examples`, or incorrect `last_dt`.
        """

        def _remaining_sessions_and_wait(quota_limit: float, quota_usage: float,
                                         scheduled_len: float) -> Tuple[float, float]:
            """
            Calculates the remaining available training sessions based on the quota
            limits and estimates the time until the next retraining session.

            Args:
                quota_limit (float): The total quota allowed (CPU or GPU) for training.
                quota_usage (float): The current quota usage (CPU or GPU) for training.
                scheduled_len (float): The estimated length of the next training session.

            Returns:
                Tuple[float, float]: A tuple containing the number of remaining sessions and
                                     the remaining wait time in hours before the next session.
            """
            remaining_quota = max(quota_limit - quota_usage, 0)
            remaining_sessions = int(remaining_quota / scheduled_len)
            remaining_life = (eol_dt - last_dt).total_seconds() / 3600
            remaining_wait = (remaining_life - remaining_quota) if remaining_sessions > 0 else remaining_life
            return remaining_sessions, remaining_wait

        now = datetime.utcnow()

        # Sanity check
        assert 0 < min_sample < 1, 'Argument `min_sample` must be a value between 0 and 1'
        assert max_examples > 0, 'Argument `max_examples` must be a positive number'
        assert (cpu_usage + gpu_usage) > 0, 'No CPU or GPU usage. AI should have been trained at least once'
        assert last_dt < now, 'Invalid datetime for last training session'
        assert last_len > 0, 'Invalid length for last training session'
        assert last_dev in ['cpu', 'gpu'], 'Invalid device'
        assert 'total' in num_trained_examples, 'Invalid number of trained examples'
        assert 'total' in num_untrained_examples, 'Invalid number of untrained examples'
        total_trained = num_trained_examples['total']
        total_untrained = num_untrained_examples['total']
        assert total_trained > 0, 'No trained examples. AI should have been trained at least once'
        assert (total_trained + total_untrained) <= max_examples, 'Maximum number of examples reached'
        # # Since outputs can be optional, the total number of examples may not match categories' count
        # if len(num_trained_examples) > 1:
        #     assert sum(v for k, v in num_trained_examples.items() if k != 'total') == total_trained
        # if len(num_untrained_examples) > 1:
        #     assert sum(v for k, v in num_untrained_examples.items() if k != 'total') == total_untrained

        # Estimate training session costs
        untrained_rate = total_untrained / total_trained
        expected_len = last_len * (1 + untrained_rate)

        # Check quota
        min_quota = min_cpu_quota if last_dev == 'cpu' else min_gpu_quota
        max_quota = max_cpu_quota if last_dev == 'cpu' else max_gpu_quota
        quota_usage = cpu_usage if last_dev == 'cpu' else gpu_usage
        if (quota_usage + expected_len) > max_quota:
            return False

        # Check maximum frequency
        if now < (last_dt + timedelta(days=min_days)):
            return False

        # Check if a minimum number of examples were uploaded
        if untrained_rate >= min_sample:
            return True

        # Check if there is a scheduled session
        scheduled_len = last_len * (1 + min_sample)
        max_wait = max_days * 24
        remaining_sessions, remaining_wait = _remaining_sessions_and_wait(quota_limit=min_quota,
                                                                          quota_usage=quota_usage,
                                                                          scheduled_len=scheduled_len)
        next_wait = remaining_wait / max(remaining_sessions, 1)

        if next_wait <= max_wait:
            time_to_next_session = next_wait
        else:
            # If the wait is higher than the maximum wait,
            # it means that no retraining could be performed with `min_quota`.
            # In such a case, consider `max_quota` instead of `min_quota`.
            remaining_sessions, remaining_wait = _remaining_sessions_and_wait(quota_limit=max_quota,
                                                                              quota_usage=quota_usage,
                                                                              scheduled_len=scheduled_len)
            time_to_next_session = max((remaining_wait / max(remaining_sessions, 1)), max_wait)

        next_session_dt = last_dt + timedelta(hours=time_to_next_session)
        if next_session_dt > (eol_dt - timedelta(hours=scheduled_len)):
            return False
        if now >= next_session_dt:
            return True

        # Check if new categories were added (i.e., categories with untrained examples and without trained examples)
        new_categories = [
            k for k, v in num_untrained_examples.items()
            if k != 'total' and v > 0 and num_trained_examples.get(k, 0) == 0
        ]
        if new_categories:
            # Compute the cost (quota usage) it would have to retrain every `min_days`.
            # If it doesn't exceed the maximum quota, retrain.
            num_sessions = math.ceil((eol_dt - last_dt).total_seconds() / (min_days * 86400))
            max_quota_usage = sum(last_len * ((1 + min_sample)**(i + 1)) for i in range(num_sessions))
            return max_quota_usage < (max_quota - quota_usage)

        return False
