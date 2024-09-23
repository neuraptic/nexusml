# TODO: Try to make this module independent from `nexusml.api`

from datetime import datetime
from datetime import timedelta
from typing import List
from unittest.mock import MagicMock

import pytest

from nexusml.api.buffers import ALBuffer
from nexusml.api.resources.ai import PredictionLog
from nexusml.database.ai import PredictionDB
from nexusml.database.core import save_to_db
from nexusml.database.services import Service
from nexusml.engine.services.active_learning import ActiveLearningService
from nexusml.engine.services.continual_learning import ContinualLearningService
from nexusml.engine.services.monitoring import ClientDB
from nexusml.engine.services.monitoring import MonitoringService
from nexusml.enums import ServiceType

pytestmark = [pytest.mark.unit, pytest.mark.fast]

############################
# CONTINUAL LEARNING TESTS #
############################


class TestContinualLearningService:

    def test_should_train(self):

        def _default_kwargs(now: datetime) -> dict:
            return {
                'eol_dt': now + timedelta(days=30),
                # Training frequency
                'min_days': 7,
                'max_days': 28,
                # Requirements and Quotas
                'min_sample': 0.2,
                'max_examples': 100000,
                'min_cpu_quota': 150,
                'min_gpu_quota': 200,
                'max_cpu_quota': 300,
                'max_gpu_quota': 400,
                'cpu_usage': 0,
                'gpu_usage': 125,
                # Last training session info
                'last_dt': now - timedelta(days=10),
                'last_len': 3,
                'last_dev': 'gpu',
                # Current examples info
                'num_trained_examples': {
                    '70854b3c-a230-4f64-8a4a-451ac0ab1ec8': 20000,
                    'd36dcca7-8251-4e13-b245-ba69e9cbae48': 20000,
                    '3456144d-34df-4ffe-8c5a-7a56d7254c65': 20000,
                    'bb5aa03e-5949-4e07-908d-3edfd114a6ca': 20000,
                    'total': 80000
                },
                'num_untrained_examples': {
                    '70854b3c-a230-4f64-8a4a-451ac0ab1ec8': 4000,
                    'd36dcca7-8251-4e13-b245-ba69e9cbae48': 4000,
                    '3456144d-34df-4ffe-8c5a-7a56d7254c65': 4000,
                    'bb5aa03e-5949-4e07-908d-3edfd114a6ca': 4000,
                    'total': 16000,
                }
            }

        now = datetime.utcnow()
        ################################################################
        # Trigger retraining with a 20% increase in number of examples #
        ################################################################
        kwargs = _default_kwargs(now=now)
        assert ContinualLearningService.should_train(**kwargs)
        ##########################################################
        # Trigger retraining with the maximum number of examples #
        ##########################################################
        kwargs = _default_kwargs(now=now)
        kwargs['num_untrained_examples'] = {
            '70854b3c-a230-4f64-8a4a-451ac0ab1ec8': 5000,
            'd36dcca7-8251-4e13-b245-ba69e9cbae48': 5000,
            '3456144d-34df-4ffe-8c5a-7a56d7254c65': 5000,
            'bb5aa03e-5949-4e07-908d-3edfd114a6ca': 5000,
            'total': 20000,
        }
        total_examples = kwargs['num_trained_examples']['total'] + kwargs['num_untrained_examples']['total']
        assert total_examples == kwargs['max_examples']
        assert ContinualLearningService.should_train(**kwargs)
        #################################################
        # Trigger retraining with the maximum GPU quota #
        #################################################
        kwargs = _default_kwargs(now=now)
        kwargs['gpu_usage'] = kwargs['max_gpu_quota'] - (kwargs['last_len'] * (1 + kwargs['min_sample'])) - 1
        assert ContinualLearningService.should_train(**kwargs)
        ##########################################
        # Trigger retraining with new categories #
        ##########################################
        kwargs = _default_kwargs(now=now)
        kwargs['gpu_usage'] = kwargs['min_gpu_quota']
        kwargs['last_dt'] = now - timedelta(days=kwargs['min_days'])
        kwargs['num_untrained_examples'] = {
            'c33cbbde-97c2-4db7-ab26-c41d27792dea': 1,
            'total': 1,
        }
        assert ContinualLearningService.should_train(**kwargs)
        ##########################################################
        # Scheduled retraining session with expected quota limit #
        ##########################################################
        kwargs = _default_kwargs(now=now)
        kwargs['eol_dt'] = now + timedelta(days=1)
        kwargs['last_dt'] = now - timedelta(days=kwargs['min_days'])
        kwargs['num_untrained_examples'] = {'total': 0}
        assert ContinualLearningService.should_train(**kwargs)
        ##########################################################
        # Scheduled retraining sessions with maximum quota limit #
        ##########################################################
        kwargs = _default_kwargs(now=now)
        kwargs['last_dt'] = now - timedelta(days=kwargs['max_days'])
        kwargs['gpu_usage'] = kwargs['min_gpu_quota'] + 1
        kwargs['num_untrained_examples'] = {'total': 0}
        assert ContinualLearningService.should_train(**kwargs)
        #############################################################
        # No scheduled retraining sessions due to end of life (EOL) #
        #############################################################
        kwargs = _default_kwargs(now=now)
        kwargs['eol_dt'] = now + timedelta(days=1)
        kwargs['last_dt'] = now - timedelta(days=kwargs['max_days'] + 1)
        kwargs['gpu_usage'] = kwargs['max_gpu_quota'] - (2 * kwargs['last_len'] * (1 + kwargs['min_sample']))
        kwargs['num_untrained_examples'] = {'total': 0}
        assert not ContinualLearningService.should_train(**kwargs)
        #####################################
        # Exceed maximum training frequency #
        #####################################
        kwargs = _default_kwargs(now=now)
        kwargs['last_dt'] = now - timedelta(days=kwargs['min_days'] - 1)
        assert not ContinualLearningService.should_train(**kwargs)
        ############################
        # Exceed maximum GPU quota #
        ############################
        kwargs = _default_kwargs(now=now)
        kwargs['gpu_usage'] = kwargs['max_gpu_quota'] - (kwargs['last_len'] * (1 + kwargs['min_sample'])) + 1
        assert not ContinualLearningService.should_train(**kwargs)
        ################################
        # REPEAT TESTS WITH CPU QUOTAS #
        ################################
        pass  # TODO: test CPU quotas


#########################
# ACTIVE LEARNING TESTS #
#########################


class TestActiveLearningService:

    def test_query(self, al_local_buffers, al_buffer_items):

        def _test_query(buffer_idx: List[int], expected_queried_idx: List[int]):
            # Create files deleted in previous query
            predictions = [al_buffer_items_[x] for x in buffer_idx]

            # Create buffer items
            buffer.write(items=predictions)

            # Force query
            buffer.task().last_al_update = None
            save_to_db(objects=buffer.task())
            actual_queried_predictions = al.query()
            assert len(buffer.read()) == 0

            # Check query
            expected_queried_predictions = [al_buffer_items_[x].dump(serialize=False) for x in expected_queried_idx]

            assert len(actual_queried_predictions) == len(expected_queried_predictions)
            assert all(x in actual_queried_predictions for x in expected_queried_predictions)
            assert all(x in expected_queried_predictions for x in actual_queried_predictions)

        # Note: indices of predictions with highest entropy in `predictions` (descending): 5, 6, 7, 3, 1, 2, 8, 0, 4, 9
        buffer: ALBuffer = al_local_buffers[0]

        service_db: Service = Service.filter_by_task_and_type(task_id=buffer.task().task_id,
                                                              type_=ServiceType.ACTIVE_LEARNING)
        client_db_agent: ClientDB = ClientDB.get(client_id=service_db.client_id)

        al_buffer_items_: list = list()
        for al_buffer_item in al_buffer_items:
            if al_buffer_item.task_id == buffer.task().task_id:
                prediction_db_obj: PredictionDB = PredictionDB.query().filter_by(
                    prediction_id=al_buffer_item.prediction_id).first()
                prediction_log: PredictionLog = PredictionLog.get(agent=client_db_agent,
                                                                  db_object_or_id=prediction_db_obj,
                                                                  check_permissions=False)
                al_buffer_items_.append(prediction_log)

        buffer.clear()
        al = ActiveLearningService(buffer=buffer, query_interval=2, max_examples_per_query=3)

        # Iteration 1
        _test_query(buffer_idx=[7, 0, 8, 9], expected_queried_idx=[7, 8, 0])

        # Iteration 2
        _test_query(buffer_idx=[7, 0, 8, 9, 6, 1, 3], expected_queried_idx=[6, 7, 3])

        # Iteration 3
        _test_query(buffer_idx=[7, 0, 8, 9, 6, 1, 3, 5], expected_queried_idx=[5, 6, 7])

        # Iteration 4
        _test_query(buffer_idx=[4, 2], expected_queried_idx=[2, 4])


#####################
# MONITORING TESTS #
####################


class TestMonitoringService:

    @pytest.fixture
    def monitoring_templates(self) -> dict:
        _monitoring_templates = {
            'ai_model':
                '52a4f64d-37b5-49f9-85e6-a793cd8271e5',
            'outputs': [{
                'element':
                    'be93cbf8-6638-41d6-9808-fad09e89471e',
                'template': [{
                    'category':
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                    'template': [
                        {
                            'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                            'mean': 0.13594628
                        },
                        {
                            'category': '3151109a-502d-4e53-926e-d0584e324a4b',
                            'mean': 0.37720984
                        },
                        {
                            'category': 'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                            'mean': 0.38571908
                        },
                        {
                            'category': 'e18fee08-c6aa-4a1e-b89c-e322f4927946',
                            'mean': 0.1011248
                        },
                    ]
                }, {
                    'category':
                        '3151109a-502d-4e53-926e-d0584e324a4b',
                    'template': [
                        {
                            'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                            'mean': 0.14991866
                        },
                        {
                            'category': '3151109a-502d-4e53-926e-d0584e324a4b',
                            'mean': 0.35641616
                        },
                        {
                            'category': 'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                            'mean': 0.25707804
                        },
                        {
                            'category': 'e18fee08-c6aa-4a1e-b89c-e322f4927946',
                            'mean': 0.23658715
                        },
                    ]
                }, {
                    'category':
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                    'template': [
                        {
                            'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                            'mean': 0.49515087
                        },
                        {
                            'category': '3151109a-502d-4e53-926e-d0584e324a4b',
                            'mean': 0.3152546
                        },
                        {
                            'category': 'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                            'mean': 0.0676632
                        },
                        {
                            'category': 'e18fee08-c6aa-4a1e-b89c-e322f4927946',
                            'mean': 0.12193134
                        },
                    ]
                }, {
                    'category':
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946',
                    'template': [
                        {
                            'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                            'mean': 0.11299642
                        },
                        {
                            'category': '3151109a-502d-4e53-926e-d0584e324a4b',
                            'mean': 0.33851384
                        },
                        {
                            'category': 'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                            'mean': 0.39530407
                        },
                        {
                            'category': 'e18fee08-c6aa-4a1e-b89c-e322f4927946',
                            'mean': 0.15318567
                        },
                    ]
                }]
            }, {
                'element':
                    '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'template': [
                    {
                        'category':
                            '9220bb77-9088-42c6-9b2b-9aaf496d6cfa',
                        'template': [
                            {
                                'category': '9220bb77-9088-42c6-9b2b-9aaf496d6cfa',
                                'mean': 0.08437385
                            },
                            {
                                'category': 'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
                                'mean': 0.02315868
                            },
                            {
                                'category': 'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                                'mean': 0.33437491
                            },
                            {
                                'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                                'mean': 0.55809257
                            },
                        ]
                    },
                    {
                        'category':
                            'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
                        'template': [
                            {
                                'category': '9220bb77-9088-42c6-9b2b-9aaf496d6cfa',
                                'mean': 0.10173946
                            },
                            {
                                'category': 'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
                                'mean': 0.28644909
                            },
                            {
                                'category': 'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                                'mean': 0.48732275
                            },
                            {
                                'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                                'mean': 0.12448869
                            },
                        ]
                    },
                    {
                        'category':
                            'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                        'template': [
                            {
                                'category': '9220bb77-9088-42c6-9b2b-9aaf496d6cfa',
                                'mean': 0.29556845
                            },
                            {
                                'category': 'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
                                'mean': 0.23905067
                            },
                            {
                                'category': 'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                                'mean': 0.25798399
                            },
                            {
                                'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                                'mean': 0.20739688
                            },
                        ]
                    },
                    {
                        'category':
                            'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                        'template': [
                            {
                                'category': '9220bb77-9088-42c6-9b2b-9aaf496d6cfa',
                                'mean': 0.47837626
                            },
                            {
                                'category': 'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
                                'mean': 0.12546526
                            },
                            {
                                'category': 'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                                'mean': 0.2833071
                            },
                            {
                                'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                                'mean': 0.11285138
                            },
                        ]
                    },
                ]
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'template': {
                    'mean': 20.37,
                    'std': 4.65
                }
            }]
        }
        return _monitoring_templates

    def test_detect_ood_predictions(self, mocker, task_and_prediction, mon_local_buffers, mon_buffer_items,
                                    monitoring_templates):
        mocker.patch.object(Service,
                            'filter_by_task_and_type',
                            return_value=MagicMock(data=monitoring_templates, spec=Service))
        buffer = mon_local_buffers[0]
        monitoring = MonitoringService(buffer=buffer,
                                       refresh_interval=100,
                                       ood_min_sample=5,
                                       ood_sensitivity=0.2,
                                       ood_smoothing=0.8)
        ood_preds = monitoring.detect_ood_predictions()
        assert len(ood_preds) == 1  # TODO: replace this line with actual tests
        assert len(buffer.read()) == 0
