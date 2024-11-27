# TODO: Try to make this module independent from `nexusml.api`

from datetime import datetime
from datetime import timedelta

import pytest

from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.tasks import Task
from nexusml.database.ai import PredictionDB
from nexusml.database.buffers import ALBufferItemDB
from nexusml.database.buffers import MonBufferItemDB
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.services import Service
from nexusml.database.tasks import TaskDB
from nexusml.engine.services.base import Buffer
from nexusml.enums import AIEnvironment

pytestmark = [pytest.mark.unit, pytest.mark.fast]

MOCK_FILE_SIZE = 350


class TestALBuffer:

    def test_read(self, al_local_buffers, al_buffer_items):
        for buffer in al_local_buffers:
            items = buffer.read()
            assert items == sorted(items, key=(lambda x: x.relevance), reverse=True)
            assert items[0].relevance > items[-1].relevance  # descending order double check

    def test_write(self, al_local_buffers):
        # Delete predictions
        empty_table(ALBufferItemDB)
        for buffer in al_local_buffers:
            task_db_obj = buffer.task()
            task_db_obj.al_buffer_items = 0
            task_db_obj.al_buffer_bytes = 0
            save_to_db(objects=task_db_obj)

        # Save predictions with enough free space in all task buffers
        buffer = al_local_buffers[0]
        task_db_obj = buffer.task()
        client = Service.query().filter_by(task_id=task_db_obj.task_id).first().client
        task = Task.get(agent=client, db_object_or_id=task_db_obj)

        preds: list = [{
            'state': 'complete',
            'inputs': [{
                'element': 'input_name_1',
                'value': ''
            }, {
                'element': 'input_name_2',
                'value': ''
            }, {
                'element': 'input_name_3',
                'value': ''
            }],
            'outputs': [{
                'element': 'output_name_0',
                'value': {
                    'category': 'name_0_0',
                    'scores': {
                        'name_0_0': 0.7,
                        'name_0_1': 0.15,
                        'name_0_2': 0.1,
                        'name_0_3': 0.05
                    }
                }
            }, {
                'element': 'output_name_1',
                'value': 59.24
            }, {
                'element': 'output_name_2',
                'value': {
                    'category': 'name_2_0',
                    'scores': {
                        'name_2_0': 0.65,
                        'name_2_1': 0.25,
                        'name_2_2': 0.07,
                        'name_2_3': 0.03
                    }
                }
            }],
            'ai_model': '52a4f64d-37b5-49f9-85e6-a793cd8271e5'
        }]
        predictions = PredictionLog.post_batch(data=preds, task=task, environment=AIEnvironment.PRODUCTION)
        buffer.write(items=predictions)
        buffer_items = ALBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()
        assert len(buffer_items) == len(preds)
        buffer_items_size = sum(
            PredictionDB.query().filter_by(prediction_id=x.prediction_id).first().size for x in buffer_items)
        assert buffer_items_size == sum(x.size() for x in predictions)
        assert task_db_obj.al_buffer_items == len(preds)
        assert task_db_obj.al_buffer_bytes == buffer_items_size
        pass  # TODO: check values

        # Verify the items with the lowest relevance are deleted when there is no free space in the buffer
        for buffer in al_local_buffers:
            task_db_obj = buffer.task()
            # Set the 3 least relevant items
            lowest_relevance_ids = [2, 4, 8]
            buffer_items = ALBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()
            for idx, item in enumerate(buffer_items):
                id_ = idx + 1
                if id_ in lowest_relevance_ids:
                    item.relevance = id_ / 1000
                else:
                    item.relevance = 10.0 + id_
            save_to_db(objects=buffer_items)
            # Set buffer size limit to current usage
            org_buffer_bytes = Buffer.MAX_BUFFER_BYTES
            Buffer.MAX_BUFFER_BYTES = task_db_obj.al_buffer_bytes
            pass  # TODO: what about buffer items limit?
            # Save predictions
            buffer.write(items=predictions)
            assert task_db_obj.al_buffer_items <= Buffer.MAX_BUFFER_ITEMS
            assert task_db_obj.al_buffer_bytes <= Buffer.MAX_BUFFER_BYTES
            assert all(ALBufferItemDB.query().filter_by(id_=id_).first() is None for id_ in lowest_relevance_ids)
            # Restore original buffer bytes
            Buffer.MAX_BUFFER_BYTES = org_buffer_bytes

    def test_clear(self, al_local_buffers):
        buffer = al_local_buffers[0]
        buffer.MAX_BUFFER_BYTES = 100000
        task_db_obj: TaskDB = buffer.task()
        client = Service.query().filter_by(task_id=task_db_obj.task_id).first().client
        task = Task.get(agent=client, db_object_or_id=task_db_obj)
        preds: list = [{
            'state': 'complete',
            'inputs': [{
                'element': 'input_name_1',
                'value': ''
            }, {
                'element': 'input_name_2',
                'value': ''
            }, {
                'element': 'input_name_3',
                'value': ''
            }],
            'outputs': [{
                'element': 'output_name_0',
                'value': {
                    'category': 'name_0_0',
                    'scores': {
                        'name_0_0': 0.7,
                        'name_0_1': 0.15,
                        'name_0_2': 0.1,
                        'name_0_3': 0.05
                    }
                }
            }, {
                'element': 'output_name_1',
                'value': 59.24
            }, {
                'element': 'output_name_2',
                'value': {
                    'category': 'name_2_0',
                    'scores': {
                        'name_2_0': 0.65,
                        'name_2_1': 0.25,
                        'name_2_2': 0.07,
                        'name_2_3': 0.03
                    }
                }
            }],
            'ai_model': '52a4f64d-37b5-49f9-85e6-a793cd8271e5'
        }]
        predictions = PredictionLog.post_batch(data=preds, task=task, environment=AIEnvironment.PRODUCTION)
        buffer.write(items=predictions)
        assert len(ALBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()) > 0
        buffer.clear()
        assert len(ALBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()) == 0


class TestMonBuffer:

    def test_read(self, mon_local_buffers, mon_buffer_items):
        for buffer in mon_local_buffers:
            items = buffer.read()
            assert items == sorted(items, key=(lambda x: x.relevance), reverse=True)
            assert items == sorted(items, key=(lambda x: x.timestamp), reverse=True)
            assert items[0].relevance > items[-1].relevance  # descending order double check
            assert items[0].timestamp > items[-1].timestamp  # descending order double check

    def test_write(self, mon_local_buffers):
        # Delete predictions
        empty_table(MonBufferItemDB)
        for buffer in mon_local_buffers:
            task_db_obj = buffer.task()
            task_db_obj.mon_buffer_items = 0
            task_db_obj.mon_buffer_bytes = 0
            save_to_db(objects=task_db_obj)

        # Save predictions with enough free space in all task buffers
        buffer = mon_local_buffers[0]
        task_db_obj = buffer.task()
        client = Service.query().filter_by(task_id=task_db_obj.task_id).first().client
        task = Task.get(agent=client, db_object_or_id=task_db_obj)

        preds: list = [{
            'state': 'complete',
            'inputs': [{
                'element': 'input_name_1',
                'value': ''
            }, {
                'element': 'input_name_2',
                'value': ''
            }, {
                'element': 'input_name_3',
                'value': ''
            }],
            'outputs': [{
                'element': 'output_name_0',
                'value': {
                    'category': 'name_0_0',
                    'scores': {
                        'name_0_0': 0.7,
                        'name_0_1': 0.15,
                        'name_0_2': 0.1,
                        'name_0_3': 0.05
                    }
                }
            }, {
                'element': 'output_name_1',
                'value': 59.24
            }, {
                'element': 'output_name_2',
                'value': {
                    'category': 'name_2_0',
                    'scores': {
                        'name_2_0': 0.65,
                        'name_2_1': 0.25,
                        'name_2_2': 0.07,
                        'name_2_3': 0.03
                    }
                }
            }],
            'ai_model': '52a4f64d-37b5-49f9-85e6-a793cd8271e5'
        }]
        predictions = PredictionLog.post_batch(data=preds, task=task, environment=AIEnvironment.PRODUCTION)
        buffer.write(items=predictions)
        buffer_items = MonBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()
        assert len(buffer_items) == len(preds)
        buffer_items_size = sum(
            PredictionDB.query().filter_by(prediction_id=x.prediction_id).first().size for x in buffer_items)
        assert buffer_items_size == sum(x.size() for x in predictions)
        assert task_db_obj.mon_buffer_items == len(preds)
        assert task_db_obj.mon_buffer_bytes == buffer_items_size
        pass  # TODO: check values

        # Verify the items with the lowest relevance are deleted when there is no free space in the buffer
        for buffer in mon_local_buffers:
            task_db_obj = buffer.task()
            # Set the 3 least relevant items
            lowest_relevance_ids = [2, 4, 8]
            buffer_items = MonBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()
            now = datetime.utcnow()
            for idx, item in enumerate(buffer_items):
                id_ = idx + 1
                if id_ in lowest_relevance_ids:
                    item.relevance = (now - timedelta(seconds=id_)).timestamp()
                else:
                    item.relevance = (now + timedelta(seconds=id_)).timestamp()
            save_to_db(objects=buffer_items)
            # Set buffer size limit to current usage
            org_buffer_bytes = Buffer.MAX_BUFFER_BYTES
            Buffer.MAX_BUFFER_BYTES = task_db_obj.mon_buffer_bytes
            pass  # TODO: what about buffer items limit?
            # Save predictions
            buffer.write(items=predictions)
            assert task_db_obj.mon_buffer_items <= Buffer.MAX_BUFFER_ITEMS
            assert task_db_obj.mon_buffer_bytes <= Buffer.MAX_BUFFER_BYTES
            assert all(MonBufferItemDB.query().filter_by(id_=id_).first() is None for id_ in lowest_relevance_ids)
            # Restore original buffer bytes
            Buffer.MAX_BUFFER_BYTES = org_buffer_bytes

    def test_clear(self, mon_local_buffers):
        buffer = mon_local_buffers[0]
        buffer.MAX_BUFFER_BYTES = 100000
        task_db_obj = buffer.task()
        client = Service.query().filter_by(task_id=task_db_obj.task_id).first().client
        task = Task.get(agent=client, db_object_or_id=task_db_obj)

        preds: list = [{
            'state': 'complete',
            'inputs': [{
                'element': 'input_name_1',
                'value': ''
            }, {
                'element': 'input_name_2',
                'value': ''
            }, {
                'element': 'input_name_3',
                'value': ''
            }],
            'outputs': [{
                'element': 'output_name_0',
                'value': {
                    'category': 'name_0_0',
                    'scores': {
                        'name_0_0': 0.7,
                        'name_0_1': 0.15,
                        'name_0_2': 0.1,
                        'name_0_3': 0.05
                    }
                }
            }, {
                'element': 'output_name_1',
                'value': 59.24
            }, {
                'element': 'output_name_2',
                'value': {
                    'category': 'name_2_0',
                    'scores': {
                        'name_2_0': 0.65,
                        'name_2_1': 0.25,
                        'name_2_2': 0.07,
                        'name_2_3': 0.03
                    }
                }
            }],
            'ai_model': '52a4f64d-37b5-49f9-85e6-a793cd8271e5'
        }]
        predictions = PredictionLog.post_batch(data=preds, task=task, environment=AIEnvironment.PRODUCTION)
        buffer.write(items=predictions)
        assert len(MonBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()) > 0
        buffer.clear()
        assert len(MonBufferItemDB.query().filter_by(task_id=task_db_obj.task_id).all()) == 0
