"""
TODO: Mock database connection. Otherwise, these tests cannot be considered as unit tests.
"""

from datetime import datetime
from datetime import timedelta
import os
from unittest.mock import MagicMock
import uuid

import pytest

from nexusml.api.jobs.event_jobs import run_mon_service
from nexusml.api.jobs.periodic_jobs import _al_service
from nexusml.api.jobs.periodic_jobs import _bill
from nexusml.api.jobs.periodic_jobs import _export_waitlist_to_csv
from nexusml.api.jobs.periodic_jobs import abort_incomplete_uploads
from nexusml.api.jobs.periodic_jobs import Service
from nexusml.api.utils import config
from nexusml.constants import DATETIME_FORMAT
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.files import TaskFileDB
from nexusml.database.files import TaskUpload
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import WaitList
from nexusml.database.subscriptions import Plan
from nexusml.database.subscriptions import quotas
from nexusml.database.subscriptions import SubscriptionDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import BillingCycle
from nexusml.enums import Currency
from nexusml.enums import TaskFileUse
from nexusml.statuses import MONITORING_ANALYZING_STATUS_CODE
from nexusml.statuses import MONITORING_UNKNOWN_ERROR_STATUS_CODE
from nexusml.statuses import MONITORING_WAITING_STATUS_CODE
from tests.api.utils import db_commit_and_expire

pytestmark = [pytest.mark.unit, pytest.mark.slow]  # TODO: Mark as "fast" when the database connection is mocked


class TestBiller:

    def test__bill(self):
        now = datetime.utcnow()
        ################
        # Set database #
        ################
        # Plans
        empty_table(Plan)
        annual_plan = Plan(
            plan_id=3,  # First two IDs are reserved for default and free plans
            name='Test Annual Plan',
            description='Test Annual Plan Description',
            price=11000,
            currency=Currency.DOLLAR,
            billing_cycle=BillingCycle.ANNUAL,
            max_tasks=20,
            max_deployments=500,
            max_predictions=(10**4),
            max_gpu_hours=(120 * 14),
            max_cpu_hours=(240 * 14),
            max_users=100,
            max_roles=10,
            max_collaborators=50,
            max_clients=5)
        monthly_plan = Plan(
            plan_id=4,  # First two IDs are reserved for default and free plans
            name='Test Monthly Plan',
            description='Test Monthly Plan Description',
            price=1000,
            currency=Currency.DOLLAR,
            billing_cycle=BillingCycle.MONTHLY,
            max_tasks=14,
            max_deployments=250,
            max_predictions=(10**4),
            max_cpu_hours=240,
            max_gpu_hours=120,
            max_users=70,
            max_roles=5,
            max_collaborators=30,
            max_clients=5)
        save_to_db([annual_plan, monthly_plan])

        # Organizations
        empty_table(OrganizationDB)
        orgs = []
        for idx in range(2, 102):  # `organization_id=1` is reserved for the main organization
            org = OrganizationDB(organization_id=idx,
                                 trn=f'org_{idx}_trn',
                                 name=f'Organization {idx}',
                                 domain=f'org{idx}.com',
                                 address=f'Organization Address {idx}')
            orgs.append(org)
        save_to_db(orgs)

        # Subscriptions
        empty_table(SubscriptionDB)
        subscriptions = []
        utc_delta = timedelta(days=(31 * round(len(orgs) * .95)), hours=round(len(orgs) / 2))
        past_utc = now - utc_delta
        for idx, org in enumerate(orgs):
            plan = annual_plan if org.organization_id % 2 == 0 else monthly_plan
            start_utc = past_utc + timedelta(days=(31 * idx), hours=idx)
            end_utcs = {0: None, 1: start_utc + timedelta(days=(366 if plan == annual_plan else 31)), 2: None}
            cancel_utcs = {
                0: None,
                1: start_utc + timedelta(days=180 if plan == annual_plan else 14),
                2: None,
                3: start_utc + timedelta(days=180 if plan == annual_plan else 14),
                4: None
            }
            subscription = SubscriptionDB(organization_id=org.organization_id,
                                          plan_id=plan.plan_id,
                                          start_at=start_utc,
                                          end_at=end_utcs[org.organization_id % len(end_utcs)],
                                          cancel_at=cancel_utcs[org.organization_id % len(cancel_utcs)],
                                          num_predictions=(4 * org.organization_id if start_utc < now else 0),
                                          num_cpu_hours=(2 * org.organization_id if start_utc < now else 0),
                                          num_gpu_hours=(org.organization_id if start_utc < now else 0))
            subscriptions.append(subscription)
        save_to_db(subscriptions)

        # Tasks
        empty_table(TaskDB)
        tasks = []
        for subscription in subscriptions:
            org_id = subscription.organization_id
            for task_id in range(1, 4):
                org_task = TaskDB(organization_id=org_id,
                                  name=f'Organization {org_id} Task {task_id}',
                                  description=f'Description of Organization {org_id} Task {task_id}',
                                  num_predictions=(4 * task_id if subscription.start_at < now else 0),
                                  num_cpu_hours=(2 * task_id if subscription.start_at < now else 0),
                                  num_gpu_hours=(task_id if subscription.start_at < now else 0))
                tasks.append(org_task)
        save_to_db(tasks)

        # Keep track of subscriptions to be billed
        subscriptions_to_bill = [x.subscription_id for x in subscriptions if x.next_bill <= now]
        assert all(x.start_at < now for x in subscriptions if x.subscription_id in subscriptions_to_bill)
        ########
        # Bill #
        ########
        _bill(BillingCycle.MONTHLY)
        _bill(BillingCycle.ANNUAL)
        ##################
        # Check database #
        ##################
        db_commit_and_expire()
        for subscription in SubscriptionDB.query().all():
            # Check periodic (monthly or annual) quota usage
            periodic_quotas = ['predictions', 'cpu', 'gpu']
            usage_fields = [quotas[x]['usage'] for x in periodic_quotas]
            org_tasks = TaskDB.query().filter_by(organization_id=subscription.organization_id).all()
            for usage_field in usage_fields:
                usage = getattr(subscription, usage_field)
                if subscription.start_at < now:
                    if subscription.subscription_id in subscriptions_to_bill:
                        assert usage == 0
                        assert all(getattr(t, usage_field) == 0 for t in org_tasks)
                    else:
                        assert usage > 0
                        assert all(getattr(t, usage_field) > 0 for t in org_tasks)
                else:
                    assert usage == 0
                    assert all(getattr(t, usage_field) == 0 for t in org_tasks)
            # Check next billing date
            if subscription.next_bill is not None:
                next_bill_days = (subscription.next_bill - subscription.start_at).days
            else:
                next_bill_days = -1
            if subscription.subscription_id in subscriptions_to_bill:
                is_active = ((subscription.end_at is None or subscription.end_at > now) and
                             (subscription.cancel_at is None or subscription.cancel_at > now))
                if is_active:
                    if subscription.plan.billing_cycle == BillingCycle.ANNUAL:
                        assert next_bill_days > 366
                    else:
                        assert next_bill_days > 31
                else:
                    assert subscription.next_bill is None
            else:
                if subscription.plan.billing_cycle == BillingCycle.ANNUAL:
                    assert next_bill_days in [365, 366]  # keep in mind leap years
                else:
                    assert next_bill_days in [28, 29, 30, 31]  # keep in mind February


class TestUploadCleaner:

    def test_upload_cleaner(self):
        empty_table(TaskFileDB)
        assert not TaskUpload.query().all()

        now = datetime.utcnow()
        max_delta = config.get('jobs')['abort_upload_after']

        for file_id in range(1, 11):
            file = TaskFileDB(file_id=file_id,
                              task_id=1,
                              filename=f'File {file_id}',
                              size=20,
                              use_for=TaskFileUse.AI_MODEL)
            save_to_db(file)
            delta = timedelta(days=(max_delta * file_id + 1) if file_id % 2 != 0 else file_id % max_delta)
            upload = TaskUpload(file_id=file_id, upload_id=f'upload_id_{file_id}', init_at=now - delta)
            save_to_db(upload)

        abort_incomplete_uploads()
        db_commit_and_expire()

        assert set(x.file_id for x in TaskUpload.query().all()) == {2, 4, 6, 8, 10}


class TestWaitListNotifier:

    def test__export_waitlist_to_csv(self):
        # Prepare database
        empty_table(WaitList)
        entries = []
        uuids = []

        for i in range(1, 11):
            uuid_ = uuid.uuid4()
            uuids.append(str(uuid_))
            entry = WaitList(uuid=uuid_,
                             email=f'test_email_{i}@testorg.com',
                             first_name=f'First Name {i}',
                             last_name=f'Last Name {i}',
                             company=f'Company {i}')
            entries.append(entry)

        save_to_db(entries)

        datetimes = [x.request_date.strftime(DATETIME_FORMAT) for x in entries]

        # Test function
        csv_file = _export_waitlist_to_csv()

        with open(csv_file.name, 'r') as f:
            # Read lines
            lines = f.read().split('\n')
            assert len(lines) == 12  # include header
            # Check header
            expected_header = 'Account ID,Email,First Name,Last Name,Company,Request Date (UTC)'
            assert lines[0] == expected_header
            # Check entries
            for i in range(1, 11):
                expected_line = (f'{uuids[i - 1]},'
                                 f'test_email_{i}@testorg.com,'
                                 f'First Name {i},'
                                 f'Last Name {i},'
                                 f'Company {i},'
                                 f'{datetimes[i - 1]}')
                assert lines[i] == expected_line

        os.remove(csv_file.name)


class TestALService:

    @pytest.fixture
    def mock_task_db_class(self, mocker):
        mock_task_db_ = mocker.patch('nexusml.api.jobs.periodic_jobs.TaskDB')
        return mock_task_db_

    @pytest.fixture
    def mock_service(self, mocker):
        mock_service_ = mocker.patch.object(target=Service,
                                            attribute='filter_by_task_and_type',
                                            return_value=MagicMock())
        return mock_service_

    @pytest.fixture
    def mock_task(self, mocker):
        mock_task_ = mocker.patch('nexusml.api.jobs.periodic_jobs.Task')
        task_ = mocker.MagicMock()
        mock_task_.get.return_value = task_
        return mock_task_

    @pytest.fixture
    def mock_al_buffer_class(self, mocker):
        mock_al_buffer_ = mocker.patch('nexusml.api.jobs.periodic_jobs.ALBuffer')
        al_buffer_instance = MagicMock()
        al_buffer_instance.queue.return_value = mocker.MagicMock()
        al_buffer_instance.update_service_status.return_value = mocker.MagicMock()
        mock_al_buffer_.return_value = al_buffer_instance
        return mock_al_buffer_

    @pytest.fixture
    def mock_active_learning_service(self, mocker):
        mock_active_learning_service_ = mocker.patch('nexusml.api.jobs.periodic_jobs.ActiveLearningService')
        al_service_instance = mocker.MagicMock()
        mock_active_learning_service_.return_value = al_service_instance
        return mock_active_learning_service_

    def test_al_service(self, mocker, mock_task_db_class, mock_service, mock_task, mock_al_buffer_class,
                        mock_active_learning_service):
        """
        Tests the initialization and execution of the active learning service.

        Test Steps:
        1. Mock necessary objects and classes including TaskDB, Service, Task, ALBuffer, BufferIO,
           and ActiveLearningService.
        2. Set up mock task_id and mock task database object with specific active learning service settings.
        3. Call the _al_service function with the mocked task_id.

        Assertions:
        - ActiveLearningService should be initialized once.
        - query method of ALBuffer instance should be called once.
        """
        # Mock objects
        task_id = 'test_task_id'
        al_buffer_instance = mock_active_learning_service.return_value

        task_db_obj = mocker.MagicMock()
        task_db_obj.settings = {
            'services': {
                'active_learning': {
                    'enabled': True,
                    'query_interval': 7,
                    'max_examples_per_query': 50
                }
            }
        }
        mock_task_db_class.get.return_value = task_db_obj

        # Call the task
        _al_service(task_id)

        # Assertions
        mock_active_learning_service.assert_called_once()
        al_buffer_instance.query.assert_called_once()


class TestRunMonService:
    """Test suite for the `run_mon_service` function, validating various service behaviors including
    successful execution, error handling, and status transitions.

    Fixtures:
        mock_mon_buffer: Mock of the mon_buffer object, simulating the input buffer and task details.
        mock_monitoring_service: Mock of the `MonitoringService` class, controlling its behaviors such as
            sample checks and status updates.
        mock_service_filter_by_task_and_type: Mock of the `Service.filter_by_task_and_type` method to simulate
            different service settings and statuses.
    """

    @pytest.fixture
    def mock_mon_buffer(self):
        """Fixture that provides a mock `mon_buffer` object with a mocked `task_id` and buffered items.

        Mocks:
            - mon_buffer.task.task_id: Simulates task identification.
            - mon_buffer.buffer_io.read_items: Provides items for further service analysis.
        """
        mock_buffer: MagicMock = MagicMock()
        mock_buffer.task.return_value.task_id = 'task_id'
        mock_buffer.buffer_io.return_value.read_items.return_value = ['item1', 'item2']
        return mock_buffer

    @pytest.fixture
    def mock_monitoring_service(self, mocker):
        """Fixture that provides a mock `MonitoringService` class and its instance, simulating
        behaviors for checking sample sufficiency, detecting out-of-distribution (OOD) predictions,
        and updating service statuses.

        Mocks:
            - MonitoringService.check_if_enough_samples: Simulates sample validation.
            - MonitoringService.detect_ood_predictions: Detects OOD predictions.
            - MonitoringService.update_service_status: Updates the monitoring service status.
        """
        mock_monitoring_service_ = mocker.patch('nexusml.api.jobs.event_jobs.MonitoringService')
        mock_monitoring_service_instance: MagicMock = MagicMock()
        mock_monitoring_service_instance.check_if_enough_samples.return_value = True
        mock_monitoring_service_instance.detect_ood_predictions = mocker.MagicMock()
        mock_monitoring_service_instance.update_service_status = mocker.MagicMock()
        mock_monitoring_service_.return_value = mock_monitoring_service_instance
        return mock_monitoring_service_

    @pytest.fixture
    def mock_service_filter_by_task_and_type(self, mocker):
        """Fixture that mocks the `Service.filter_by_task_and_type` method to return a service instance
        with predefined settings and initial waiting status code.

        Mocks:
            - Service.filter_by_task_and_type: Simulates service retrieval based on task and type.
        """
        settings: dict = {
            'refresh_interval': 200,
            'ood_predictions': {
                'min_sample': 50,
                'sensitivity': 0.7,
                'smoothing': 0.8
            }
        }
        mock_service_instance: MagicMock = MagicMock()
        mock_service_instance.settings = settings
        mock_service_instance.status = {'code': MONITORING_WAITING_STATUS_CODE}
        mock_service_ = mocker.patch.object(target=Service,
                                            attribute='filter_by_task_and_type',
                                            return_value=mock_service_instance)

        return mock_service_

    def test_run_mon_service_success(self, mock_service_filter_by_task_and_type, mock_mon_buffer,
                                     mock_monitoring_service):
        """Test for successful execution of `run_mon_service`.

        Steps:
            - Initialize MonitoringService and verify status updates.
            - Validate detection of OOD predictions.

        Assertions:
            - MonitoringService is called and initialized.
            - Service status transitions are triggered (ANALYZING -> WAITING).
            - OOD prediction detection is invoked once.
        """
        mon_service_instance = mock_monitoring_service.return_value

        run_mon_service(mock_mon_buffer)

        mock_monitoring_service.assert_called_once()
        mon_service_instance.update_service_status.assert_any_call(code=MONITORING_ANALYZING_STATUS_CODE)
        mon_service_instance.detect_ood_predictions.assert_called_once()
        mon_service_instance.update_service_status.assert_any_call(code=MONITORING_WAITING_STATUS_CODE)

    def test_run_mon_service_no_service_found(self, mock_mon_buffer, mock_monitoring_service,
                                              mock_service_filter_by_task_and_type):
        """Test `run_mon_service` when no service is found.

        Steps:
            - Simulate `Service.filter_by_task_and_type` returning None.

        Assertions:
            - AttributeError is raised.
            - MonitoringService is not called.
        """
        mock_service_filter_by_task_and_type.return_value = None

        with pytest.raises(AttributeError):
            run_mon_service(mock_mon_buffer)

        mock_monitoring_service.assert_not_called()

    def test_run_mon_service_not_enough_samples(self, mock_mon_buffer, mock_monitoring_service,
                                                mock_service_filter_by_task_and_type):
        """Test `run_mon_service` when there are not enough samples.

        Steps:
            - Configure MonitoringService to return insufficient samples.

        Assertions:
            - Service status is not updated.
            - OOD prediction detection is not invoked.
        """
        mon_service_instance = mock_monitoring_service.return_value
        mon_service_instance.check_if_enough_samples.return_value = False

        run_mon_service(mock_mon_buffer)

        mon_service_instance.update_service_status.assert_not_called()
        mon_service_instance.detect_ood_predictions.assert_not_called()

    def test_run_mon_service_status_not_waiting(self, mocker, mock_mon_buffer, mock_monitoring_service,
                                                mock_service_filter_by_task_and_type):
        """Test `run_mon_service` when the service status is not in a waiting state.

        Steps:
            - Mock the service to have a non-waiting status.

        Assertions:
            - Service status and OOD prediction are not updated or invoked.
        """
        mon_service_instance = mock_service_filter_by_task_and_type.return_value
        mon_service_instance.status = {'code': 'some_other_status_code'}
        mock_service_filter_by_task_and_type.filter_by_task_and_type.return_value = mon_service_instance

        instance = mock_monitoring_service.return_value
        instance.update_service_status = mocker.MagicMock()

        run_mon_service(mock_mon_buffer)

        instance.update_service_status.assert_not_called()
        mon_service_instance.detect_ood_predictions.assert_not_called()

    def test_run_mon_service_exception(self, mock_mon_buffer, mock_monitoring_service,
                                       mock_service_filter_by_task_and_type):
        """Test `run_mon_service` to handle an exception during OOD prediction detection.

        Steps:
            - Simulate an exception during `detect_ood_predictions`.

        Assertions:
            - Service status is updated to UNKNOWN_ERROR.
        """
        mon_service_instance = mock_monitoring_service.return_value
        mon_service_instance.detect_ood_predictions.side_effect = Exception('Test exception')

        run_mon_service(mock_mon_buffer)

        mon_service_instance.update_service_status.assert_any_call(code=MONITORING_UNKNOWN_ERROR_STATUS_CODE)
