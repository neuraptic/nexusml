""" Background jobs running periodically. """
# TODO: Move the code related to services to `nexusml.engine.services`
import csv
from datetime import datetime
from datetime import timedelta
import os
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

from celery import shared_task
from flask import current_app
from flask import render_template
from flask_mail import Message
from sqlalchemy import and_ as sql_and
from sqlalchemy import func as sql_func
from sqlalchemy import or_ as sql_or
from sqlalchemy import text as sql_text

from nexusml.api.ext import mail
from nexusml.api.jobs.event_jobs import train
from nexusml.api.resources.organizations import User
from nexusml.api.resources.tasks import Task
from nexusml.api.utils import config
from nexusml.api.views.ai import trigger_pending_test_predictions
from nexusml.constants import API_NAME
from nexusml.constants import DATETIME_FORMAT
from nexusml.database.ai import AIModelDB
from nexusml.database.core import db_commit
from nexusml.database.core import db_query
from nexusml.database.core import delete_from_db
from nexusml.database.core import empty_table
from nexusml.database.core import retry_on_deadlock
from nexusml.database.examples import CommentDB
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExCategory
from nexusml.database.files import OrgUpload
from nexusml.database.files import TaskUpload
from nexusml.database.myaccount import AccountSettings
from nexusml.database.notifications import AggregatedNotificationDB
from nexusml.database.notifications import NotificationDB
from nexusml.database.organizations import InvitationDB
from nexusml.database.organizations import UserDB
from nexusml.database.organizations import WaitList
from nexusml.database.services import Service
from nexusml.database.subscriptions import get_active_subscription
from nexusml.database.subscriptions import Plan
from nexusml.database.subscriptions import quotas
from nexusml.database.subscriptions import SubscriptionDB
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.engine.buffers import ALBufferIO
from nexusml.engine.services.active_learning import ActiveLearningService
from nexusml.engine.services.active_learning import ALBuffer
from nexusml.engine.services.continual_learning import ContinualLearningService
from nexusml.enums import BillingCycle
from nexusml.enums import ElementType
from nexusml.enums import NotificationEvent
from nexusml.enums import NotificationSource
from nexusml.enums import NotificationStatus
from nexusml.enums import NotificationType
from nexusml.enums import ServiceType
from nexusml.env import ENV_NOTIFICATION_EMAIL
from nexusml.env import ENV_WAITLIST_EMAIL
from nexusml.statuses import AL_STOPPED_STATUS_CODE
from nexusml.statuses import CL_STOPPED_STATUS_CODE

##############################
# CONTINUAL LEARNING SERVICE #
##############################


@shared_task
def run_cl_service():
    """
    Runs the Continual Learning Service.

    This function catches all exceptions and logs them, so no exceptions are raised to the caller.
    """
    for task_db_obj in TaskDB.query().all():
        try:
            _cl_service.delay(task_id=task_db_obj.task_id)
        except Exception as e:
            print(e)


@shared_task
def _cl_service(task_id: int):
    """
    Executes the Continual Learning Service for a specific task.

    Steps:
        - Retrieves configuration and task details.
        - Checks the subscription status and production AI model.
        - Determines if retraining is needed based on the configuration and usage.
        - Triggers a new training session if necessary.

    Args:
        task_id (int): The ID of the task to process.
    """
    cl_config = config.get('engine')['services']['continual_learning']

    task_db_obj = TaskDB.get(task_id=task_id)

    # Get organization's subscription
    if task_db_obj.organization_id is None:
        return
    subscription = get_active_subscription(organization_id=task_db_obj.organization_id)
    if subscription is None:
        return
    eol_dt = subscription.next_bill

    # Get production AI model
    production_model = AIModelDB.get(model_id=task_db_obj.prod_model_id)
    if production_model is None:
        return

    # Get latest AI model
    latest_model = (AIModelDB.query().filter_by(task_id=task_db_obj.task_id).order_by(
        AIModelDB.created_at.desc()).first())

    # Get service client
    cl_service: Service = Service.filter_by_task_and_type(task_id=task_db_obj.task_id,
                                                          type_=ServiceType.CONTINUAL_LEARNING)

    # Check if service is stopped
    if cl_service.status['code'] == CL_STOPPED_STATUS_CODE:
        return

    # Get task
    task = Task.get(agent=cl_service.client, db_object_or_id=task_db_obj)

    # Get training stats
    trained_cats = _count_examples_per_category(task=task_db_obj, trained=True)
    untrained_cats = _count_examples_per_category(task=task_db_obj, trained=False)

    total_trained = ExampleDB.query().filter_by(task_id=task_db_obj.task_id, trained=True).count()
    total_untrained = ExampleDB.query().filter_by(task_id=task_db_obj.task_id, trained=False).count()

    num_trained_examples = {**dict(trained_cats), 'total': total_trained}
    num_untrained_examples = {**dict(untrained_cats), 'total': total_untrained}

    # Get GPU/CPU quotas by billing cycle
    # They are defined by year
    min_cpu_quota = cl_config['min_cpu_quota']
    min_gpu_quota = cl_config['min_gpu_quota']
    max_cpu_quota = cl_config['max_cpu_quota']
    max_gpu_quota = cl_config['max_gpu_quota']

    # If the billing cycle is monthly, divide them by 12
    if Plan.get(plan_id=subscription.plan_id).billing_cycle == BillingCycle.MONTHLY:
        min_cpu_quota /= 12
        min_gpu_quota /= 12
        max_cpu_quota /= 12
        max_gpu_quota /= 12

    # Trigger a new training session if the AI should be retrained
    should_train = ContinualLearningService.should_train(eol_dt=eol_dt,
                                                         min_days=cl_config['min_days'],
                                                         max_days=cl_config['max_days'],
                                                         min_sample=cl_config['min_sample'],
                                                         max_examples=task_db_obj.max_examples,
                                                         min_cpu_quota=min_cpu_quota,
                                                         min_gpu_quota=min_gpu_quota,
                                                         max_cpu_quota=max_cpu_quota,
                                                         max_gpu_quota=max_gpu_quota,
                                                         cpu_usage=task_db_obj.num_cpu_hours,
                                                         gpu_usage=task_db_obj.num_gpu_hours,
                                                         last_dt=latest_model.created_at,
                                                         last_len=latest_model.training_time,
                                                         last_dev=latest_model.training_device.name,
                                                         num_trained_examples=num_trained_examples,
                                                         num_untrained_examples=num_untrained_examples)
    if not should_train:
        return

    # Train the model
    train(task_uuid=task.uuid(), model_uuid=str(production_model.uuid))


def _count_examples_per_category(task: TaskDB, trained: bool):
    """
    Counts the number of examples per category for a given task and training status.

    Args:
        task (TaskDB): The task for which to count examples.
        trained (bool): The training status of the examples to count.

    Returns:
        list: A list of tuples containing category UUIDs and their respective counts.
    """
    return (db_query(CategoryDB.uuid, sql_func.count(ExCategory.value)).join(
        CategoryDB,
        CategoryDB.category_id == ExCategory.value).join(ElementDB, ElementDB.element_id == ExCategory.element_id).join(
            ExampleDB, ExampleDB.example_id == ExCategory.example_id).filter(
                sql_and(ElementDB.element_type == ElementType.OUTPUT, ExampleDB.task_id == task.task_id,
                        ExampleDB.trained.is_(trained))).group_by(ExCategory.value).all())


###########################
# ACTIVE LEARNING SERVICE #
###########################


@shared_task
def run_al_service():
    """
    Runs the Active Learning Service.

    This function catches all exceptions and logs them, so no exceptions are raised to the caller.
    """
    for task_db_obj in TaskDB.query().all():
        try:
            _al_service.delay(task_id=task_db_obj.task_id)
        except Exception as e:
            print(str(e))


@shared_task
def _al_service(task_id) -> None:
    """
    Runs the Active Learning Service for a specific task.

    Steps:
    1. Retrieve the task and service details from the database.
    2. Check if the active learning service has been stopped.
    3. Initialize the active learning service using the retrieved settings (query interval and maximum query examples).
    4. Perform the active learning query to fetch new examples.
    5. Handle any errors by updating the service status to indicate an unknown error.

    Args:
        task_id (int): The ID of the task for which the active learning service is being run.

    Returns:
        None

    Raises:
        Any exceptions during the query execution are caught and handled internally by updating the service status with
        a generic error code ('90300').
    """
    default_query_interval: int = 7  # query interval in days
    default_max_examples_per_query: int = 50  # maximum number of examples per query

    task_db_obj: TaskDB = TaskDB.get(task_id=task_id)
    al_service_db: Service = Service.filter_by_task_and_type(task_id=task_db_obj.task_id,
                                                             type_=ServiceType.ACTIVE_LEARNING)
    # Check if service is stopped
    if al_service_db.status['code'] == AL_STOPPED_STATUS_CODE:
        return

    # Get service settings
    al_settings: dict = al_service_db.settings

    query_interval: int = al_settings.get('query_interval', default_query_interval)
    max_query_examples: int = al_settings.get('max_examples_per_query', default_max_examples_per_query)

    al_buffer: ALBuffer = ALBuffer(buffer_io=ALBufferIO(task=task_db_obj))

    # Initialize service
    al_service: ActiveLearningService = ActiveLearningService(buffer=al_buffer,
                                                              query_interval=query_interval,
                                                              max_examples_per_query=max_query_examples)

    try:
        al_service.query()

    except Exception:
        # NOTE: be more specific about the error and log the error
        al_service.update_service_status(code='90300')  # "ALUnknownError" status


#################
# NOTIFICATIONS #
#################

_EmailNotification = Union[NotificationDB, AggregatedNotificationDB]


@shared_task
def notify():
    """
    Sends notifications to users.

    Steps:
        1. Marks unsent notifications as "sending".
        2. Retrieves the list of users with email notifications enabled.
        3. Sends notifications to each user.

    Raises:
        Exception: If sending notifications fails.
    """

    def _mark_unsent_notifications(db_model: Type[_EmailNotification], user_ids: List[int]) -> Dict[int, List[int]]:
        """
        Marks unsent notifications as "sending".

        Args:
            db_model (Type[_EmailNotification]): type of notification
            user_ids (List[int]): IDs of the users that have email notifications enabled.

        Returns:
            dict: list of marked unsent notifications for each user.
        """
        surrogate_key = 'notification_id' if db_model == NotificationDB else 'agg_notification_id'

        unsent_notifications = ((db_model.query().filter(db_model.recipient.in_(user_ids),
                                                         db_model.status == NotificationStatus.UNSENT)).all())

        user_unsent_notifications = {}

        for notification in unsent_notifications:
            # Mark notification
            notification.status = NotificationStatus.SENDING

            # Add notification to user
            recipient = notification.recipient
            if recipient not in user_unsent_notifications:
                user_unsent_notifications[recipient] = []
            user_unsent_notifications[recipient].append(getattr(notification, surrogate_key))

        db_commit()

        return user_unsent_notifications

    # Filter unsent notifications and update their status to "sending"
    with current_app.app_context():
        # Filter the users that have email notifications enabled
        # Note: push notifications are not implemented yet.
        user_ids = [x.user_id for x in AccountSettings.query().filter_by(notifications=NotificationType.EMAIL).all()]

        # Filter and mark unsent notifications
        user_unsent_notifications = _mark_unsent_notifications(db_model=NotificationDB, user_ids=user_ids)
        user_unsent_agg_notifications = _mark_unsent_notifications(db_model=AggregatedNotificationDB, user_ids=user_ids)

    # Send pending notifications to every user
    for user_id in set(user_unsent_notifications.keys()).union(set(user_unsent_agg_notifications.keys())):
        _notify_user.delay(user_id=user_id,
                           notifications=user_unsent_notifications.get(user_id, []),
                           agg_notifications=user_unsent_agg_notifications.get(user_id, []))


@shared_task
def _notify_user(user_id: int, notifications: List[int], agg_notifications: List[int]):
    """
    Sends email notifications to a specific user.

    Steps:
        1. Retrieves user and their notifications.
        2. Sends email notifications.
        3. Marks notifications as sent.

    Args:
        user_id (int): The ID of the user to notify.
        notifications (List[int]): List of notification IDs.
        agg_notifications (List[int]): List of aggregated notification IDs.

    Raises:
        Exception: If sending email notifications fails.
    """
    with current_app.app_context():
        user_db_obj = UserDB.get(user_id=user_id)

        notifications = [NotificationDB.get(notification_id=x) for x in notifications]
        agg_notifications = [AggregatedNotificationDB.get(agg_notification_id=x) for x in agg_notifications]

        all_notifications = notifications + agg_notifications

        # If the user doesn't exist anymore, delete notifications and continue
        if user_db_obj is None:
            delete_from_db(all_notifications)
            return

        # Download Auth0 user data
        recipient = User.download_auth0_user_data(auth0_id_or_email=user_db_obj.auth0_id)

        # Email notifications
        _send_email_notifications(sender=(API_NAME, os.environ[ENV_NOTIFICATION_EMAIL]),
                                  recipient=recipient,
                                  notifications=all_notifications)

        # Mark sent notifications
        try:
            for notification in all_notifications:
                notification.status = NotificationStatus.SENT
            db_commit()
        except Exception:
            pass


def _send_email_notifications(sender: Union[str, Tuple[str]], recipient: dict,
                              notifications: Iterable[_EmailNotification]):
    """
    Sends notifications to the user's email.

    Steps:
        1. Prepares the message body based on the notifications.
        2. Sends the email using the configured mail service.

    Args:
        sender (Union[str, Tuple[str]]): The sender of the email.
        recipient (dict): The recipient's information.
        notifications (Iterable[_EmailNotification]): The notifications to send.

    Raises:
        Exception: If sending the email fails.
    """
    aggregated_notifications = [n for n in notifications if isinstance(n, AggregatedNotificationDB)]
    notifications = [n for n in notifications if n not in aggregated_notifications]
    #########################
    # Prepare message body: #
    #########################
    ############
    # 1. Tasks #
    ############
    # Task updates
    updated_tasks = [
        TaskDB.get_from_uuid(n.source_uuid)
        for n in notifications
        if n.source_type == NotificationSource.TASK and n.event == NotificationEvent.UPDATE
    ]
    updated_tasks_agg = [
        n.count
        for n in aggregated_notifications
        if n.source_type == NotificationSource.TASK and n.event == NotificationEvent.UPDATE
    ]
    task_updates_count = updated_tasks_agg[0] if updated_tasks_agg else 0
    ################
    # 2. AI models #
    ################
    # AI models
    new_models = [
        AIModelDB.get_from_uuid(n.source_uuid)
        for n in notifications
        if n.source_type == NotificationSource.AI_MODEL and n.event == NotificationEvent.CREATION
    ]
    new_models_agg = [
        n.count
        for n in aggregated_notifications
        if n.source_type == NotificationSource.AI_MODEL and n.event == NotificationEvent.CREATION
    ]
    model_releases_count = new_models_agg[0] if new_models_agg else 0
    ###############
    # 3. Examples #
    ###############
    # Example comments
    example_comments = [
        CommentDB.get_from_uuid(n.source_uuid)
        for n in notifications
        if n.source_type == NotificationSource.COMMENT and n.event == NotificationEvent.CREATION
    ]
    example_comments = [(ExampleDB.get(example_id=c.example_id).uuid, c) for c in example_comments]
    pass  # TODO: get the identifier(s) defined by the client instead of the example's UUID
    pass  # TODO: get the name of the user who made the comment
    example_comments_agg = [
        n.count
        for n in aggregated_notifications
        if n.source_type == NotificationSource.COMMENT and n.event == NotificationEvent.CREATION
    ]
    example_comments_count = example_comments_agg[0] if example_comments_agg else 0

    # Example updates
    example_updates = [
        ExampleDB.get_from_uuid(n.source_uuid)
        for n in notifications
        if n.source_type == NotificationSource.EXAMPLE and n.event == NotificationEvent.UPDATE
    ]
    example_updates_agg = [
        n.count
        for n in aggregated_notifications
        if n.source_type == NotificationSource.EXAMPLE and n.event == NotificationEvent.UPDATE
    ]
    example_updates_count = example_updates_agg[0] if example_updates_agg else 0

    # Example deletions
    example_deletions = [
        n for n in notifications
        if n.source_type == NotificationSource.EXAMPLE and n.event == NotificationEvent.DELETION
    ]
    pass  # TODO: currently, we cannot retrieve deleted example's fields
    example_deletions_agg = [
        n.count
        for n in aggregated_notifications
        if n.source_type == NotificationSource.EXAMPLE and n.event == NotificationEvent.DELETION
    ]
    example_deletions_count = example_deletions_agg[0] if example_deletions_agg else 0

    # Example creations
    example_creations = [
        ExampleDB.get_from_uuid(n.source_uuid)
        for n in notifications
        if n.source_type == NotificationSource.EXAMPLE and n.event == NotificationEvent.CREATION
    ]
    example_creations_agg = [
        n.count
        for n in aggregated_notifications
        if n.source_type == NotificationSource.EXAMPLE and n.event == NotificationEvent.CREATION
    ]
    example_creations_count = example_creations_agg[0] if example_creations_agg else 0
    #################
    # HTML template #
    #################
    html = render_template('email_notifications.html',
                           app_name=API_NAME,
                           first_name=recipient['first_name'],
                           updated_tasks=updated_tasks,
                           task_updates_count=task_updates_count,
                           released_models=new_models,
                           model_releases_count=model_releases_count,
                           example_comments=example_comments,
                           example_comments_count=example_comments_count,
                           example_updates=example_updates,
                           example_updates_count=example_updates_count,
                           example_deletions=example_deletions,
                           example_deletions_count=example_deletions_count,
                           example_creations=example_creations,
                           example_creations_count=example_creations_count)
    ##############
    # Send email #
    ##############
    subject = f'[{API_NAME}] {recipient["first_name"]}, you have new notifications'
    msg = Message(sender=sender, recipients=[recipient['email']], subject=subject, html=html)
    mail.send(msg)


#################
# SUBSCRIPTIONS #
#################


@shared_task
def bill():
    """
    Bills subscriptions based on their billing cycle.

    Raises:
        Exception: If billing process fails.
    """
    _bill(BillingCycle.MONTHLY)
    _bill(BillingCycle.ANNUAL)


def _bill(cycle: BillingCycle):
    """
    Processes billing for a specific billing cycle.

    Steps:
        1. Filters unbilled subscriptions.
        2. Resets periodic quota usage.
        3. Updates billing dates for active and inactive subscriptions.

    Args:
        cycle (BillingCycle): The billing cycle to process.

    Raises:
        Exception: If billing process fails.
    """
    now = datetime.utcnow()

    # Filter unbilled subscriptions
    unbilled_filter = (Plan.billing_cycle == cycle, SubscriptionDB.next_bill <= now)

    unbilled_subscriptions = (SubscriptionDB.query().join(
        Plan, Plan.plan_id == SubscriptionDB.plan_id).filter(*unbilled_filter))

    active_unbilled = (unbilled_subscriptions.filter(
        sql_or(SubscriptionDB.end_at.is_(None), SubscriptionDB.end_at > now),
        sql_or(SubscriptionDB.cancel_at.is_(None), SubscriptionDB.cancel_at > now)).subquery())

    inactive_unbilled = (unbilled_subscriptions.filter(
        sql_or(SubscriptionDB.end_at <= now, SubscriptionDB.cancel_at <= now)).subquery())

    # Filter unbilled organizations (for task-level usage)
    unbilled_org_ids = (db_query(SubscriptionDB.organization_id).join(
        Plan, Plan.plan_id == SubscriptionDB.plan_id).filter(*unbilled_filter))

    # Restart periodic quota usage
    init_periodic_quota_usage = {
        quotas['predictions']['usage']: 0,
        quotas['cpu']['usage']: 0,
        quotas['gpu']['usage']: 0
    }

    # Bill
    # TODO: in the future, when generating invoices, we can use the difference between `end_at` or `cancel_at`
    #       and `next_bill` to compute the proportional cost to be billed.

    # Note: we build nested subqueries like `db_query(active_unbilled.c.subscription_id)` because MySQL doesn't allow
    #       querying an individual attribute (`Subscription.subscription_id`) of the table being updated. Nested
    #       subqueries force MySQL to create a temporary table.

    # Note: we pass `synchronize_session=False` because default value ("evaluate") raises the following error:
    #       sqlalchemy.exc.InvalidRequestError: Could not evaluate current criteria in Python: "Cannot evaluate Select"

    # Warning: using `synchronize_session=False` causes Python objects to be outdated and their values may not match
    #          the corresponding values stored in the database.

    # Warning: `DATE_ADD` is a MySQL function.

    @retry_on_deadlock
    def _restart_periodic_task_quota_usage():
        (TaskDB.query().filter(TaskDB.organization_id.in_(unbilled_org_ids)).update(init_periodic_quota_usage,
                                                                                    synchronize_session=False))

    @retry_on_deadlock
    def _bill_active():
        (SubscriptionDB.query().filter(SubscriptionDB.subscription_id.in_(db_query(
            active_unbilled.c.subscription_id))).update(
                {
                    **init_periodic_quota_usage, 'next_bill':
                        sql_text(f'DATE_ADD({SubscriptionDB.__tablename__}.next_bill, INTERVAL 1 '
                                 f"{'month' if cycle is BillingCycle.MONTHLY else 'year'})")
                },
                synchronize_session=False))

    @retry_on_deadlock
    def _bill_inactive():
        (SubscriptionDB.query().filter(SubscriptionDB.subscription_id.in_(db_query(
            inactive_unbilled.c.subscription_id))).update({
                **init_periodic_quota_usage, 'next_bill': None
            },
                                                          synchronize_session=False))

    _restart_periodic_task_quota_usage()
    _bill_active()
    _bill_inactive()

    db_commit()


###########
# UPLOADS #
###########


@shared_task
def abort_incomplete_uploads():
    """
    Aborts incomplete uploads that have exceeded the maximum allowed time.

    Raises:
        Exception: If aborting incomplete uploads fails.
    """
    now = datetime.utcnow()
    max_delta = timedelta(days=config.get('jobs')['abort_upload_after'])
    OrgUpload.query().filter(OrgUpload.init_at < (now - max_delta)).delete()
    TaskUpload.query().filter(TaskUpload.init_at < (now - max_delta)).delete()
    db_commit()


#############
# WAIT LIST #
#############


@shared_task
def send_waitlist():
    """
    Sends a CSV file containing all the entries in the wait list.

    Raises:
        Exception: If sending the wait list fails.
    """

    sender = (API_NAME, os.environ[ENV_NOTIFICATION_EMAIL])
    recipient = os.environ[ENV_WAITLIST_EMAIL]

    # Read wait list
    csv_file = _export_waitlist_to_csv()
    if csv_file is None:
        return

    # Prepare message
    subject = f'[{API_NAME}] Wait List'
    html = render_template('waitlist_notification.html', app_name=API_NAME)
    msg = Message(sender=sender, recipients=[recipient], subject=subject, html=html)

    # Attach wait list (CSV file)
    with current_app.open_resource(csv_file.name) as fp:
        attachment_name = f'NexusML-Wait-List-{datetime.utcnow().strftime("%Y-%m-%d")}.csv'
        msg.attach(attachment_name, 'text/csv', fp.read())

    # Send email
    mail.send(msg)

    # Delete CSV file
    os.remove(csv_file.name)


def _export_waitlist_to_csv() -> Optional[tempfile.NamedTemporaryFile]:
    """
    Exports the wait list to a CSV file.

    Returns:
        Optional[tempfile.NamedTemporaryFile]: The temporary CSV file, or None if the wait list is empty.
    """
    db_entries = WaitList.query().order_by(WaitList.id_).all()
    if not db_entries:
        return

    csv_file = tempfile.NamedTemporaryFile(delete=False)
    csv_rows = [[str(x.uuid), x.email, x.first_name, x.last_name, x.company,
                 x.request_date.strftime(DATETIME_FORMAT)] for x in db_entries]

    with open(csv_file.name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Account ID', 'Email', 'First Name', 'Last Name', 'Company', 'Request Date (UTC)'])
        writer.writerows(csv_rows)

    csv_file.close()

    empty_table(WaitList)

    return csv_file


############################
# PENDING TEST PREDICTIONS #
############################


@shared_task
def trigger_all_pending_test_predictions():
    """
    Triggers all pending test predictions in all tasks.

    This function catches all exceptions, so no exceptions are raised to the caller.
    """
    for task in TaskDB.query().yield_per(1000):
        try:
            trigger_pending_test_predictions(task_id=task.task_id, max_attempts=1)
        except Exception:
            continue


#######################
# EXPIRED INVITATIONS #
#######################


@shared_task
def remove_expired_invitations(expiry_days: int = 7) -> None:
    """
    Removes expired invitations from the database based on the given expiration days threshold.

    This function calculates the expiration threshold using the current time and the given number of days.
    It then queries the database for invitations that were created before this threshold and deletes them.

    This function catches all exceptions, so no exceptions are raised to the caller.

    Args:
        expiry_days (int): The number of days after which an invitation is considered expired. Defaults to 7.
    """
    now = datetime.utcnow()
    expiration_threshold = now - timedelta(days=expiry_days)

    try:
        expired_invitations: list = InvitationDB.query().filter(InvitationDB.created_at < expiration_threshold).all()
        if expired_invitations:
            delete_from_db(expired_invitations)
    except Exception:
        pass
