from typing import Tuple, Union

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import DECIMAL
from sqlalchemy import Enum
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship

from nexusml.constants import AL_SERVICE_NAME
from nexusml.constants import CL_SERVICE_NAME
from nexusml.constants import FREE_PLAN_ID
from nexusml.constants import INFERENCE_SERVICE_NAME
from nexusml.constants import MONITORING_SERVICE_NAME
from nexusml.constants import TESTING_SERVICE_NAME
from nexusml.database.base import DBModel
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.core import db_rollback
from nexusml.database.core import save_to_db
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import MutableEntity
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import UserDB
from nexusml.database.services import al_client_scopes
from nexusml.database.services import ALServiceSettings
from nexusml.database.services import cl_client_scopes
from nexusml.database.services import CLServiceSettings
from nexusml.database.services import inference_client_scopes
from nexusml.database.services import InferenceServiceSettings
from nexusml.database.services import monitoring_client_scopes
from nexusml.database.services import MonitoringServiceSettings
from nexusml.database.services import Service
from nexusml.database.services import testing_client_scopes
from nexusml.database.services import TestingServiceSettings
from nexusml.database.subscriptions import get_active_subscription
from nexusml.database.subscriptions import Plan
from nexusml.database.utils import set_status
from nexusml.enums import ElementMultiValue
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import ServiceType
from nexusml.enums import TaskType
from nexusml.statuses import al_stopped_status
from nexusml.statuses import cl_stopped_status
from nexusml.statuses import cl_waiting_status
from nexusml.statuses import inference_stopped_status
from nexusml.statuses import monitoring_stopped_status
from nexusml.statuses import Status
from nexusml.statuses import task_created_status
from nexusml.statuses import testing_stopped_status


class TaskDB(MutableEntity):
    """
    Note: don't subclass `OrganizationER` because a Task can exist beyond an Organization

    Attributes:
        - task_id (PK): surrogate key
        - organization_id (FK): surrogate key of the organization that administers the task
        - name: name of the task
        - description: description of the task
        - type_: type of the task (classification, regression, object detection, object segmentation, etc.)
        - status: JSON containing information about the task status
        - icon (FK): surrogate key of the file containing the image
        - prod_model_id: surrogate key of the AI model running in production
        - test_model_id: surrogate key of the AI model running in testing environment
        - max_deployments: maximum number of deployments
        - max_predictions: maximum number of predictions per billing cycle
        - max_gpu_hours: maximum number of GPU hours per billing cycle
        - max_cpu_hours: maximum number of CPU hours per billing cycle
        - max_examples: maximum number of examples
        - space_limit: space quota limit (in bytes)
        - num_deployments: current number of deployments
        - num_predictions: current number of predictions in current billing cycle
        - num_gpu_hours: current number of GPU hours in current billing cycle
        - num_cpu_hours: current number of CPU hours in current billing cycle
        - num_examples: current number of examples
        - space_usage: space quota usage (in bytes)
        - last_al_update (datetime): timestamp of the last update of the Active Learning Service.
        - last_mon_update (datetime): timestamp of the last update of the Monitoring Service.
        - al_buffer_items (int): number of items in the Active Learning buffer.
        - mon_buffer_items (int): number of items in the Monitoring buffer.
        - al_buffer_bytes (int): size of items in the Active Learning buffer in bytes.
        - mon_buffer_bytes (int): size of items in the Monitoring buffer in bytes.
    """

    __tablename__ = 'tasks'
    """
    Columns
    """
    task_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    organization_id = Column(MEDIUMINT(unsigned=True), ForeignKey(OrganizationDB.organization_id, ondelete='SET NULL'))
    name = Column(String(128), nullable=False)
    description = Column(Text)
    # TODO: Consider using `Column('type', Enum(ServiceType))` to keep "type" at SQL level
    type_ = Column(Enum(TaskType))
    status = Column(JSON(none_as_null=True), nullable=False, default=Status(template=task_created_status).to_dict())
    # We set the icon in `database.models.files` due to the circular dependency
    # icon = Column(INTEGER(unsigned=True), ForeignKey(TaskFileDB.file_id, ondelete='SET NULL'))
    # Production AI model
    prod_model_id = Column(INTEGER(unsigned=True))  # add Foreign Key later, as `ai_models` table doesn't exist yet
    # Testing AI model
    test_model_id = Column(INTEGER(unsigned=True))  # add Foreign Key later, as `ai_models` table doesn't exist yet
    # Quota limits
    max_deployments = Column(INTEGER(unsigned=True), nullable=False)
    max_predictions = Column(INTEGER(unsigned=True), nullable=False)
    max_gpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False)
    max_cpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False)
    max_examples = Column(INTEGER(unsigned=True), nullable=False)
    space_limit = Column(BIGINT(unsigned=True), nullable=False)
    # Quota usage
    num_deployments = Column(INTEGER(unsigned=True), nullable=False, default=0)
    num_predictions = Column(INTEGER(unsigned=True), nullable=False, default=0)
    num_gpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False, default=0)
    num_cpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False, default=0)
    num_examples = Column(INTEGER(unsigned=True), nullable=False, default=0)
    space_usage = Column(BIGINT(unsigned=True), nullable=False, default=0)
    # Services info
    last_al_update = Column(DateTime)  # last update of Active Learning Service
    last_mon_update = Column(DateTime)  # last update of Monitoring Service
    # Number of items in buffers
    al_buffer_items = Column(INTEGER(unsigned=True), nullable=False, default=0)
    mon_buffer_items = Column(INTEGER(unsigned=True), nullable=False, default=0)
    # Size of items in buffers
    al_buffer_bytes = Column(INTEGER, nullable=False, default=0)  # JSONs + files
    mon_buffer_bytes = Column(INTEGER, nullable=False, default=0)
    """
    Relationships
    """
    # Parents (Many-to-One relationships)
    organization = relationship('OrganizationDB')

    # Children (One-to-Many relationships)
    # Note: don't pass `backref` because `TaskER` defines a parent (many-to-one) relationship with this class
    # Note: 'metadata' is internally used by SQLAlchemy
    elements = relationship('ElementDB', cascade='all, delete-orphan', lazy='selectin')

    def set_status(self, status: Status, commit: bool = True):
        set_status(db_object=self, status=status, group='task', commit=commit)

    def to_dict(self) -> dict:
        _dict = super().to_dict()
        _dict['status'] = Status.from_dict(self.status).to_dict(include_state=True, expand_status=True)
        return _dict

    def input_elements(self) -> list:
        return [x for x in self.elements if x.element_type == ElementType.INPUT]

    def output_elements(self) -> list:
        return [x for x in self.elements if x.element_type == ElementType.OUTPUT]

    def metadata_elements(self) -> list:
        return [x for x in self.elements if x.element_type == ElementType.METADATA]

    ############################
    # Initialization functions #
    ############################

    def init_task(self, create_services: bool = True, ignore_errors: bool = False):
        # Note: This function used to perform more actions in the past. Now, it only creates services.
        if create_services:
            try:
                self._create_services()
            except Exception as e:
                if ignore_errors:
                    db_rollback()
                else:
                    raise e

    def _create_services(self):

        def _create_service(type_: ServiceType) -> Service:
            service_names = {
                ServiceType.INFERENCE: INFERENCE_SERVICE_NAME,
                ServiceType.CONTINUAL_LEARNING: CL_SERVICE_NAME,
                ServiceType.ACTIVE_LEARNING: AL_SERVICE_NAME,
                ServiceType.MONITORING: MONITORING_SERVICE_NAME,
                ServiceType.TESTING: TESTING_SERVICE_NAME
            }
            service_init_status = {
                ServiceType.INFERENCE: inference_stopped_status,
                ServiceType.CONTINUAL_LEARNING: cl_waiting_status,
                ServiceType.ACTIVE_LEARNING: al_stopped_status,
                ServiceType.MONITORING: monitoring_stopped_status,
                ServiceType.TESTING: testing_stopped_status
            }
            service_stopped_status = {
                ServiceType.INFERENCE: inference_stopped_status,
                ServiceType.CONTINUAL_LEARNING: cl_stopped_status,
                ServiceType.ACTIVE_LEARNING: al_stopped_status,
                ServiceType.MONITORING: monitoring_stopped_status,
                ServiceType.TESTING: testing_stopped_status
            }
            service_settings = {
                ServiceType.INFERENCE: InferenceServiceSettings,
                ServiceType.CONTINUAL_LEARNING: CLServiceSettings,
                ServiceType.ACTIVE_LEARNING: ALServiceSettings,
                ServiceType.MONITORING: MonitoringServiceSettings,
                ServiceType.TESTING: TestingServiceSettings
            }
            client_scopes = {
                ServiceType.INFERENCE: inference_client_scopes,
                ServiceType.CONTINUAL_LEARNING: cl_client_scopes,
                ServiceType.ACTIVE_LEARNING: al_client_scopes,
                ServiceType.MONITORING: monitoring_client_scopes,
                ServiceType.TESTING: testing_client_scopes
            }

            client = ClientDB(organization_id=self.organization_id, name=service_names[type_])
            save_to_db(client)
            client.update_api_key(scopes=client_scopes[type_], never_expire=True)

            if get_active_subscription(organization_id=self.organization_id).plan_id == FREE_PLAN_ID:
                status_templates = service_stopped_status
            else:
                status_templates = service_init_status

            return Service(client_id=client.client_id,
                           task_id=self.task_id,
                           type_=type_,
                           status=Status(template=status_templates[type_]).to_dict(),
                           settings=service_settings[type_]().to_dict())

        new_services = []

        inference_service = Service.filter_by_task_and_type(task_id=self.task_id, type_=ServiceType.INFERENCE)
        if inference_service is None:
            inference_service = _create_service(type_=ServiceType.INFERENCE)
            new_services.append(inference_service)

        cl_service = Service.filter_by_task_and_type(task_id=self.task_id, type_=ServiceType.CONTINUAL_LEARNING)
        if cl_service is None:
            cl_service = _create_service(type_=ServiceType.CONTINUAL_LEARNING)
            new_services.append(cl_service)

        al_service = Service.filter_by_task_and_type(task_id=self.task_id, type_=ServiceType.ACTIVE_LEARNING)
        if al_service is None:
            al_service = _create_service(type_=ServiceType.ACTIVE_LEARNING)
            new_services.append(al_service)

        monitoring_service = Service.filter_by_task_and_type(task_id=self.task_id, type_=ServiceType.MONITORING)
        if monitoring_service is None:
            monitoring_service = _create_service(type_=ServiceType.MONITORING)
            new_services.append(monitoring_service)

        testing_service = Service.filter_by_task_and_type(task_id=self.task_id, type_=ServiceType.TESTING)
        if testing_service is None:
            testing_service = _create_service(type_=ServiceType.TESTING)
            new_services.append(testing_service)

        if new_services:
            save_to_db(new_services)


class TaskER(DBModel):
    """ Represents an entity or an association of a task.

    Attributes:
        - task_id (FK): parent task's surrogate key
    """
    __abstract__ = True

    @declared_attr
    def task_id(cls):
        return Column(MEDIUMINT(unsigned=True), ForeignKey(TaskDB.task_id, ondelete='CASCADE'), nullable=False)

    @declared_attr
    def task(cls):
        return relationship('TaskDB')

    @classmethod
    def filter_by_task(cls, task_id) -> list:
        return cls.query().filter_by(task_id=task_id).all()


class ElementDB(MutableEntity, TaskER):
    """ Client-defined schema elements.

    Attributes:
        - element_id (PK): surrogate key
        - name: name of the element (unique for each task)
        - display_name: name to be shown
        - description: description of the element
        - element_type: input, output, or metadata
        - value_type: type of the values assigned to the element
        - multi_value: allow multiple values with the following formats: "unordered", "ordered", "time_series".
                       If `None`, the element will accept only one value.
        - required: all examples must have a value for the element
        - nullable: allow null values
    """
    __tablename__ = 'elements'
    __table_args__ = (UniqueConstraint('name', 'task_id'),)

    element_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    name = Column(String(32), nullable=False)
    display_name = Column(String(64))
    description = Column(String(256))
    element_type = Column(Enum(ElementType), nullable=False)
    value_type = Column(Enum(ElementValueType), nullable=False)
    multi_value = Column(Enum(ElementMultiValue))
    required = Column(Boolean, nullable=False, default=True)
    nullable = Column(Boolean, nullable=False, default=True)

    # Children (One-to-Many relationships)
    categories = db.relationship('CategoryDB', backref='element', cascade='all, delete-orphan', lazy='dynamic')


class CategoryDB(MutableEntity):
    """ Categories defined for each element.

    Attributes:
        - category_id (PK): surrogate key
        - name: name of the category (unique for each task element)
        - display_name: name to be shown
        - description: description of the category
        - color: hexadecimal color assigned to the category
    """
    __tablename__ = 'categories'
    __table_args__ = (UniqueConstraint('name', 'element_id'),)

    category_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    element_id = Column(INTEGER(unsigned=True), ForeignKey(ElementDB.element_id, ondelete='CASCADE'), nullable=False)
    name = Column(String(32), nullable=False)
    display_name = Column(String(64))
    description = Column(String(256))
    color = Column(String(6))


#############
# Listeners #
#############


@event.listens_for(TaskDB, 'before_insert')
def _set_task_quota_limits_before_insert(mapper, connection, target):
    """
    Sets the quota limits for the task before inserting it into the database if not provided.
    If the associated organization has an active subscription, the limits are set according to the subscription plan.
    Otherwise, the limits are set according to the free plan.
    """

    def _set_quota_limit_if_none(limit_name, limit_value):
        current_value = getattr(target, limit_name)
        if current_value is None:
            setattr(target, limit_name, limit_value)
        elif current_value > limit_value:
            raise ValueError()

    invalid_limits = []

    # Get the active subscription for the organization
    active_subscription = get_active_subscription(organization_id=target.organization_id)
    active_plan = active_subscription.plan if active_subscription is not None else None

    # Get plan quota limits
    if active_plan is not None:
        max_deployments = active_plan.max_deployments
        max_predictions = active_plan.max_predictions
        max_examples = active_plan.max_examples
        max_gpu_hours = active_plan.max_gpu_hours
        max_cpu_hours = active_plan.max_cpu_hours
        space_limit = active_plan.space_limit
    else:
        max_deployments = Plan.free_plan_max_deployments()
        max_predictions = Plan.free_plan_max_predictions()
        max_examples = Plan.free_plan_max_examples()
        max_gpu_hours = Plan.free_plan_max_gpu_hours()
        max_cpu_hours = Plan.free_plan_max_cpu_hours()
        space_limit = Plan.free_plan_space_limit()

    # Set quota limits
    try:
        _set_quota_limit_if_none('max_deployments', max_deployments)
    except ValueError:
        invalid_limits.append('max_deployments')

    try:
        _set_quota_limit_if_none('max_predictions', max_predictions)
    except ValueError:
        invalid_limits.append('max_predictions')

    try:
        _set_quota_limit_if_none('max_examples', max_examples)
    except ValueError:
        invalid_limits.append('max_examples')

    try:
        _set_quota_limit_if_none('max_gpu_hours', max_gpu_hours)
    except ValueError:
        invalid_limits.append('max_gpu_hours')

    try:
        _set_quota_limit_if_none('max_cpu_hours', max_cpu_hours)
    except ValueError:
        invalid_limits.append('max_cpu_hours')

    try:
        _set_quota_limit_if_none('space_limit', space_limit)
    except ValueError:
        invalid_limits.append('space_limit')

    # Verify that all quota limits are valid
    if invalid_limits:
        raise ValueError(f'Invalid quota limits: {", ".join(invalid_limits)}')


#########
# Utils #
#########


def copy_task_to_organization(src_task: TaskDB, agent: Union[UserDB, ClientDB]) -> Tuple[TaskDB, dict]:
    """
    Copies a task (including the schema) to the specified agent's organization.

    WARNING: don't use this function to copy tasks with many elements or categories. The copy is performed at
            Python level instead of SQL level, so it may not fit into memory.

    Returns:
            TaskDB: new copy of the task.
            dict: new primary keys (including children's).
                  For example: {'element_id': {1: 106, 2: 107}}
                               indicates the copy of `element_id = 1` is `element_id = 106`.
    """

    if isinstance(agent, UserDB):
        fixed_values = {
            'created_by_user': agent.user_id,
            'modified_by_user': agent.user_id,
            'synced_by_users': [agent.user_id]
        }
    else:
        fixed_values = {
            'created_by_client': agent.client_id,
            'modified_by_client': agent.client_id,
            'synced_by_clients': [agent.client_id]
        }

    # Duplicate task and schema
    dst_task, pk_maps = src_task.duplicate(fixed_values=fixed_values)
    dst_task.organization_id = agent.organization_id
    db_commit()

    ####################################################################
    # The code below was a first attempt to do everything at SQL level #
    ####################################################################
    # def _copy_direct_children(db_model: TaskER, dst_task: TaskDB, cols_to_exclude: Iterable):
    #     """ Copy the direct children of the source task to the destination task at SQL level. """
    #
    #     # Prepare stuff
    #     if isinstance(agent, UserDB):
    #         creator_vals = [agent.user_id, agent.user_id, text("'[%d]'" % agent.user_id)]
    #         creator_cols = ['created_by_user', 'modified_by_user', 'synced_by_users']
    #     else:
    #         creator_vals = [agent.client_id, agent.client_id, text("'[%d]'" % agent.client_id)]
    #         creator_cols = ['created_by_client', 'modified_by_client', 'synced_by_clients']
    #
    #     # Run SQL statement
    #     # Note: don't use Python-side UUID generator, as it will be called once for all rows (not for each row)
    #     db_model_cols = sorted(list(db_model.columns() - set(cols_to_exclude)))
    #     select_cols = ['task_id', 'uuid'] + db_model_cols + creator_cols
    #     select_stmt = (
    #         select(dst_task.task_id,
    #                mysql_uuid_v4(),
    #                *(getattr(db_model, col) for col in db_model_cols),
    #                *creator_vals)
    #         .where(db_model.task_id == src_task.task_id)
    #     )
    #     insert_stmt = db_model.__table__.insert().from_select(select_cols, select_stmt)
    #     db_execute(insert_stmt)
    #
    # entity_cols = {k for k, v in {**Entity.__dict__, **MutableEntity.__dict__}.items()
    #                if isinstance(v, (Column, declared_attr))}
    #
    # """
    # Copy task definition
    # """
    # task_cols = (TaskDB.columns()
    #              - entity_cols
    #              - {'task_id', 'organization_id', 'prod_model_id', 'prod_model_arn', 'status'})
    # dst_task = TaskDB(organization_id=agent.organization_id, **{col: getattr(src_task, col) for col in task_cols})
    # save_to_db(dst_task)
    #
    # """
    # Copy task schema
    # """
    # # Elements
    # _copy_direct_children(db_model=ElementDB,
    #                       dst_task=dst_task,
    #                       cols_to_exclude=entity_cols.union({'element_id', 'task_id'}))
    # db_commit()
    #
    # # Categories
    # pass
    # db_commit()
    #

    return dst_task, pk_maps
