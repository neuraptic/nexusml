# TODO: Try to make this module independent from `nexusml.api`

from datetime import datetime
from typing import Optional

from dateutil.relativedelta import relativedelta
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import DECIMAL
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT
from sqlalchemy.dialects.mysql import SMALLINT
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.orm import relationship

from nexusml.api.utils import config
from nexusml.constants import DEFAULT_PLAN_ID
from nexusml.constants import FREE_PLAN_ID
from nexusml.constants import MYSQL_BIGINT_MAX_UNSIGNED
from nexusml.constants import MYSQL_INT_MAX_UNSIGNED
from nexusml.constants import MYSQL_MEDIUMINT_MAX_UNSIGNED
from nexusml.constants import MYSQL_SMALLINT_MAX_UNSIGNED
from nexusml.database.base import DBModel
from nexusml.database.core import db_query
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import ImmutableEntity
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import OrganizationER
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles
from nexusml.database.organizations import UserDB
from nexusml.database.utils import save_or_ignore_duplicate
from nexusml.enums import BillingCycle
from nexusml.enums import Currency

quotas = {
    'tasks': {
        'usage': 'num_tasks',
        'limit': 'max_tasks',
        'extra': 'extra_tasks'
    },
    'deployments': {
        'usage': 'num_deployments',
        'limit': 'max_deployments',
        'extra': 'extra_deployments'
    },
    'predictions': {
        'usage': 'num_predictions',
        'limit': 'max_predictions',
        'extra': 'extra_predictions'
    },
    'cpu': {
        'usage': 'num_cpu_hours',
        'limit': 'max_cpu_hours',
        'extra': 'extra_cpu_hours'
    },
    'gpu': {
        'usage': 'num_gpu_hours',
        'limit': 'max_gpu_hours',
        'extra': 'extra_gpu_hours'
    },
    'examples': {
        'usage': 'num_examples',
        'limit': 'max_examples',
        'extra': 'extra_examples'
    },
    'space': {
        'usage': 'space_usage',
        'limit': 'space_limit',
        'extra': 'extra_space'
    },
    'users': {
        'usage': 'num_users',
        'limit': 'max_users',
        'extra': 'extra_users'
    },
    'roles': {
        'usage': 'num_roles',
        'limit': 'max_roles',
        'extra': 'extra_roles'
    },
    'collaborators': {
        'usage': 'num_collaborators',
        'limit': 'max_collaborators',
        'extra': 'extra_collaborators'
    },
    'clients': {
        'usage': 'num_clients',
        'limit': 'max_clients',
        'extra': 'extra_clients'
    }
}


def _default_next_bill(context):
    """Calculate the next billing date based on the start date and billing cycle of the plan.

    This function determines the next billing date by retrieving the `start_at` and `plan_id`
    parameters from the context. It checks if the billing cycle of the plan is monthly or annual,
    and calculates the next billing date accordingly.

    Steps:
    1. Retrieve `start_at` and `plan_id` from the context's current parameters.
    2. Determine if the plan's billing cycle is monthly.
    3. Calculate the next billing date by adding one month or one year to the `start_at` date.

    Args:
        context: The SQLAlchemy context with current parameters.

    Returns:
        datetime: The next billing date.
    """
    start_at = context.get_current_parameters()['start_at']
    plan_id = context.get_current_parameters()['plan_id']
    bill_monthly = Plan.get(plan_id=plan_id).billing_cycle == BillingCycle.MONTHLY
    return start_at + (relativedelta(months=1) if bill_monthly else relativedelta(years=1))


#########
# Plans #
#########


class Plan(ImmutableEntity):
    """ Plans

        Attributes:
            - plan_id (PK): surrogate key
            - name: name of the plan (e.g. "Standard", "Plus", "Enterprise", etc.)
            - description: internal description of the plan
            - organization_id (FK): surrogate key of the organization for which the plan was created
                                    (only for custom plans)
            - price: price
            - currency: "dollar" or "euro"
            - billing_cycle: "annual" or "monthly"
            - max_tasks: maximum number of tasks
            - max_deployments: maximum number of deployments
            - max_predictions: maximum number of predictions per billing cycle
            - max_gpu_hours: maximum number of CPU hours per billing cycle
            - max_cpu_hours: maximum number of GPU hours per billing cycle
            - max_examples: maximum number of examples
            - space_limit: space quota limit (in bytes)
            - max_users: maximum number of users
            - max_roles: maximum number of roles
            - max_collaborators: maximum number of external users (those not belonging to the organization)
                                 having access to at least one resource of the organization
            - max_clients: maximum number of clients (apps)
        """
    __tablename__ = 'plans'

    @staticmethod
    def free_plan_max_tasks() -> int:
        """Return Free Plan's maximum number of tasks."""
        return config.get('limits')['quotas']['free_plan']['max_tasks']

    @staticmethod
    def free_plan_max_deployments() -> int:
        """Return Free Plan's maximum number of deployments."""
        return config.get('limits')['quotas']['free_plan']['max_deployments']

    @staticmethod
    def free_plan_max_predictions() -> int:
        """Return Free Plan's maximum number of predictions."""
        return config.get('limits')['quotas']['free_plan']['max_predictions']

    @staticmethod
    def free_plan_max_examples() -> int:
        """Return Free Plan's maximum number of examples."""
        return config.get('limits')['quotas']['free_plan']['max_examples']

    @staticmethod
    def free_plan_max_gpu_hours() -> float:
        """Return Free Plan's maximum number of GPU hours."""
        return config.get('limits')['quotas']['free_plan']['max_gpu_hours']

    @staticmethod
    def free_plan_max_cpu_hours() -> float:
        """Return Free Plan's maximum number of CPU hours."""
        return config.get('limits')['quotas']['free_plan']['max_cpu_hours']

    @staticmethod
    def free_plan_space_limit() -> int:
        """Return Free Plan's space limit."""
        return config.get('limits')['quotas']['free_plan']['space_limit']

    plan_id = Column(SMALLINT(unsigned=True), primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False)
    description = Column(Text)
    organization_id = Column(MEDIUMINT(unsigned=True), ForeignKey(OrganizationDB.organization_id, ondelete='CASCADE'))
    price = Column(DECIMAL(precision=9, scale=2, asdecimal=False), nullable=False)
    currency = Column(Enum(Currency), nullable=False)
    billing_cycle = Column(Enum(BillingCycle), nullable=False)

    # Quota limits
    max_tasks = Column(SMALLINT(unsigned=True), nullable=False)
    max_deployments = Column(MEDIUMINT(unsigned=True), nullable=False)
    max_predictions = Column(INTEGER(unsigned=True), nullable=False, default=free_plan_max_predictions)
    max_gpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False),
                           nullable=False,
                           default=free_plan_max_gpu_hours)
    max_cpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False),
                           nullable=False,
                           default=free_plan_max_cpu_hours)
    max_examples = Column(INTEGER(unsigned=True), nullable=False, default=free_plan_max_examples)
    space_limit = Column(BIGINT(unsigned=True), nullable=False, default=free_plan_space_limit)
    max_users = Column(SMALLINT(unsigned=True), nullable=False)
    max_roles = Column(SMALLINT(unsigned=True), nullable=False)
    max_collaborators = Column(SMALLINT(unsigned=True), nullable=False)
    max_clients = Column(SMALLINT(unsigned=True), nullable=False)


class PlanExtra(ImmutableEntity):
    """ Extra quotas that can be added to Plans

        Attributes:
            - extra_id (PK): surrogate key
            - plan_id (FK): surrogate key of the plan in which extra quotas are defined
            - price: price (currency is inherited from parent Plan)
            - extra_tasks: extra tasks
            - extra_deployments: extra deployments
            - extra_predictions: extra predictions in current billing cycle
            - extra_gpu_hours: extra CPU hours in current billing cycle
            - extra_cpu_hours: extra GPU hours in current billing cycle
            - extra_examples: extra examples
            - extra_space: extra space (in bytes)
            - extra_users: extra users
            - extra_roles: extra roles
            - extra_collaborators: extra collaborators
            - extra_clients: extra clients
    """
    __tablename__ = 'plan_extras'

    extra_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    plan_id = Column(SMALLINT(unsigned=True), ForeignKey(Plan.plan_id, ondelete='CASCADE'), nullable=False)
    price = Column(DECIMAL(precision=9, scale=2, asdecimal=False), nullable=False)
    extra_tasks = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    extra_deployments = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    extra_predictions = Column(INTEGER(unsigned=True), nullable=False, default=0)
    extra_gpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False, default=0)
    extra_cpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False, default=0)
    extra_examples = Column(INTEGER(unsigned=True), nullable=False, default=0)
    extra_space = Column(BIGINT(unsigned=True), nullable=False, default=0)
    extra_users = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    extra_roles = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    extra_collaborators = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    extra_clients = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)


#################
# Subscriptions #
#################


class SubscriptionDB(ImmutableEntity, OrganizationER):
    """ Subscriptions to Plans

    Attributes:
        - subscription_id (PK): surrogate key
        - organization_id (FK): surrogate key of the organization to which the subscription belongs
        - plan_id (FK): surrogate key of the plan to which the organization is subscribed
        - start_at: start date
        - end_at: end date
        - cancel_at: cancellation date
        - next_bill: next billing date
        - num_tasks: current number of tasks
        - num_deployments: current number of deployments
        - num_predictions: current number of predictions in current billing cycle
        - num_gpu_hours: current number of CPU hours in current billing cycle
        - num_cpu_hours: current number of GPU hours in current billing cycle
        - num_examples: current number of examples
        - space_usage: space usage (in bytes)
        - num_users: current number of users
        - num_roles: current number of roles
        - num_collaborators: current number of collaborators
                             (see `max_collaborators` for more info about a "collaborator")
        - num_clients: current number of clients (apps)
    """
    __tablename__ = 'subscriptions'

    subscription_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    plan_id = Column(SMALLINT(unsigned=True), ForeignKey(Plan.plan_id, ondelete='CASCADE'), nullable=False)

    # Periods
    start_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_at = Column(DateTime)
    cancel_at = Column(DateTime)
    next_bill = Column(DateTime, default=_default_next_bill)

    # Quota usage
    num_tasks = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    num_deployments = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    num_predictions = Column(INTEGER(unsigned=True), nullable=False, default=0)
    num_gpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False, default=0)
    num_cpu_hours = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False, default=0)
    num_examples = Column(INTEGER(unsigned=True), nullable=False, default=0)
    space_usage = Column(BIGINT(unsigned=True), nullable=False, default=0)
    num_users = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    num_roles = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    num_collaborators = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)
    num_clients = Column(MEDIUMINT(unsigned=True), nullable=False, default=0)

    # Parents (Many-to-One relationships)
    plan = relationship('Plan')

    # Children (One-to-Many relationships)
    discounts = relationship('SubscriptionDiscount',
                             backref='subscription',
                             cascade='all, delete-orphan',
                             lazy='selectin')
    extras = relationship('SubscriptionExtra', cascade='all, delete-orphan', lazy='selectin')


class SubscriptionDiscount(ImmutableEntity):
    """ Discounts applied to Subscriptions

        Attributes:
            - discount_id (PK): surrogate key
            - subscription_id (FK): surrogate key of the subscription to which the discount is applied
            - percentage: number between 1 and 100
            - start_at: start date
            - end_at: end date
            - cancel_at: cancellation date
    """
    __tablename__ = 'subscription_discounts'

    discount_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    subscription_id = Column(MEDIUMINT(unsigned=True),
                             ForeignKey(SubscriptionDB.subscription_id, ondelete='CASCADE'),
                             nullable=False)
    percentage = Column(TINYINT(unsigned=True), nullable=False)  # TODO: range constraint (1<=x<=100)
    start_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_at = Column(DateTime)
    cancel_at = Column(DateTime)


class SubscriptionExtra(ImmutableEntity):
    """ Extra quotas added to the subscription

        Attributes:
            - sub_extra_id (PK): surrogate key
            - subscription_id (FK): subscription's surrogate key
            - extra_id (FK): extra's surrogate key
            - start_at: start date
            - end_at: end date
            - cancel_at: cancellation date
    """
    __tablename__ = 'subscription_extras'

    sub_extra_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    subscription_id = Column(MEDIUMINT(unsigned=True),
                             ForeignKey(SubscriptionDB.subscription_id, ondelete='CASCADE'),
                             nullable=False)
    extra_id = Column(MEDIUMINT(unsigned=True), ForeignKey(PlanExtra.extra_id, ondelete='CASCADE'), nullable=False)
    start_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_at = Column(DateTime)
    cancel_at = Column(DateTime)

    # Parents (Many-to-One relationships)
    subscription = relationship('SubscriptionDB')
    extra = relationship('PlanExtra')


###################
# API rate limits #
###################


class _RateLimits(DBModel):
    """ API rate limits

    Attributes:
        - requests_per_day: number of requests per day
        - requests_per_hour: number of requests per hour
        - requests_per_minute: number of requests per minute
        - requests_per_second: number of requests per second
    """
    __abstract__ = True

    @staticmethod
    def _default_requests_per_day() -> int:
        """Return the default number of requests per day."""
        return config.get('limits')['requests']['requests_per_day']

    @staticmethod
    def _default_requests_per_hour() -> int:
        """Return the default number of requests per hour."""
        return config.get('limits')['requests']['requests_per_hour']

    @staticmethod
    def _default_requests_per_minute() -> int:
        """Return the default number of requests per minute."""
        return config.get('limits')['requests']['requests_per_minute']

    @staticmethod
    def _default_requests_per_second() -> int:
        """Return the default number of requests per second."""
        return config.get('limits')['requests']['requests_per_second']

    requests_per_day = Column(INTEGER(unsigned=True), nullable=False, default=_default_requests_per_day)
    requests_per_hour = Column(INTEGER(unsigned=True), nullable=False, default=_default_requests_per_hour)
    requests_per_minute = Column(INTEGER(unsigned=True), nullable=False, default=_default_requests_per_minute)
    requests_per_second = Column(MEDIUMINT(unsigned=True), nullable=False, default=_default_requests_per_second)


class UserRateLimits(_RateLimits):
    """ User API rate limits

    Attributes:
        - user_id (PK, FK): user's surrogate key
    """
    __tablename__ = 'user_rate_limits'

    user_id = Column(INTEGER(unsigned=True), ForeignKey(UserDB.user_id, ondelete='CASCADE'), primary_key=True)


class RoleRateLimits(_RateLimits):
    """ Role API rate limits

    Attributes:
        - role_id (PK, FK): role's surrogate key
    """
    __tablename__ = 'role_rate_limits'

    role_id = Column(MEDIUMINT(unsigned=True), ForeignKey(RoleDB.role_id, ondelete='CASCADE'), primary_key=True)


class ClientRateLimits(_RateLimits):
    """ Client API rate limits

    Attributes:
        - client_id (PK, FK): client's surrogate key
    """
    __tablename__ = 'client_rate_limits'

    client_id = Column(MEDIUMINT(unsigned=True), ForeignKey(ClientDB.client_id, ondelete='CASCADE'), primary_key=True)


def get_active_subscription(organization_id: int) -> Optional[SubscriptionDB]:
    """Retrieve the active subscription for a given organization.

    This function fetches all subscriptions for the specified organization and filters them
    to find the active subscription, if any. A subscription is considered active if it has started,
    and neither ended nor canceled.

    Steps:
    1. Fetch all subscriptions for the specified organization.
    2. Filter subscriptions to find the active one.
    3. Ensure there is at most one active subscription.

    Args:
        organization_id (int): The ID of the organization.

    Returns:
        Optional[SubscriptionDB]: The active subscription if found, otherwise None.
    """

    def _is_active(subscription: SubscriptionDB) -> bool:
        started = subscription.start_at <= datetime.utcnow()
        ended = subscription.end_at is not None and subscription.end_at <= datetime.utcnow()
        canceled = subscription.cancel_at is not None and subscription.cancel_at <= datetime.utcnow()
        return started and not (ended or canceled)

    org_subscriptions = SubscriptionDB.query().filter_by(organization_id=organization_id).all()
    active_subscriptions = [x for x in org_subscriptions if _is_active(subscription=x)]
    assert len(active_subscriptions) <= 1
    return active_subscriptions[0] if active_subscriptions else None


def get_user_roles_rate_limits(user_id: int) -> Optional[RoleRateLimits]:
    """Retrieve the API rate limits for the roles assigned to a user.

    This function joins the RoleRateLimits with the user_roles table to fetch the rate limits
    for the roles assigned to the specified user. It currently assumes single-role users.

    Steps:
    1. Query the RoleRateLimits joined with user_roles.
    2. Filter by the specified user ID.
    3. Ensure there is at most one set of rate limits.

    Args:
        user_id (int): The ID of the user.

    Returns:
        Optional[RoleRateLimits]: The rate limits for the user's role if found, otherwise None.
    """
    role_limits = (db_query(RoleRateLimits).join(
        user_roles, user_roles.c.role_id == RoleRateLimits.role_id).filter(user_roles.c.user_id == user_id).all())

    if not role_limits:
        return
    assert len(role_limits) == 1  # we currently assume single-role users
    return role_limits[0]


#################
# Default plans #
#################


def create_default_plans():
    """
    Creates the default plans if they do not already exist.

    Raises:
        IntegrityError: If an integrity error other than a unique constraint violation occurs.
    """
    # Create the Default Plan with no quota limits
    default_plan = Plan(
        plan_id=DEFAULT_PLAN_ID,
        name='Default Plan',
        description='Default plan with no quota limits',
        price=0,
        currency=Currency.DOLLAR,
        billing_cycle=BillingCycle.MONTHLY,
        max_tasks=MYSQL_SMALLINT_MAX_UNSIGNED,
        max_deployments=MYSQL_MEDIUMINT_MAX_UNSIGNED,
        max_predictions=MYSQL_INT_MAX_UNSIGNED,
        max_gpu_hours=99999.99,  # DECIMAL(precision=7, scale=2)
        max_cpu_hours=99999.99,  # DECIMAL(precision=7, scale=2)
        max_examples=MYSQL_INT_MAX_UNSIGNED,
        space_limit=MYSQL_BIGINT_MAX_UNSIGNED,
        max_users=MYSQL_SMALLINT_MAX_UNSIGNED,
        max_roles=MYSQL_SMALLINT_MAX_UNSIGNED,
        max_collaborators=MYSQL_SMALLINT_MAX_UNSIGNED,
        max_clients=MYSQL_SMALLINT_MAX_UNSIGNED)

    save_or_ignore_duplicate(default_plan)

    # Create the Free Plan for new organizations
    _free_plan_quotas = config.get('limits')['quotas']['free_plan']

    free_plan = Plan(plan_id=FREE_PLAN_ID,
                     name='Free Plan',
                     description='Free plan for new organizations',
                     price=0,
                     currency=Currency.DOLLAR,
                     billing_cycle=BillingCycle.MONTHLY,
                     max_tasks=_free_plan_quotas['max_tasks'],
                     max_deployments=_free_plan_quotas['max_deployments'],
                     max_predictions=_free_plan_quotas['max_predictions'],
                     max_gpu_hours=_free_plan_quotas['max_gpu_hours'],
                     max_cpu_hours=_free_plan_quotas['max_cpu_hours'],
                     max_examples=_free_plan_quotas['max_examples'],
                     space_limit=_free_plan_quotas['space_limit'],
                     max_users=_free_plan_quotas['max_users'],
                     max_roles=_free_plan_quotas['max_roles'],
                     max_collaborators=_free_plan_quotas['max_collaborators'],
                     max_clients=_free_plan_quotas['max_apps'])

    save_or_ignore_duplicate(free_plan)
