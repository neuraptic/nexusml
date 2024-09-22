from datetime import datetime

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT
from sqlalchemy.ext.declarative import declared_attr

from nexusml.database.base import BinaryUUID
from nexusml.database.base import Entity
from nexusml.database.organizations import UserDB
from nexusml.database.tasks import TaskDB
from nexusml.database.tasks import TaskER
from nexusml.enums import NotificationEvent
from nexusml.enums import NotificationSource
from nexusml.enums import NotificationStatus
from nexusml.enums import NotificationType


class _Notification(Entity, TaskER):
    """
    Abstract base class for notifications in the system. This class defines the common
    attributes and methods for notifications.

    Attributes:
        source_type (NotificationSource): The type of source that generated the notification.
        event (NotificationEvent): The event type that triggered the notification.
        type (NotificationType): The delivery method of the notification.
        status (NotificationStatus): The current status of the notification.
        read (bool): Indicates if the notification has been read.

    Methods:
        recipient: Declared attribute for the recipient of the notification.
        task_id: Declared attribute for the associated task, if any.
        filter_by_recipient: Class method to filter notifications by recipient.
        filter_by_recipient_task: Class method to filter notifications by recipient and task.
    """
    __abstract__ = True
    source_type = Column(Enum(NotificationSource), nullable=False)
    event = Column(Enum(NotificationEvent), nullable=False)
    # TODO: Consider using `Column('type', Enum(NotificationType))` to avoid using a reserved word
    type = Column(Enum(NotificationType), nullable=False, default=NotificationType.POLLING)
    status = Column(Enum(NotificationStatus), nullable=False, default=NotificationStatus.UNSENT)

    # For future use (currently, notifications are always deleted)
    read = Column(Boolean, nullable=False, default=False)

    @declared_attr
    def recipient(cls):
        return Column(INTEGER(unsigned=True), ForeignKey(UserDB.user_id, ondelete='CASCADE'), nullable=False)

    @declared_attr
    def task_id(cls):
        # Override method to make `task_id` nullable
        return Column(MEDIUMINT(unsigned=True), ForeignKey(TaskDB.task_id, ondelete='CASCADE'))

    @classmethod
    def filter_by_recipient(cls, recipient: int) -> list:
        """
        Filters notifications by recipient.

        Args:
            recipient (int): The recipient's user ID.

        Returns:
            list: A list of notifications for the given recipient.
        """
        return cls.query().filter_by(recipient=recipient).all()

    @classmethod
    def filter_by_recipient_task(cls, recipient: int, task_id: int) -> list:
        """
        Filters notifications by recipient and task.

        Args:
            recipient (int): The recipient's user ID.
            task_id (int): The task ID.

        Returns:
            list: A list of notifications for the given recipient and task.
        """
        return cls.query().filter_by(recipient=recipient, task_id=task_id).all()


class NotificationDB(_Notification):
    """
    Notification database model for individual notifications.

    Attributes:
        notification_id (PK): Surrogate key for the notification.
        task_id (FK): Task's surrogate key (only for task notifications).
        recipient: User's surrogate key.
        created_at: Creation datetime.
        source_type: Source type (item of `NotificationSource`).
        source_uuid: Source UUID (for filtering purposes).
        source_url: Source URL.
        event: "creation", "update", "deletion", or "message".
        type: "polling", "push", or "email".
        status: "unsent", "sending", or "sent".
        read: Whether the notification was read or not.

    Methods:
        filter_by_recipient_source: Filters notifications by recipient and source UUID.
    """
    __tablename__ = 'notifications'

    notification_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    created_at = Column(DATETIME, nullable=False, default=datetime.utcnow)
    source_uuid = Column(BinaryUUID, nullable=False)
    source_url = Column(Text, nullable=False)

    @classmethod
    def filter_by_recipient_source(cls, recipient: int, source_uuid: str) -> list:
        """
        Filters notifications by recipient and source UUID.

        Args:
            recipient (int): The recipient's user ID.
            source_uuid (str): The source UUID.

        Returns:
            list: A list of notifications for the given recipient and source UUID.
        """
        return cls.query().filter_by(recipient=recipient, source_uuid=source_uuid).all()


class AggregatedNotificationDB(_Notification):
    """
    Notification database model for aggregated notifications.

    Attributes:
        agg_notification_id (PK): Surrogate key for the aggregated notification.
        task_id (FK): Task's surrogate key (only for task notifications).
        source_type: Source type (item of `NotificationSource`).
        event: "creation", "update", "deletion", or "message".
        recipient: User's surrogate key.
        since: Oldest datetime among the aggregated notifications.
        count: Number of notifications that were aggregated.
        type: "polling", "push", or "email".
        status: "unsent", "sending", or "sent".
        read: Whether the notification was read or not.
    """
    __tablename__ = 'notifications_agg'
    __table_args__ = (UniqueConstraint('task_id', 'source_type', 'event', 'recipient'),)

    agg_notification_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    since = Column(DATETIME, nullable=False)
    count = Column(MEDIUMINT(unsigned=True), nullable=False)
