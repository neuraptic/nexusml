from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT

from nexusml.database.base import DBModel
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import UserDB
from nexusml.enums import NotificationType


class AccountSettings(DBModel):
    """
    Represents server-side settings for a user's account.

    Attributes:
        user_id (INTEGER, PK, FK): The user's surrogate key.
        notifications (Enum): The notification type, which can be "polling", "push", or "email".
    """
    __tablename__ = 'account_settings'

    user_id = Column(INTEGER(unsigned=True), ForeignKey(UserDB.user_id, ondelete='CASCADE'), primary_key=True)
    notifications = Column(Enum(NotificationType), nullable=False, default=NotificationType.POLLING)
    pass  # Here we'll be adding future server-side settings


class AccountClientSettings(DBModel):
    """
    Represents client-side settings for a user's account based on specific client versions.

    Attributes:
        user_id (INTEGER, PK, FK): The user's surrogate key.
        client_id (MEDIUMINT, PK, FK): The client's surrogate key.
        client_version (String, PK): The version of the client.
        settings (JSON): The user settings for the specified client version in JSON format.
    """
    __tablename__ = 'account_client_settings'

    user_id = Column(INTEGER(unsigned=True), ForeignKey(UserDB.user_id, ondelete='CASCADE'), primary_key=True)
    client_id = Column(MEDIUMINT(unsigned=True), ForeignKey(ClientDB.client_id, ondelete='CASCADE'), primary_key=True)
    client_version = Column(String(64), primary_key=True)
    settings = Column(JSON(none_as_null=True))
