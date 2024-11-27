from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_NOTIFICATION
from nexusml.api.resources.base import Resource
from nexusml.api.schemas.myaccount import AggregatedNotificationSchema
from nexusml.api.schemas.myaccount import NotificationSchema
from nexusml.database.notifications import AggregatedNotificationDB
from nexusml.database.notifications import NotificationDB


class Notification(Resource):
    """
    Represents an individual notification.
    """

    @classmethod
    def db_model(cls):
        """
        Returns the database model for the notification resource.

        Returns:
            NotificationDB: The database model for notifications.
        """
        return NotificationDB

    @classmethod
    def load_schema(cls):
        """
        Returns the schema for loading notifications.

        Returns:
            NotificationSchema: The schema used to load notifications.
        """
        return NotificationSchema

    @classmethod
    def dump_schema(cls):
        """
        Returns the schema for dumping notifications.

        Returns:
            NotificationSchema: The schema used to dump notifications.
        """
        return NotificationSchema

    @classmethod
    def location(cls) -> str:
        """
        Returns the endpoint location for notifications.

        Returns:
            str: The endpoint location for notifications.
        """
        return ENDPOINT_MYACCOUNT_NOTIFICATION


class AggregatedNotification(Resource):
    """
    Represents an aggregated notification.
    """

    @classmethod
    def db_model(cls):
        """
        Returns the database model for the aggregated notification resource.

        Returns:
            AggregatedNotificationDB: The database model for aggregated notifications.
        """
        return AggregatedNotificationDB

    @classmethod
    def load_schema(cls):
        """
        Returns the schema for loading aggregated notifications.

        Returns:
            AggregatedNotificationSchema: The schema used to load aggregated notifications.
        """
        return AggregatedNotificationSchema

    @classmethod
    def dump_schema(cls):
        """
        Returns the schema for dumping aggregated notifications.

        Returns:
            AggregatedNotificationSchema: The schema used to dump aggregated notifications.
        """
        return AggregatedNotificationSchema

    @classmethod
    def location(cls) -> str:
        """
        Returns the endpoint location for aggregated notifications.

        Returns:
            str: The endpoint location for aggregated notifications.
        """
        return ENDPOINT_MYACCOUNT_NOTIFICATION
