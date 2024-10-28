from nexusml.api.endpoints import ENDPOINT_TAG
from nexusml.api.resources.tasks import TaskResource
from nexusml.api.schemas.tags import TagRequest
from nexusml.api.schemas.tags import TagResponse
from nexusml.database.tags import TagDB
from nexusml.enums import NotificationSource


class Tag(TaskResource):
    """
    Represents a tag.
    """

    @classmethod
    def db_model(cls):
        """
        Returns the database model associated with the Tag resource.

        Returns:
            TagDB: The database model for tags.
        """
        return TagDB

    @classmethod
    def load_schema(cls):
        """
        Returns the schema used to load data for the Tag resource.

        Returns:
            TagRequest: The schema for loading tag data.
        """
        return TagRequest

    @classmethod
    def dump_schema(cls):
        """
        Returns the schema used to dump data from the Tag resource.

        Returns:
            TagResponse: The schema for dumping tag data.
        """
        return TagResponse

    @classmethod
    def location(cls) -> str:
        """
        Returns the endpoint location for the Tag resource.

        Returns:
            str: The endpoint location for tags.
        """
        return ENDPOINT_TAG

    @classmethod
    def notification_source_type(cls) -> NotificationSource:
        """
        Returns the notification source type for the Tag resource.

        Returns:
            NotificationSource: The notification source type for tags.
        """
        return NotificationSource.TAG
