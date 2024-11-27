from typing import List

from flask import jsonify
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs

from nexusml.api.resources.base import dump
from nexusml.api.resources.base import Resource
from nexusml.api.resources.tags import Tag
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.tags import TagRequest
from nexusml.api.schemas.tags import TagResponse
from nexusml.api.views.base import create_view
from nexusml.api.views.core import process_delete_request
from nexusml.api.views.core import process_get_request
from nexusml.api.views.core import process_post_or_put_request
from nexusml.constants import SWAGGER_TAG_TAGS
from nexusml.database.tags import TagDB

################
# Define views #
################

_View = create_view(resource_types=[Task, Tag])


class TagsView(_View):
    """
    View for handling HTTP requests related to tags associated with a specific task.
    """

    @doc(tags=[SWAGGER_TAG_TAGS])
    @marshal_with(TagResponse(many=True))
    def get(self, task_id: str, resources: List[Resource]):
        """
        Handles GET requests to retrieve tags for a specific task.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): A list of resources associated with the task.

        Returns:
            Response: The response containing a list of tags for the specified task.
        """
        task = resources[-1]
        tags = [
            Tag.get(agent=task.agent(), db_object_or_id=x, parents=resources)
            for x in TagDB.filter_by_task(task_id=task.db_object().task_id)
        ]
        return jsonify(dump(tags))

    @doc(tags=[SWAGGER_TAG_TAGS])
    @use_kwargs(TagRequest, location='json')
    @marshal_with(TagResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handles POST requests to create a new tag for a specific task.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): A list of resources associated with the task.
            **kwargs: JSON data for the new tag.

        Returns:
            Response: The response containing the created tag.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=Tag,
                                           parents=resources,
                                           json=kwargs)


class TagView(_View):
    """
    View for handling HTTP requests related to a specific tag.
    """

    @doc(tags=[SWAGGER_TAG_TAGS])
    def delete(self, task_id: str, tag_id: str, resources: List[Resource]):
        """
        Handles DELETE requests to remove a specific tag.

        Args:
            task_id (str): The ID of the task.
            tag_id (str): The ID of the tag to be deleted.
            resources (List[Resource]): A list of resources associated with the tag.

        Returns:
            Response: The response indicating the deletion status.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TAGS])
    @marshal_with(TagResponse)
    def get(self, task_id: str, tag_id: str, resources: List[Resource]):
        """
        Handles GET requests to retrieve a specific tag.

        Args:
            task_id (str): The ID of the task.
            tag_id (str): The ID of the tag to be retrieved.
            resources (List[Resource]): A list of resources associated with the tag.

        Returns:
            Response: The response containing the requested tag.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TAGS])
    @use_kwargs(TagRequest, location='json')
    @marshal_with(TagResponse)
    def put(self, task_id: str, tag_id: str, resources: List[Resource], **kwargs):
        """
        Handles PUT requests to update a specific tag.

        Args:
            task_id (str): The ID of the task.
            tag_id (str): The ID of the tag to be updated.
            resources (List[Resource]): A list of resources associated with the tag.
            **kwargs: JSON data for updating the tag.

        Returns:
            Response: The response containing the updated tag.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)
