from typing import List

from flask import jsonify
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs
from marshmallow import fields
from sqlalchemy import and_ as sql_and
from sqlalchemy import or_ as sql_or

from nexusml.api.resources.base import dump
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.examples import Comment
from nexusml.api.resources.examples import Example
from nexusml.api.resources.examples import Shape
from nexusml.api.resources.examples import Slice
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.examples import CommentRequest
from nexusml.api.schemas.examples import CommentResponse
from nexusml.api.schemas.examples import ExampleBatchRequest
from nexusml.api.schemas.examples import ExampleBatchResponse
from nexusml.api.schemas.examples import ExampleRequest
from nexusml.api.schemas.examples import ExampleResponse
from nexusml.api.schemas.examples import ExamplesPage
from nexusml.api.schemas.examples import ShapeRequest
from nexusml.api.schemas.examples import ShapeResponse
from nexusml.api.schemas.examples import SliceRequest
from nexusml.api.schemas.examples import SliceResponse
from nexusml.api.views.base import create_view
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import error_response
from nexusml.api.views.core import process_delete_request
from nexusml.api.views.core import process_get_request
from nexusml.api.views.core import process_post_or_put_request
from nexusml.api.views.utils import get_examples_or_predictions
from nexusml.api.views.utils import paging_url_params
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import SWAGGER_TAG_EXAMPLES
from nexusml.database.core import save_to_db
from nexusml.database.examples import CommentDB
from nexusml.database.examples import ex_tags
from nexusml.database.examples import ExampleDB
from nexusml.database.organizations import UserDB
from nexusml.database.tags import TagDB
from nexusml.enums import ResourceAction

################
# Define views #
################

_ExampleView = create_view(resource_types=[Task, Example])
_CommentView = create_view(resource_types=[Task, Example, Comment])
_ShapeView = create_view(resource_types=[Task, Example, Shape])
_SliceView = create_view(resource_types=[Task, Example, Slice])


class ExamplesView(_ExampleView):
    """
    API endpoint for managing examples within a task. Supports querying examples with various filters and pagination.
    """

    _order_by_fields = ['created_at', 'modified_at', 'activity_at', 'labeling_status']
    _url_params = {
        **paging_url_params(collection_name='examples'), 'order_by':
            fields.String(description='Parameter to order by. Default: "activity_at"'),
        'order':
            fields.String(description='"asc" (ascending) or "desc" (descending). Default: "desc"'),
        'created_at':
            fields.String(description='Examples created at the given datetime'),
        'created_at[min]':
            fields.String(description='Examples created after the given datetime (inclusive)'),
        'created_at[max]':
            fields.String(description='Examples created before the given datetime (inclusive)'),
        'modified_at':
            fields.String(description='Examples modified at the given datetime'),
        'modified_at[min]':
            fields.String(description='Examples modified after the given datetime (inclusive)'),
        'modified_at[max]':
            fields.String(description='Examples modified before the given datetime (inclusive)'),
        'activity_at':
            fields.String(description='Examples with activity at the given datetime'),
        'activity_at[min]':
            fields.String(description='Examples with activity after the given datetime (inclusive)'),
        'activity_at[max]':
            fields.String(description='Examples with activity before the given datetime (inclusive)'),
        'only_with_comments':
            fields.Boolean(description='Only examples with comments'),
        'labeling_status':
            fields.String(description='Labeling status of the example: '
                          '"unlabeled" | "pending_review" | "labeled" | "rejected".'),
        'tag':
            fields.String(description='Tag (use names instead of IDs)')
    }

    @doc(tags=[SWAGGER_TAG_EXAMPLES],
         description='To represent AND and OR operators within the value of a query parameter, use "," for AND and "|" '
         'for OR. For example: \n\n- `tag=<tag_1>,<tag_2>` (AND)\n- `tag=<tag_1>|<tag_2>` (OR)\n\n'
         'For datetimes, use ISO 8601 format (YYYY-MM-DDTHH:MM:SS), e.g.: `created_at=2021-04-28T16:24:03`'
         '\n\nIn addition to the predefined query parameters, input/output/metadata elements can also be '
         'filtered following the format `<element_name>=<value>`.')
    @use_kwargs(_url_params, location='query')
    @marshal_with(ExamplesPage)
    def get(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Retrieves a paginated list of examples based on provided filters and query parameters.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources, including the task.
            **kwargs: Additional keyword arguments for filtering and pagination.

        Returns:
            Response: The response with a paginated list of examples.
        """

        def _build_tag_filters(task: Task, tag_text: str):
            if '|' in tag_text:
                filters = [_build_tag_filters(task=task, tag_text=x) for x in tag_text.split('|')]
                return sql_or(*filters)
            elif ',' in tag_text:
                filters = [_build_tag_filters(task=task, tag_text=x) for x in tag_text.split(',')]
                return sql_and(*filters)
            else:
                tag = TagDB.get_from_id(id_value=tag_text, parent=task.db_object())
                if tag is None:
                    raise ResourceNotFoundError(f'Tag "{tag_text}" not found')
                return ex_tags.c.tag_id == tag.tag_id

        agent = agent_from_token()

        task = resources[-1]
        assert isinstance(task, Task)

        # Check user permissions (token scopes are already validated)
        if isinstance(agent, UserDB):
            Example.check_permissions(organization=task.db_object().organization,
                                      action=ResourceAction.READ,
                                      user=agent,
                                      check_parents=False)  # parent permissions already check when loading parent task

        # Init filters
        filters = []
        db_models = []

        # Comment filters
        if kwargs.get('only_with_comments', False):
            filters.append(CommentDB.comment_id.isnot(None))
            db_models.append(CommentDB)

        # Labeling status filters
        if 'labeling_status' in kwargs:
            lbl_status = kwargs['labeling_status']
            if ',' in lbl_status:
                return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE,
                                      message='Invalid query. An example can only have one labeling status')
            filters.append(ExampleDB.labeling_status.in_(lbl_status.split('|')))

        # Tag filters
        if 'tag' in kwargs:
            filters.append(_build_tag_filters(task=task, tag_text=kwargs['tag']))
            db_models.append(ex_tags)

        # Get examples
        page_resources, page_db_objects = get_examples_or_predictions(agent=agent,
                                                                      task=task,
                                                                      resource_type=Example,
                                                                      predefined_query_params=kwargs,
                                                                      supported_order_by=self._order_by_fields,
                                                                      extra_filters=(filters, db_models))

        # Update examples' sync state
        for db_object in page_db_objects:
            db_object.update_sync_state(agent=agent, commit=False)
        save_to_db(page_db_objects)

        # Build response
        return jsonify(page_resources)

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @use_kwargs(ExampleBatchRequest, location='json')
    @marshal_with(ExampleBatchResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a batch of examples for the specified task.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources, including the task.
            **kwargs: Additional keyword arguments containing the batch data.

        Returns:
            Response: The response with the created examples.
        """
        agent = agent_from_token()

        task = resources[-1]
        assert isinstance(task, Task)

        # Check user permissions (token scopes are already validated)
        if isinstance(agent, UserDB):
            Example.check_permissions(organization=task.db_object().organization,
                                      action=ResourceAction.CREATE,
                                      user=agent,
                                      check_parents=False)  # parent permissions already check when loading parent task

        # Save examples
        examples = Example.post_batch(data=kwargs['batch'], task=task)

        # Prepare response data
        batch = {'batch': Example.dump_batch(examples=examples, task=task, serialize=False)}
        dumped_batch = ExampleBatchResponse().dump(batch)

        # Build and return response
        response = jsonify(dumped_batch)
        response.status_code = HTTP_POST_STATUS_CODE

        return response


class ExampleView(_ExampleView):
    """
    API endpoint for managing a single example within a task. Supports retrieving, updating, and deleting an example.
    """

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    def delete(self, task_id: str, example_id: str, resources: List[Resource]) -> Response:
        """
        Deletes the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.

        Returns:
            Response: The response with delete status.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @marshal_with(ExampleResponse)
    def get(self, task_id: str, example_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Retrieves the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with the example details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @use_kwargs(ExampleRequest, location='json')
    @marshal_with(ExampleResponse)
    def put(self, task_id: str, example_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.
            **kwargs: Additional keyword arguments containing the update data.

        Returns:
            Response: The response with the updated example details.
        """
        example = resources[-1]
        return process_post_or_put_request(agent=example.agent(), resource_or_model=example, json=kwargs)


class CommentsView(_CommentView):
    """
    API endpoint for managing comments within an example. Supports retrieving and creating comments.
    """

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @marshal_with(CommentResponse(many=True))
    def get(self, task_id: str, example_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves comments for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.

        Returns:
            Response: The response with the list of comments.
        """
        example = resources[-1]
        example_collection = example.get_collection(collection_name='comments')
        return jsonify(dump(example_collection))

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @use_kwargs(CommentRequest, location='json')
    @marshal_with(CommentResponse)
    def post(self, task_id: str, example_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a comment for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.
            **kwargs: Additional keyword arguments containing the comment data.

        Returns:
            Response: The response with the created comment details.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=Comment,
                                           parents=resources,
                                           json=kwargs)


class ShapesView(_ShapeView):
    """
    API endpoint for managing shapes within an example. Supports retrieving, creating, and deleting shapes.
    """

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    def delete(self, task_id: str, example_id: str, resources: List[Resource]) -> Response:
        """
        Deletes all shapes for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.

        Returns:
            Response: The response with delete status.
        """
        task = resources[-2]
        example = resources[-1]

        example.clear_collection(collection_name='shapes', notify_to=task.users_with_access())
        example.db_object().trained = False
        example.persist()

        return Response(status=HTTP_DELETE_STATUS_CODE)

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @marshal_with(ShapeResponse(many=True))
    def get(self, task_id: str, example_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves shapes for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.

        Returns:
            Response: The response with the list of shapes.
        """
        example = resources[-1]
        example_collection = example.get_collection(collection_name='shapes')
        return jsonify(dump(example_collection))

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @use_kwargs(ShapeRequest, location='json')
    @marshal_with(ShapeResponse)
    def post(self, task_id: str, example_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a shape for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.
            **kwargs: Additional keyword arguments containing the shape data.

        Returns:
            Response: The response with the created shape details.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=Shape,
                                           parents=resources,
                                           json=kwargs)


class ShapeView(_ShapeView):
    """
    API endpoint for managing a single shape within an example. Supports retrieving, updating, and deleting a shape.
    """

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    def delete(self, task_id: str, example_id: str, shape_id: str, resources: List[Resource]) -> Response:
        """
        Deletes the specified shape.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            shape_id (str): The ID of the shape.
            resources (List[Resource]): List of resources, including the shape.

        Returns:
            Response: The response with delete status.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @marshal_with(ShapeResponse)
    def get(self, task_id: str, example_id: str, shape_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the specified shape.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            shape_id (str): The ID of the shape.
            resources (List[Resource]): List of resources, including the shape.

        Returns:
            Response: The response with the shape details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @use_kwargs(ShapeRequest, location='json')
    @marshal_with(ShapeResponse)
    def put(self, task_id: str, example_id: str, shape_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the specified shape.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            shape_id (str): The ID of the shape.
            resources (List[Resource]): List of resources, including the shape.
            **kwargs: Additional keyword arguments containing the update data.

        Returns:
            Response: The response with the updated shape details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)


class SlicesView(_SliceView):
    """
    API endpoint for managing slices within an example. Supports retrieving, creating, and deleting slices.
    """

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    def delete(self, task_id: str, example_id: str, resources: List[Resource]) -> Response:
        """
        Deletes all slices for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.

        Returns:
            Response: The response with delete status.
        """
        task = resources[-2]
        example = resources[-1]

        example.clear_collection(collection_name='slices', notify_to=task.users_with_access())
        example.db_object().trained = False
        example.persist()

        return Response(status=HTTP_DELETE_STATUS_CODE)

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @marshal_with(SliceResponse(many=True))
    def get(self, task_id: str, example_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves slices for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.

        Returns:
            Response: The response with the list of slices.
        """
        example = resources[-1]
        example_collection = example.get_collection(collection_name='slices')
        return jsonify(dump(example_collection))

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @use_kwargs(SliceRequest, location='json')
    @marshal_with(SliceResponse)
    def post(self, task_id: str, example_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a slice for the specified example.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            resources (List[Resource]): List of resources, including the example.
            **kwargs: Additional keyword arguments containing the slice data.

        Returns:
            Response: The response with the created slice details.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=Slice,
                                           parents=resources,
                                           json=kwargs)


class SliceView(_SliceView):
    """
    API endpoint for managing a single slice within an example. Supports retrieving, updating, and deleting a slice.
    """

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    def delete(self, task_id: str, example_id: str, slice_id: str, resources: List[Resource]) -> Response:
        """
        Deletes the specified slice.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            slice_id (str): The ID of the slice.
            resources (List[Resource]): List of resources, including the slice.

        Returns:
            Response: The response with delete status.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @marshal_with(SliceResponse)
    def get(self, task_id: str, example_id: str, slice_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the specified slice.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            slice_id (str): The ID of the slice.
            resources (List[Resource]): List of resources, including the slice.

        Returns:
            Response: The response with the slice details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_EXAMPLES])
    @use_kwargs(SliceRequest, location='json')
    @marshal_with(SliceResponse)
    def put(self, task_id: str, example_id: str, slice_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the specified slice.

        Args:
            task_id (str): The ID of the task.
            example_id (str): The ID of the example.
            slice_id (str): The ID of the slice.
            resources (List[Resource]): List of resources, including the slice.
            **kwargs: Additional keyword arguments containing the update data.

        Returns:
            Response: The response with the updated slice details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)
