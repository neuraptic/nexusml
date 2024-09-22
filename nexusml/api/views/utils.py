from datetime import datetime
import itertools
from typing import List, Optional, Tuple, Type, Union

from flask import request
from marshmallow import fields
from marshmallow import validate
from marshmallow import ValidationError
from sqlalchemy import and_ as sql_and
from sqlalchemy import cast as sql_cast
from sqlalchemy import or_ as sql_or
from sqlalchemy import String
from sqlalchemy import text as sql_text
from sqlalchemy.sql import ClauseElement

from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.base import UnprocessableRequestError
from nexusml.api.resources.examples import Example
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.base import PageSchema
from nexusml.api.utils import config
from nexusml.api.views.core import get_page_db_objects
from nexusml.constants import DATETIME_FORMAT
from nexusml.database.ai import PredBoolean
from nexusml.database.ai import PredCategory
from nexusml.database.ai import PredFloat
from nexusml.database.ai import PredictionDB
from nexusml.database.ai import PredInteger
from nexusml.database.ai import PredScores
from nexusml.database.ai import PredText
from nexusml.database.ai import PredValue
from nexusml.database.base import DBModel
from nexusml.database.base import Entity
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExBoolean
from nexusml.database.examples import ExCategory
from nexusml.database.examples import ExFloat
from nexusml.database.examples import ExInteger
from nexusml.database.examples import ExText
from nexusml.database.examples import ExValue
from nexusml.database.organizations import Agent
from nexusml.database.organizations import UserDB
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.enums import ElementType
from nexusml.enums import ResourceAction

RANGE_MIN = '[min]'
RANGE_MAX = '[max]'


def paging_url_params(collection_name: str) -> dict:
    """
    Generates paging URL parameters for a given collection name.

    This function creates a dictionary of paging URL parameters, including 'page', 'per_page',
    and 'total_count'. It includes validation to ensure 'per_page' does not exceed a maximum
    limit and provides default values for missing parameters.

    Steps:
    1. Define a validation function for 'per_page' to check against maximum limit.
    2. Define a default value function for 'per_page'.
    3. Create and return a dictionary of URL parameters with descriptions and validations.

    Args:
        collection_name (str): The name of the collection for which paging parameters are generated.

    Returns:
        dict: A dictionary containing paging URL parameters.
    """

    def validate_per_page(n):
        max_per_page = config.get('views')['max_items_per_page']

        if n < 1:
            raise ValidationError('Page size must be positive')

        if n > max_per_page:
            raise ValidationError(f'Maximum number of {collection_name} per page ({max_per_page}) exceeded')

    def default_per_page() -> int:
        return config.get('views')['default_items_per_page']

    return {
        'page':
            fields.Integer(description='Page number', validate=validate.Range(min=1), missing=1),
        'per_page':
            fields.Integer(description=f'Number of {collection_name} per page.',
                           validate=validate_per_page,
                           missing=default_per_page),
        'total_count':
            fields.Boolean(description=(f'Return the total number of {collection_name} '
                                        '(including all pages)'),
                           missing=False)
    }


def build_field_filter(db_model: Type[Entity], field: str, value: str):
    """
    Constructs a SQLAlchemy filter for a specific field and value in a database model.

    This function builds a filter based on the provided field and value. It supports
    '|' for OR operations and ',' for AND operations within the value string.

    Steps:
    1. Check for '|' in the value and create an OR filter if present.
    2. Check for ',' in the value and create an AND filter if present.
    3. Return a simple equality filter if no special characters are found.

    Args:
        db_model (Type[Entity]): The database model type.
        field (str): The field name to filter on.
        value (str): The value to filter by.

    Returns:
        ClauseElement: A SQLAlchemy filter clause.
    """
    if '|' in value:
        filters = [build_field_filter(db_model, field, v) for v in value.split('|')]
        return sql_or(*filters)
    elif ',' in value:
        filters = [build_field_filter(db_model, field, v) for v in value.split(',')]
        return sql_and(*filters)
    else:
        return getattr(db_model, field) == value


def build_datetime_filter(db_model: Type[Entity], datetime_field: str, datetime_value: str):
    """
    Constructs a SQLAlchemy datetime filter for a specific field and value in a database model.

    This function builds a filter for datetime fields, supporting ranges with '[min]' and '[max]'
    suffixes and '|' for OR operations.

    Steps:
    1. Check for '|' in the datetime value and create an OR filter if present.
    2. Check for ',' in the datetime value and create an AND filter if present.
    3. Handle '[min]' and '[max]' suffixes to create range filters.
    4. Return a simple equality filter for datetime values.

    Args:
        db_model (Type[Entity]): The database model type.
        datetime_field (str): The datetime field name to filter on.
        datetime_value (str): The datetime value to filter by.

    Returns:
        ClauseElement: A SQLAlchemy filter clause.
    """
    if '|' in datetime_value:
        filters = [build_datetime_filter(db_model, datetime_field, dt) for dt in datetime_value.split('|')]
        return sql_or(*filters)
    elif ',' in datetime_value:
        filters = [build_datetime_filter(db_model, datetime_field, dt) for dt in datetime_value.split(',')]
        return sql_and(*filters)
    else:
        if datetime_value.strip().lower().endswith('z'):
            datetime_value = datetime_value[:-1]
        if RANGE_MIN in datetime_field:
            datetime_field = datetime_field[:datetime_field.index(RANGE_MIN)]
            return getattr(db_model, datetime_field) >= datetime.strptime(datetime_value, DATETIME_FORMAT)
        elif RANGE_MAX in datetime_field:
            datetime_field = datetime_field[:datetime_field.index(RANGE_MAX)]
            return getattr(db_model, datetime_field) <= datetime.strptime(datetime_value, DATETIME_FORMAT)
        else:
            return getattr(db_model, datetime_field) == datetime.strptime(datetime_value, DATETIME_FORMAT)


def build_element_value_filters(resource_type: Type[Union[Example, PredictionLog]], task: Task,
                                url_params: dict) -> Tuple[list, list]:
    """
    Builds a list of filters and required database models based on element values for a resource type.

    This function constructs filters for each element based on the provided URL parameters. It supports
    multiple filter formats, including regular expressions for text fields.

    Note: we use elements' resource models instead of database models to use cache and avoid hitting database

    Steps:
    1. Define an inner function to create element filters.
    2. Iterate over URL parameters to generate filters for each element.
    3. Add filters and corresponding database models to the result lists.

    Args:
        resource_type (Type[Union[Example, PredictionLog]]): The resource type (Example or PredictionLog).
        task (Task): The task associated with the resource.
        url_params (dict): The URL parameters for filtering.

    Returns:
        Tuple[list, list]: A tuple containing a list of filters and a list of required database models.
    """

    # TODO: consider using `[min]`-`[max]` syntax to filter by intervals as `build_datetime_filter()` does

    def _element_filter(element: ElementDB,
                        value_db_model: Union[ExValue, PredValue],
                        value_string: str,
                        idx: int = None):

        supported_db_models = (ExBoolean, ExInteger, ExFloat, ExText, ExCategory, PredBoolean, PredInteger, PredFloat,
                               PredText, PredCategory, PredScores)

        # Check supported value formats for each value type
        if value_db_model not in supported_db_models:
            raise InvalidDataError(f'Element "{element.name}" cannot be filtered')
        elif value_db_model in (ExBoolean, PredBoolean):
            if value_string.strip().lower() not in ['true', 'false', '1', '0']:
                raise InvalidDataError(f'Invalid filter for element "{element.name}": "{value_string}"')
        elif value_db_model not in (ExText, PredText) and '*' in value_string:
            raise InvalidDataError('Regular expressions are supported only for texts')

        # Set value comparison
        if '*' in value_string:
            if not value_string.startswith('^'):
                value_string = '^' + value_string
            if not value_string.endswith('$'):
                value_string += '$'
            value_comp = sql_cast(value_db_model.value, String).op('regexp')(value_string)
        else:
            if value_db_model in (ExBoolean, PredBoolean):
                value_comp = (value_db_model.value == (value_string.strip().lower() in ['true', '1']))
            elif value_db_model in (ExInteger, PredInteger):
                value_comp = value_db_model.value == int(value_string)
            elif value_db_model in (ExFloat, PredFloat):
                value_comp = value_db_model.value == float(value_string)
            elif value_db_model in (ExCategory, PredCategory):
                cat_db_object = CategoryDB.get_from_id(id_value=value_string, parent=element)
                if cat_db_object is None:
                    raise ResourceNotFoundError(f'Category "{value_string}" not found')
                value_comp = value_db_model.value == cat_db_object.category_id
            elif value_db_model == PredScores:
                param = 'category'
                if idx is not None:
                    param += f'_{(idx + 1)}'
                value_comp = sql_text(f"value->>'$.category' = :{param}").bindparams(**{param: value_string.strip()})
            else:
                value_comp = value_db_model.value == value_string

        # Return filter
        return value_comp

    rsrc_db_model = ExampleDB if resource_type == Example else PredictionDB
    value_models = rsrc_db_model.value_type_models()

    # Build each element's filter(s)
    filters = []
    db_models = []

    for param, value in url_params.items():
        # Get element
        try:
            element = ElementDB.get_from_id(id_value=param, parent=task.db_object())
            value_db_model = value_models[element.value_type]
            if value_db_model == PredCategory and element.element_type == ElementType.OUTPUT:
                # In prediction logs, categorical output values are given in category-score format
                value_db_model = PredScores
        except Exception:
            raise UnprocessableRequestError(f'Element "{param}" not found')

        # Add filter(s)
        if '|' in value:
            element_filters = [
                _element_filter(element=element, value_db_model=value_db_model, value_string=value_regex, idx=idx)
                for idx, value_regex in enumerate(value.split('|'))
            ]
            filters.append(sql_or(*element_filters))
        elif ',' in value:
            element_filters = [
                _element_filter(element=element, value_db_model=value_db_model, value_string=value_regex, idx=idx)
                for idx, value_regex in enumerate(value.split(','))
            ]
            filters.append(sql_and(*element_filters))
        else:
            element_filter = _element_filter(element=element, value_db_model=value_db_model, value_string=value)
            filters.append(element_filter)
        filters.append(value_db_model.element_id == element.element_id)

        # Add database model
        if value_db_model not in db_models:
            db_models.append(value_db_model)

    return filters, db_models


DBFilters = Tuple[List[ClauseElement], List[Type[DBModel]]]
ExamplesOrPredictions = Tuple[PageSchema, List[Union[ExampleDB, PredictionDB]]]


def get_examples_or_predictions(agent: Agent,
                                task: Task,
                                resource_type: Type[Union[Example, PredictionLog]],
                                predefined_query_params: dict,
                                supported_order_by: List[str],
                                extra_filters: Optional[DBFilters] = None) -> ExamplesOrPredictions:
    """
    Retrieves examples or predictions based on various filters and query parameters.

    This function checks permissions, builds filters, applies them to the query, and retrieves
    the requested page of results. It supports ordering and handles extra filters if provided.

    Steps:
    1. Check permissions for the agent.
    2. Initialize filters and required database models.
    3. Build datetime and element value filters from query parameters.
    4. Apply filters to the query and retrieve the results.
    5. Order the results based on the specified ordering criteria.
    6. Return the paginated results and the corresponding database objects.

    Args:
        agent (Agent): The agent making the request.
        task (Task): The task associated with the examples or predictions.
        resource_type (Type[Union[Example, PredictionLog]]): The type of resource to retrieve.
        predefined_query_params (dict): Predefined query parameters for filtering and ordering.
        supported_order_by (List[str]): A list of supported fields for ordering.
        extra_filters (Optional[DBFilters]): Additional filters to apply.

    Returns:
        ExamplesOrPredictions: A tuple containing the paginated results and the corresponding database objects.
    """
    # Check permissions
    # TODO: is this check necessary?
    if isinstance(agent, UserDB):
        resource_type.check_permissions(organization=task.db_object().organization,
                                        action=ResourceAction.READ,
                                        user=agent)

    # Init filters
    filters = []

    # Set required database models for filters
    db_models = []

    # Datetime filters
    if resource_type == Example:
        datetime_fields = ['created_at', 'modified_at', 'activity_at']
    else:
        datetime_fields = ['created_at']

    for dt_field in itertools.product(datetime_fields, ['', RANGE_MIN, RANGE_MAX]):
        dt_field = ''.join(dt_field)
        if dt_field in predefined_query_params:
            try:
                dt_filter = build_datetime_filter(db_model=resource_type.db_model(),
                                                  datetime_field=dt_field,
                                                  datetime_value=predefined_query_params[dt_field])
                filters.append(dt_filter)
            except ValueError as e:
                raise InvalidDataError(f'Invalid query. Malformed value for `{dt_field}`: "{str(e)}"')

    # Element value filters
    elem_value_query_params = {k: v for k, v in request.args.items() if k not in predefined_query_params}

    elem_value_filters, elem_value_db_models = build_element_value_filters(resource_type=resource_type,
                                                                           task=task,
                                                                           url_params=elem_value_query_params)
    filters += elem_value_filters
    db_models += elem_value_db_models

    # Add extra filters (if any)
    if extra_filters:
        filters += extra_filters[0]
        db_models += extra_filters[1]

    # Apply filters
    filters = set(filters)
    db_models = set(db_models)  # TODO: debug to ensure there are no duplicates

    if filters or db_models:
        query = resource_type.db_model().query()
        for db_model in db_models:
            query = query.outerjoin(db_model)  # `outerjoin()` instead of `join()` to always keep all rows
        db_query = query.filter(sql_and(resource_type.db_model().task_id == task.db_object().task_id, *filters))
    else:
        db_query = resource_type.db_model().query().filter_by(task_id=task.db_object().task_id)

    # Get ordering
    # TODO: should we use "created_at" as the default ordering field for both examples and predictions?
    default_order_by_field = 'activity_at' if resource_type == Example else 'created_at'

    order_by_field = predefined_query_params.get('order_by', default_order_by_field).lower()
    if order_by_field not in supported_order_by:
        raise InvalidDataError(f'Invalid ordering criterion "{order_by_field}"')

    if order_by_field == 'created_at':
        # Ordering by "created_at" is equivalent to ordering by an incremental primary key field.
        # The latter is faster because the primary key is indexed.
        order_by_field = 'example_id' if resource_type == Example else 'prediction_id'

    order_by_col = getattr(resource_type.db_model(), order_by_field)

    order = predefined_query_params.get('order', 'desc').lower()
    if order not in ['asc', 'desc']:
        raise InvalidDataError('Invalid ordering')
    if order == 'desc':
        order_by_col = order_by_col.desc()

    # Remove duplicates (multi-value elements may take the same value at different indices)
    # TODO: find a more efficient way
    db_query = db_query.distinct()

    # Get specified page
    page_db_objects = get_page_db_objects(query=db_query,
                                          page_number=predefined_query_params['page'],
                                          per_page=predefined_query_params['per_page'],
                                          order_by=order_by_col,
                                          total_count=predefined_query_params['total_count'])
    db_objects = page_db_objects['data'].items

    # Build JSONs
    page = dict(page_db_objects)
    if resource_type == Example:
        page['data'] = Example.dump_batch(examples=db_objects, task=task)
    else:
        page['data'] = PredictionLog.dump_batch(predictions=db_objects, task=task)

    # Return JSONs and database objects
    return PageSchema().dump(page), db_objects
