from typing import Dict, Iterable, List, Optional, Type, Union

from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import QuotaError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.files import TaskFile
from nexusml.api.utils import config
from nexusml.database.ai import PredCategory
from nexusml.database.ai import PredFile
from nexusml.database.ai import PredictionDB
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExCategory
from nexusml.database.examples import ExFile
from nexusml.database.examples import ExShape
from nexusml.database.examples import ExSlice
from nexusml.database.examples import ShapeDB
from nexusml.database.examples import SliceDB
from nexusml.database.files import TaskFileDB as FileDB
from nexusml.database.organizations import UserDB
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.database.tasks import TaskER
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import TaskFileUse
from nexusml.utils import FILE_TYPES


def check_quota_usage(name: str, usage: Union[int, float], limit: Union[int, float], description: str = None):
    """
    Checks whether quota usage exceeds the defined limit. Raises a `QuotaError` if the usage exceeds the limit.

    For space quotas, the limit is converted to GB or MB depending on the size, and a descriptive error message
    is raised.
    For other quotas, the default description or quota name is used in the error message.

    Args:
        name (str): The name of the quota (e.g., 'space').
        usage (Union[int, float]): The current usage of the quota.
        limit (Union[int, float]): The allowed limit for the quota.
        description (str, optional): A description of the quota. If not provided, a default description is used.

    Raises:
        QuotaError: If the quota usage exceeds the defined limit.
    """
    if usage <= limit:
        return
    if name == 'space':
        description = description or 'Space quota limit'
        quota_gb = limit / (1024**3)
        quota_to_show = quota_gb if quota_gb >= 1 else quota_gb * 1024
        unit_to_show = 'GB' if quota_gb >= 1 else 'MB'
        raise QuotaError(f'{description} ({quota_to_show:.2f} {unit_to_show}) exceeded')
    else:
        description = description or name
        raise QuotaError(f'{description} ({limit}) exceeded')


def preload_task_db_objects(task: TaskDB,
                            db_model: Type[TaskER],
                            limit: Optional[int] = None) -> Dict[str, Dict[str, TaskER]]:
    """
    Preloads database objects associated with a given task, organizing them by UUID, public ID, and name.

    The function fetches all objects related to the task from the specified `db_model` and organizes them into
    dictionaries based on UUID, public ID, and name. The total number of objects preloaded can be limited by
    the `limit` argument.

    Steps:
        1. Query the database for objects related to the task.
        2. If a limit is set and the number of objects exceeds the limit, raise a RuntimeError.
        3. Organize the objects into dictionaries keyed by UUID, public ID, and name.

    Args:
        task (TaskDB): The task whose related objects are to be preloaded.
        db_model (Type[TaskER]): The database model class for the objects.
        limit (Optional[int]): The maximum number of objects to preload. If exceeded, raises an error.

    Returns:
        dict: A dictionary containing preloaded database objects, organized by UUID, public ID, and name.

    Raises:
        RuntimeError: If the number of objects exceeds the specified limit.

    Dictionary structure:
    {
        "uuid": {
            "<uuid>": <db_object>,
            "<uuid>": <db_object>,
            ...
        },
        "public_id": {
            "<public_id>": <db_object>,
            "<public_id>": <db_object>,
            ...
        },
        "name": {
            "<name>": <db_object>,
            "<name>": <db_object>,
            ...
        },
    }
    """
    db_objects_query = db_model.query().filter(db_model.task_id == task.task_id)

    if limit is not None and db_objects_query.count() > limit:
        raise RuntimeError(f'Maximum number of objects to preload exceeded: {limit}')

    db_objects_by_uuids = dict()
    db_objects_by_pub_ids = dict()
    db_objects_by_names = dict()

    for db_object in db_objects_query.all():
        db_objects_by_uuids[db_object.uuid] = db_object
        db_objects_by_pub_ids[db_object.public_id] = db_object
        db_objects_by_names[db_object.name] = db_object

    return {'uuid': db_objects_by_uuids, 'public_id': db_objects_by_pub_ids, 'name': db_objects_by_names}


def preload_task_categories(task: TaskDB) -> Optional[Dict[str, Dict[str, CategoryDB]]]:
    """
    Preloads task categories associated with the given task and organizes them by UUID, public ID, and name.

    This function retrieves categories for the provided task and organizes them into dictionaries based on
    UUID, public ID, and name. If the number of categories exceeds the configured maximum, it returns `None`.

    Steps:
        1. Query the database for categories associated with the task.
        2. Check if the number of categories exceeds the configured limit; if so, return `None`.
        3. Organize the categories into dictionaries keyed by UUID, public ID, and name.

    Args:
        task (TaskDB): The task whose categories are to be preloaded.

    Returns: Optional[Dict[str, Dict[str, CategoryDB]]]: A dictionary of preloaded categories or `None` if the limit
    is exceeded.

    Dictionary structure:
    {
        "uuid": {
            "<uuid>": <category>,
            "<uuid>": <category>,
            ...
        },
        "public_id": {
            "<public_id>": <category>,
            "<public_id>": <category>,
            ...
        },
        "name": {
            "<name>": <category>,
            "<name>": <category>,
            ...
        },
    }
    """

    categories_query = (CategoryDB.query().join(
        ElementDB, ElementDB.element_id == CategoryDB.element_id).filter(ElementDB.task_id == task.task_id))

    if categories_query.count() > config.get('limits')['tasks']['max_preloaded_categories']:
        return

    cats_by_uuids = dict()
    cats_by_pub_ids = dict()
    cats_by_names = dict()

    for category in categories_query.all():
        cats_by_uuids[category.uuid] = category
        cats_by_pub_ids[category.public_id] = category
        cats_by_names[category.name] = category

    return {'uuid': cats_by_uuids, 'public_id': cats_by_pub_ids, 'name': cats_by_names}


def get_preloaded_db_object(
        id_: str, preloaded_db_objects: Dict[str, Dict[str, Union[TaskER, CategoryDB]]]) -> Union[TaskER, CategoryDB]:
    """
    Retrieves a preloaded database object by its UUID, public ID, or name.

    This function searches the preloaded database objects using the provided `id_`. It looks for the object
    in the dictionaries of UUIDs, public IDs, and names.

    Note: `preloaded_db_objects` must follow the same format as that followed by the
          values returned by `preload_task_db_objects` and `preload_task_categories`.

    Args:
        id_ (str): The identifier to search for (UUID, public ID, or name).
        preloaded_db_objects (Dict[str, Dict[str, Union[TaskER, CategoryDB]]]): Preloaded objects, grouped by
                                                                                identifier type.

    Returns:
        Union[TaskER, CategoryDB]: The preloaded database object.

    Raises:
        ResourceNotFoundError: If the object cannot be found in the preloaded database objects.
    """
    db_objs_by_uuid = preloaded_db_objects['uuid']
    db_objs_by_pub_id = preloaded_db_objects['public_id']
    db_objs_by_name = preloaded_db_objects['name']

    if id_ in db_objs_by_uuid:
        return db_objs_by_uuid[id_]
    elif id_ in db_objs_by_pub_id and db_objs_by_pub_id[id_].public_id == id_:
        return db_objs_by_pub_id[id_]
    elif id_ in db_objs_by_name:
        return db_objs_by_name[id_]
    else:
        raise ResourceNotFoundError()


def validate_element_values(data: dict,
                            collection: str,
                            preloaded_elements: Dict[str, Dict[str, ElementDB]],
                            allowed_value_types: Optional[Iterable[ElementValueType]] = None,
                            check_required: bool = True,
                            excluded_required: Optional[List[ElementType]] = None) -> None:
    """
    Validates the values assigned to elements in a collection, ensuring they meet required criteria.

    This function checks if the values assigned to elements in a given collection adhere to the allowed
    value types, if required elements are provided, and whether null and multi-value conditions are respected.

    Steps:
        1. Validate the existence and type of provided elements.
        2. Ensure that required elements are present in the data, unless excluded.
        3. Verify that nullable and multi-value conditions are met for each element.
        4. Raise an error with details if any validation fails.

    Args:
        data (dict): The data containing element values.
        collection (str): The collection name that contains the element values.
        preloaded_elements (Dict[str, Dict[str, ElementDB]]): Preloaded elements.
        allowed_value_types (Optional[Iterable[ElementValueType]]): List of allowed value types for elements.
        check_required (bool): Whether to check if all required elements are provided in the data.
        excluded_required (Optional[List[ElementType]]): Element types to exclude from the required check.

    Raises:
        InvalidDataError: If any element validation fails (invalid elements, missing required elements, etc.).
    """

    def _validate_elements() -> Dict[str, List[str]]:
        """
        Returns the names of the elements with errors. Four types (keys):
            - "invalid": non-existent elements or elements with not allowed value types.
            - "required": missing required elements.
            - "nullable": non-nullable elements with null values.
            - "multi_value": non-multi-value elements with multiple values.
        """
        errors = {'invalid': [], 'required': [], 'nullable': [], 'multi_value': []}

        # Get provided element values
        provided_element_values = data.get(collection, []) or []  # Avoid `None`
        provided_elements = [x['element'] for x in provided_element_values]

        # Get provided element database objects
        provided_element_db_objs = []
        for elem_id in provided_elements:
            try:
                elem_db_obj = get_preloaded_db_object(id_=elem_id, preloaded_db_objects=preloaded_elements)
                provided_element_db_objs.append(elem_db_obj)
            except ResourceNotFoundError:
                errors['invalid'].append(elem_id)

        # Check if provided elements have allowed value types
        if allowed_value_types:
            for elem_db_obj in provided_element_db_objs:
                if elem_db_obj.value_type not in allowed_value_types:
                    errors['invalid'].append(elem_db_obj.name)

        # Check if all required elements are present in provided data
        elements = preloaded_elements['uuid'].values()
        if check_required:
            for x in elements:
                if not x.required or x.element_type in excluded_required:
                    continue
                if not (x.name in provided_elements or x.public_id in provided_elements or x.uuid in provided_elements):
                    errors['required'].append(x.name)

        # Check if nulls and/or multiple values are supported for provided elements
        elements_with_values = []
        for element_value in provided_element_values:
            try:
                element = get_preloaded_db_object(id_=element_value['element'], preloaded_db_objects=preloaded_elements)
            except ResourceNotFoundError:
                assert element_value['element'] in errors['invalid']
                continue
            # Check nullable
            if element_value['value'] is None and not element.nullable:
                errors['nullable'].append(element.name)
            # Check multi-value
            multi_value = isinstance(element_value['value'], list) or element.name in elements_with_values
            if multi_value and element.multi_value is None:
                errors['multi_value'].append(element.name)
            elements_with_values.append(element.name)

        return errors

    excluded_required = excluded_required if excluded_required is not None else []

    value_errors = _validate_elements()

    invalid_elements = value_errors['invalid']
    missing_elements = value_errors['required']
    wrong_nulls = value_errors['nullable']
    wrong_multi_values = value_errors['multi_value']

    error_msgs = []

    if invalid_elements:
        elems = ', '.join(f'{x}' for x in invalid_elements)
        error_msg = f'Invalid elements: {elems}'
        error_msgs.append(error_msg)
    if missing_elements:
        elems = ', '.join(f'{x}' for x in missing_elements)
        error_msg = f'Missing required elements: {elems}'
        error_msgs.append(error_msg)
    if wrong_nulls:
        elems = ', '.join(f'{x}' for x in wrong_nulls)
        error_msg = f'Elements not supporting null values: {elems}'
        error_msgs.append(error_msg)
    if wrong_multi_values:
        elems = ', '.join(f'{x}' for x in wrong_multi_values)
        error_msg = f'Elements not supporting multiple values: {elems}'
        error_msgs.append(error_msg)

    if error_msgs:
        raise InvalidDataError('\n'.join(error_msgs))


def split_element_values_into_db_collections(data: dict, collection: str, db_model: Type[Union[ExampleDB, PredictionDB,
                                                                                               ShapeDB, SliceDB]],
                                             task: TaskDB, preloaded_elements: Dict[str, Dict[str, ElementDB]],
                                             preloaded_categories: Dict[str, Dict[str, CategoryDB]]):
    """
    Splits element values into separate database collections based on their types.

    This function takes element values provided in a collection and assigns them to the correct database collections.
    It converts IDs for files, categories, shapes, and slices into database primary keys, while also verifying the
    element's value type and structure (e.g., multi-value support).

    Steps:
        1. Map JSON collections to database collections.
        2. Separate the values of multi-value elements and assign each value a unique index.
        3. Assign values to the appropriate database collections based on element types.

    Args:
        data (dict): The data containing element values.
        collection (str): The name of the collection in the JSON data.
        db_model (Type[Union[ExampleDB, PredictionDB, ShapeDB, SliceDB]]): The database model class.
        task (TaskDB): The task to which the elements are related.
        preloaded_elements (Dict[str, Dict[str, ElementDB]]): Preloaded elements.
        preloaded_categories (Dict[str, Dict[str, CategoryDB]]): Preloaded categories.

    Raises:
        InvalidDataError: If any issues are found with the element values.
    """
    errors = []

    # Get collection data
    if collection not in data:
        return

    if isinstance(data[collection], list):
        element_values = data[collection]
    elif data[collection] is None:
        element_values = []
    else:
        raise InvalidDataError(f'Invalid collection "{collection}": {data[collection]}')

    # Mapping from JSON collection to database collection for assigned values
    value_collections = db_model.value_collections()
    value_type_collections = db_model.value_type_collections()
    assert all(x in value_collections for x in value_type_collections.values())

    # Separate the values of multi-value elements and register the index of each value
    provided_elements = []

    for element_value in list(element_values):  # Loop over a copy of `element_values` to modify the original collection
        # Get element
        try:
            element = get_preloaded_db_object(id_=element_value['element'], preloaded_db_objects=preloaded_elements)
        except ResourceNotFoundError:
            errors.append(f'Element "{element_value["element"]}" not found')
            continue

        # Verify the element has not been provided yet
        if element in provided_elements:
            errors.append(f'Duplicate element "{element_value["element"]}". For multi-value elements,'
                          f'provide a list of values in "value" field')
            continue
        provided_elements.append(element)

        # Skip non-multi-value elements
        if not isinstance(element_value['value'], (list, set)):
            continue
        if not element.multi_value:
            errors.append(f'Multiple values provided for single-value element "{element_value["element"]}"')

        # Create a new element-value item for each value provided for the element
        element_values.remove(element_value)
        for idx, value in enumerate(element_value['value']):
            new_element_value = dict(element_value)
            new_element_value['value'] = value
            new_element_value['index'] = idx + 1
            element_values.append(new_element_value)

    # Split element values into database collections
    for x in value_collections:
        if x in data:
            assert isinstance(data[x], list)
        else:
            data[x] = []

    for element_value in element_values:
        # Get element
        try:
            element = get_preloaded_db_object(id_=element_value['element'], preloaded_db_objects=preloaded_elements)
        except ResourceNotFoundError:
            errors.append(f'Element "{element_value["element"]}" not found')
            continue

        # Check if we are handling prediction scores (JSON containing category-score pairs)
        is_pred_score = (db_model == PredictionDB and element.element_type == ElementType.OUTPUT and
                         element.value_type == ElementValueType.CATEGORY)

        # Convert files, categories, shapes, and slices' IDs into database primary keys
        if element_value['value']:
            # Files
            if element.value_type in FILE_TYPES:
                file = FileDB.get_from_id(id_value=element_value['value'], parent=task)
                if file is None:
                    errors.append(f'File "{element_value["value"]}" not found')
                    continue
                file_usages = {
                    ElementType.INPUT: TaskFileUse.INPUT,
                    ElementType.OUTPUT: TaskFileUse.OUTPUT,
                    ElementType.METADATA: TaskFileUse.METADATA
                }
                if file.use_for != file_usages[element.element_type]:
                    errors.append(f'Invalid use for file "{element_value["value"]}"')
                    continue
                element_value['value'] = file.file_id
            # Categories
            elif element.value_type == ElementValueType.CATEGORY and not is_pred_score:
                try:
                    if preloaded_categories:
                        category = get_preloaded_db_object(id_=element_value['value'],
                                                           preloaded_db_objects=preloaded_categories)

                    else:
                        category = CategoryDB.get_from_id(id_value=element_value['value'], parent=element)
                        if category is None:
                            raise ResourceNotFoundError()
                except ResourceNotFoundError:
                    errors.append(f'Category "{element_value["value"]}" not found')
                    continue
                element_value['value'] = category.category_id
            # Shapes and slices
            # Note: in predictions, shapes and slices are given in JSON format.
            elif element.value_type in [ElementValueType.SHAPE, ElementValueType.SLICE] and db_model == ExampleDB:
                if element.value_type == ElementValueType.SHAPE:
                    elem_value_db_model = ShapeDB
                    elem_value_db_model_pk = 'shape_id'
                else:
                    elem_value_db_model = SliceDB
                    elem_value_db_model_pk = 'slice_id'

                elem_value_db_obj = elem_value_db_model.get_from_id(id_value=element_value['value'])
                if elem_value_db_obj is None:
                    errors.append(f'{element.value_type.name.capitalize()} "{element_value["value"]}" not found')
                    continue

                element_value['value'] = getattr(elem_value_db_obj, elem_value_db_model_pk)

        # Add the element value assignment to the corresponding database collection
        elem_value_collection = value_type_collections[element.value_type] if not is_pred_score else 'pred_scores'
        data[elem_value_collection].append(element_value)

    if errors:
        raise InvalidDataError(f'Invalid element values. Errors found: {errors}')

    data.pop(collection)


def merge_element_values_from_db_collections(data: dict,
                                             db_model: Type[Union[ExampleDB, PredictionDB, ShapeDB, SliceDB]],
                                             preloaded_elements: Dict[str, Dict[str, ElementDB]] = None):
    """
    Merges element values from various database collections into a single structured format.

    This function processes the element values from the database, converting file, category, shape,
    and slice primary keys into public identifiers. It also ensures that multi-value elements are
    returned as lists, even if there is only one value in the list.

    Steps:
        1. Convert the database primary keys (files, categories, shapes, slices) to their public IDs.
        2. Collect the element values from different collections and merge them into the `data`.
        3. Ensure that multi-value elements are returned as lists, preserving the order of values.

    Args:
        data (dict): A dictionary containing the element data.
        db_model (Type[Union[ExampleDB, PredictionDB, ShapeDB, SliceDB]]): The database model for the objects.
        preloaded_elements (Dict[str, Dict[str, ElementDB]], optional): Preloaded elements by UUID, public ID, and name.
    """
    db_collections = db_model.value_collections()
    elem_values_by_element = dict()
    elem_value_list = []

    # Convert files, categories, shapes, and slices' database primary keys to public identifiers
    for db_collection in db_collections:
        # Files
        if db_collection.endswith('_files'):
            for elem_value in data.get(db_collection, []):
                f = elem_value.pop('file', None)
                elem_value['value'] = f.public_id if isinstance(f, FileDB) else f
        # Categories
        elif db_collection.endswith('_categories'):
            for elem_value in data.get(db_collection, []):
                cat = elem_value.pop('category', None)
                elem_value['value'] = cat.name if isinstance(cat, CategoryDB) else cat
        # Shapes and slices
        # Note: in predictions, shapes and slices are given in JSON format.
        elif db_collection.endswith(('_shapes', '_slices')) and db_model == ExampleDB:
            if db_collection.endswith('_shapes'):
                db_rel_name = 'shape'
                elem_value_db_model = ShapeDB
            else:
                db_rel_name = 'slice'
                elem_value_db_model = SliceDB

            for elem_value in data.get(db_collection, []):
                elem_value_db_obj = elem_value.pop(db_rel_name, None)
                if isinstance(elem_value_db_obj, elem_value_db_model):
                    elem_value['value'] = elem_value_db_obj.public_id
                else:
                    elem_value['value'] = elem_value_db_obj

    # Get the value(s) for each element and remove database collection from data
    for db_collection in [c for c in db_collections if c in data]:
        elem_value_list += data.pop(db_collection)
    for element_value in elem_value_list:
        element_key = (element_value['element'], element_value.get('is_target', False))
        if element_key not in elem_values_by_element:
            elem_values_by_element[element_key] = []
        elem_values_by_element[element_key].append(element_value)

    # Save the value(s).
    # Note: the order of values assigned to multi-value elements is preserved (based on the `index` field).
    # Note: values assigned to multi-value elements are always returned in list format, even in the case of
    #       single-value lists.
    dst_fields = {ElementType.INPUT: 'inputs', ElementType.OUTPUT: 'outputs', ElementType.METADATA: 'metadata'}
    for field in dst_fields.values():
        data[field] = list()

    if db_model == PredictionDB:
        data['targets'] = []

    for (element_id, is_target), element_values in elem_values_by_element.items():
        # Get element
        if preloaded_elements:
            try:
                element = get_preloaded_db_object(id_=element_id, preloaded_db_objects=preloaded_elements)
            except ResourceNotFoundError:
                raise Exception  # this should never happen
        else:
            element = ElementDB.get(element_id=element_values[0]['element_id'])

        # Target values should be used only for computing metrics on predicted output values
        if is_target:
            assert db_model == PredictionDB and element.element_type == ElementType.OUTPUT

        # Put value(s) in the right format (single-value or list)
        if len(element_values) > 1:
            element_value = [x['value'] for x in sorted(element_values, key=lambda x: x['index'])]
        else:
            if element and element.multi_value is not None:
                element_value = [element_values[0]['value']]
            else:
                element_value = element_values[0]['value']

        # Add element value
        dst_field = dst_fields[element.element_type] if not is_target else 'targets'
        data[dst_field].append({'element': element_id, 'value': element_value})


def dump_element_values(db_object: Union[ExampleDB, PredictionDB], element_names: Dict[int, str]) -> dict:
    """
    Dumps the element values associated with the provided example or prediction database object.

    This function collects all element values from an example or prediction and converts database
    relationships (e.g., category, file, shape, slice) into a dictionary format. The names of the elements
    are provided in a mapping from the element's surrogate key.

    Steps:
        1. Retrieve the element values from the example or prediction.
        2. Convert the related objects (categories, files, shapes, slices) to their dictionary representations.
        3. Group the values by the database collections (inputs, outputs, metadata, etc.).
        4. Return the formatted element values.

    Args:
        db_object (Union[ExampleDB, PredictionDB]): The example or prediction database object.
        element_names (dict): A dictionary where the key is the element's surrogate key,
        and the value is the element's name.

    Returns:
        dict: A dictionary containing the dumped element values, grouped by collection.
    """
    element_values = {x: list() for x in db_object.value_collections()}

    cat_db_model = ExCategory if isinstance(db_object, ExampleDB) else PredCategory
    file_db_model = ExFile if isinstance(db_object, ExampleDB) else PredFile

    for collection, values in element_values.items():
        for elem_value_db_object in getattr(db_object, collection):
            elem_value = elem_value_db_object.to_dict()
            elem_value['element'] = element_names[elem_value_db_object.element_id]
            if isinstance(elem_value_db_object, cat_db_model):
                elem_value['category'] = elem_value_db_object.category
            elif isinstance(elem_value_db_object, file_db_model):
                elem_value['file'] = elem_value_db_object.file
            elif isinstance(elem_value_db_object, ExShape):
                elem_value['shape'] = elem_value_db_object.shape
            elif isinstance(elem_value_db_object, ExSlice):
                elem_value['slice'] = elem_value_db_object.slice
            values.append(elem_value)

    return element_values


def delete_example_or_prediction(example_or_prediction: Resource, notify_to: Iterable[UserDB] = None):
    """
    Deletes the provided example or prediction, including its associated files, and sends notifications if required.

    This function deletes an example or prediction resource and also removes any associated files from the database.
    Files that are not associated with other examples or predictions are deleted from the storage system, ensuring
    that storage quotas are updated accordingly.

    Steps:
        1. Retrieve the IDs of files associated with the example or prediction.
        2. Delete the example or prediction using the superclass method.
        3. Check if any associated files are linked to other examples or predictions; if not, delete the file.
        4. Send notifications as required.

    Args:
        example_or_prediction (Resource): The resource representing the example or prediction to be deleted.
        notify_to (Iterable[UserDB], optional): Users to be notified about the deletion.

    Raises:
        ResourceNotFoundError: If the file or example cannot be found.
    """
    resource = example_or_prediction

    # Get associated files' IDs to delete them later
    files_db_collection = 'ex_files' if example_or_prediction.db_model() == ExampleDB else 'pred_files'
    file_db_objects = [x.file for x in getattr(resource.db_object(), files_db_collection)]

    # Delete example or prediction
    super(resource.__class__, resource).delete(notify_to=notify_to)

    # Delete associated files (only if they're not associated with more examples or predictions)
    # TODO: Any way to avoid importing `TaskFile`? This may lead to circular imports
    #       WARNING: we need `TaskFile.delete()` logic because it removes the file from S3 and updates task quota usage
    for file_db_obj in file_db_objects:
        num_other_assoc = ExFile.query().filter_by(value=file_db_obj.file_id).count()
        num_other_assoc += PredFile.query().filter_by(value=file_db_obj.file_id).count()
        if num_other_assoc == 0:
            file = TaskFile.get(agent=resource.agent(),
                                db_object_or_id=file_db_obj,
                                parents=resource.parents(),
                                check_permissions=False,
                                check_parents=False)
            file.delete(notify_to=notify_to)
