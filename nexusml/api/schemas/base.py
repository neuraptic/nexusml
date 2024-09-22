import base64
import builtins
import copy
from datetime import datetime
import inspect
from typing import Any, Iterable, Mapping, Optional, Union
import warnings

from marshmallow import fields
from marshmallow import post_dump
from marshmallow import post_load
from marshmallow import pre_dump
from marshmallow import pre_load
from marshmallow import Schema
from marshmallow import validate
from marshmallow import ValidationError
from marshmallow.types import StrSequenceOrSet
from marshmallow_enum import EnumField

from nexusml.constants import DATETIME_FORMAT
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import UserDB

__all__ = [
    'BaseSchema', 'ResourceRequestSchema', 'ResourceResponseSchema', 'PageSchema', 'AgentSchema',
    'ImmutableResourceResponseSchema', 'MutableResourceResponseSchema', 'StateSchema', 'StatusRequestSchema',
    'StatusResponseSchema', 'BytesField'
]


class BaseSchema(Schema):

    def __bool__(self) -> bool:
        for field in self._declared_fields:  # TODO: Why don't we access `self.fields` instead?
            try:
                if getattr(self, field):
                    return True
            except AttributeError:
                continue
            except ValueError:
                if getattr(self, field) is not None:
                    return True

        return False

    @pre_load
    def _convert_lower_to_upper_in_enums(self, data, **kwargs):
        """ Converts the values of enum fields to uppercase before loading. """
        enum_fields = {field_name for field_name, field in self.fields.items() if isinstance(field, EnumField)}
        for field_name in enum_fields:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = data[field_name].upper()
        return data

    @post_dump
    def _convert_upper_to_lower_in_enums(self, data, **kwargs):
        """ Converts the values of enum fields to lowercase after dumping. """
        enum_fields = {field_name for field_name, field in self.fields.items() if isinstance(field, EnumField)}
        for field_name in enum_fields:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = data[field_name].lower()
        return data

    @post_load
    def _escape_reserved_words(self, data, **kwargs) -> dict:
        original_data = data
        data = copy.deepcopy(original_data)

        for field in original_data:
            if field not in dir(builtins):
                continue
            escaped_field = field + '_'
            data[escaped_field] = data.pop(field)

        return data

    @pre_dump
    def _unescape_reserved_words(self, data, **kwargs) -> dict:
        original_data = data
        data = copy.deepcopy(original_data)

        for field in original_data:
            if not field.endswith('_'):
                continue
            unescaped_field = field[:-1]  # remove the trailing underscore
            if unescaped_field not in dir(builtins):
                continue
            data[unescaped_field] = data.pop(field)

        return data

    # TODO: Swagger docs are showing datetime string examples with the `Z` suffix (corresponds to the UTC timezone)
    #       and users may make requests with timezone-aware datetimes (not supported). That's why we add/remove this
    #       suffix in schemas.

    @pre_load
    def _remove_utc_tz_from_datetimes(self, data, **kwargs) -> dict:
        data = copy.deepcopy(data)

        for field, value in data.items():
            if not isinstance(self.fields.get(field), fields.DateTime):
                continue
            if value and value.strip().lower().endswith('z'):
                data[field] = data[field][:-1]

        return data

    @post_dump
    def _datetimes_to_strings(self, data, **kwargs) -> dict:
        data = copy.deepcopy(data)

        for field, value in data.items():
            if not value:
                continue
            # Note: generic fields (`marshmallow.fields.Field`) may contain `datetime` objects
            if isinstance(value, datetime):
                data[field] = value.strftime(DATETIME_FORMAT) + 'Z'
            elif isinstance(self.fields.get(field), fields.DateTime):
                data[field] += 'Z'

        return data

    @post_dump
    def _remove_empty_collections(self, data, **kwargs) -> dict:
        transformed_data = copy.deepcopy(data)

        for field, value in data.items():
            if not isinstance(value, Iterable):
                continue
            # TODO: Why don't we access `self.fields` instead?
            if not (value or self._declared_fields[field].required):
                transformed_data.pop(field)

        return transformed_data

    @pre_load
    def _check_empty_strings(self, data, **kwargs):
        empty_req_str_fields = []
        for field, value in data.items():
            declared_field = self._declared_fields.get(field)  # TODO: Why don't we access `self.fields` instead?
            if isinstance(declared_field, fields.String) and declared_field.required and value == '':
                empty_req_str_fields.append(field)
        if len(empty_req_str_fields) > 0:
            raise ValidationError(f'Fields {empty_req_str_fields} cannot be empty.')
        return data


#######################################################################################################################
# Requests/Responses.                                                                                                 #

# We use separate request and response schemas because `flask-apispec` shows all the fields defined in the schemas in #
# Swagger regardless of the `load_only` and `dump_only` arguments passed to the field constructor. It only adds a     #
# `readOnly` tag to Swagger docs in dump-only fields.                                                                 #
#######################################################################################################################


class ResourceRequestSchema(BaseSchema):
    _DUMPING_ERROR_MSG = "Request schemas don't support dumping"

    def dump(self, obj: Any, *, many: Optional[bool] = None):
        raise NotImplementedError(self._DUMPING_ERROR_MSG)

    def dumps(self, obj: Any, *args, many: Optional[bool] = None, **kwargs):
        raise NotImplementedError(self._DUMPING_ERROR_MSG)

    @pre_load
    def _remove_protected_fields(self, data, **kwargs) -> dict:
        data = copy.deepcopy(data)
        if 'uuid' in data:
            data.pop('uuid')
        if 'id' in data:
            data.pop('id')
        if 'created_at' in data:
            data.pop('created_at')
        if 'created_by' in data:
            data.pop('created_by')
        if 'modified_at' in data:
            data.pop('modified_at')
        if 'modified_by' in data:
            data.pop('modified_by')
        if 'activity_at' in data:
            data.pop('activity_at')
        return data


class ResourceResponseSchema(BaseSchema):

    class Meta:
        include = {
            'uuid': fields.String(required=True, description='Universal identifier of the resource'),
            'id': fields.String(required=True, description='Identifier of the resource in the API')
        }

    @classmethod
    def _warn_loading(cls):
        stack = inspect.stack()
        caller = stack[1]
        message = (f'Use of `{cls.__name__}.{caller.function}()` is discouraged. '
                   f'Response schemas should not be used for loading data')
        warnings.warn(message, UserWarning)

    def load(
        self,
        data: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]],
        *,
        many: Optional[bool] = None,
        partial: Optional[Union[bool, StrSequenceOrSet]] = None,
        unknown: Optional[str] = None,
    ):
        """
        WARNING: Response schemas should not be used for loading data.
        """
        # We cannot raise a `NotImplementedError` as we do in `ResourceRequestSchema.dump()`
        # because `marshmallow.schema.Schema.validate()` calls `load()`.
        self._warn_loading()
        super().load(data=data, many=many, partial=partial, unknown=unknown)

    def loads(
        self,
        json_data: str,
        *,
        many: Optional[bool] = None,
        partial: Optional[Union[bool, StrSequenceOrSet]] = None,
        unknown: Optional[str] = None,
        **kwargs,
    ):
        """
        WARNING: Response schemas should not be used for loading data.
        """
        self._warn_loading()
        super().loads(json_data=json_data, many=many, partial=partial, unknown=unknown, **kwargs)

    @pre_dump
    def _remove_non_string_fields(self, data, **kwargs) -> dict:
        data = copy.deepcopy(data)

        # TODO: figure out why `data` is including an extra bytes-like key when an image (bytes) is given
        ################################################################################################################
        invalid_fields = [f for f in data if not isinstance(f, str)]
        for f in invalid_fields:
            data.pop(f)
        ################################################################################################################

        return data


class PageSchema(BaseSchema):

    class PageLinks(BaseSchema):
        previous = fields.String(required=True, allow_none=True, description='Link to previous page')
        current = fields.String(required=True, allow_none=True, description='Link to current page')
        next = fields.String(required=True, allow_none=True, description='Link to next page')

    data = fields.List(fields.Dict,
                       required=True,
                       description='Items in the requested page (or in the first page, if not specified)')
    links = fields.Nested(PageLinks, required=True, description='Links to previous, current, and next pages')
    total_count = fields.Integer(description='Total number of items in the collection '
                                 '(only present if `total_count=True` was specified)')


###############################################
# Resources created/modified by users/clients #
###############################################


class AgentSchema(ResourceResponseSchema):

    class Meta:
        include = {
            'type':
                fields.String(required=True,
                              validate=validate.OneOf(choices=['user', 'client']),
                              description='"user" or "client"')
        }


def _replace_agent_id(data: dict, field: str, **kwargs) -> dict:
    """ Replace internal user/client ID with UUID + Public ID. """
    data = copy.deepcopy(data)

    agent = None

    try:
        user = UserDB.get(user_id=int(data[field + '_user']))
        agent = {'id': user.public_id, 'uuid': user.uuid, 'type': 'user'}
    except Exception:
        pass

    try:
        client = ClientDB.get(client_id=int(data[field + '_client']))
        agent = {'id': client.public_id, 'uuid': client.uuid, 'type': 'client'}
    except Exception:
        pass

    data[field] = agent

    return data


class ImmutableResourceResponseSchema(ResourceResponseSchema):
    created_at = fields.DateTime(format=DATETIME_FORMAT, required=True, description='Created at')
    created_by = fields.Nested(AgentSchema, required=True, allow_none=True, description='Created by')

    @pre_dump
    def _replace_created_by(self, data, **kwargs) -> dict:
        return _replace_agent_id(data=data, field='created_by', **kwargs)


class MutableResourceResponseSchema(ImmutableResourceResponseSchema):
    modified_at = fields.DateTime(format=DATETIME_FORMAT, required=True, description='Last modified at')
    modified_by = fields.Nested(AgentSchema, required=True, allow_none=True, description='Last modified by')

    @pre_dump
    def _replace_modified_by(self, data, **kwargs) -> dict:
        return _replace_agent_id(data=data, field='modified_by', **kwargs)


#######################
# States and Statuses #
#######################


class StateSchema(BaseSchema):
    code = fields.String(required=True, description='State code')
    name = fields.String(required=True, description='Name')
    display_name = fields.String(required=True, description='Name to be shown')
    description = fields.String(required=True, description='Description')


class StatusRequestSchema(BaseSchema):
    code = fields.String(required=True, description='Status code')
    started_at = fields.DateTime(format=DATETIME_FORMAT, description='Start UTC datetime')
    updated_at = fields.DateTime(format=DATETIME_FORMAT, allow_none=True, description='Last update UTC datetime')
    ended_at = fields.DateTime(format=DATETIME_FORMAT, allow_none=True, description='End UTC datetime')
    details = fields.Dict(allow_none=True, description='Status-specific details')


class StatusResponseSchema(BaseSchema):
    state_code = fields.String(required=True, description='State code')
    status_code = fields.String(required=True, description='Status code')
    name = fields.String(required=True, description='Status name')
    display_name = fields.String(required=True, description='Name to be shown')
    description = fields.String(required=True, description='Status description')
    started_at = fields.DateTime(format=DATETIME_FORMAT, required=True, description='Start UTC datetime')
    updated_at = fields.DateTime(format=DATETIME_FORMAT,
                                 required=True,
                                 allow_none=True,
                                 description='Last update UTC datetime')
    ended_at = fields.DateTime(format=DATETIME_FORMAT, required=True, allow_none=True, description='End UTC datetime')
    details = fields.Dict(required=True, allow_none=True, description='Status-specific details')
    prev_status = fields.String(required=True, allow_none=True, description='Code of the previous status')

    @pre_dump
    def _parse_datetimes(self, data, **kwargs) -> dict:
        data = copy.deepcopy(data)
        if isinstance(data['started_at'], str):
            data['started_at'] = datetime.strptime(data['started_at'], DATETIME_FORMAT)
        if isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.strptime(data['updated_at'], DATETIME_FORMAT)
        if isinstance(data['ended_at'], str):
            data['ended_at'] = datetime.strptime(data['ended_at'], DATETIME_FORMAT)
        return data


#######################
# Special field types #
#######################


class BytesField(fields.String):
    """
    Marshmallow field that serializes bytes to a base64-encoded string and deserializes a base64-encoded string to bytes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if value is not None:
            return base64.b64encode(value).decode('ascii')

    def _deserialize(self, value, attr, data, **kwargs):
        if value is not None:
            return base64.b64decode(value)
