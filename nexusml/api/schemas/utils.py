import copy
from datetime import datetime
import re
import string

from marshmallow import fields
from marshmallow import post_dump
from marshmallow import ValidationError

from nexusml.api.schemas.base import BaseSchema
from nexusml.constants import DATETIME_FORMAT


class ElementValue(BaseSchema):
    element = fields.String(required=True, description='Name of the element to which the value refers')
    value = fields.Field(required=True,
                         allow_none=True,
                         description='Assigned value. For files and shapes, provide the ID')

    @post_dump
    def _datetimes_to_strings(self, data, **kwargs) -> dict:
        """
        We override this method because `BaseSchema` adds the UTC timezone suffix ("Z") to datetime strings,
        and we cannot make any assumption about the actual timezone of the element value.
        """
        dt = data['value']
        if not (dt and isinstance(dt, datetime)):
            return data
        data = copy.deepcopy(data)
        data['value'] = dt.strftime(DATETIME_FORMAT)
        return data


def add_hex_color_number_sign(data: dict) -> dict:
    if 'color' not in data:
        return data
    data = copy.deepcopy(data)
    if data['color'] and not data['color'].startswith('#'):
        data['color'] = '#' + data['color']
    return data


def remove_hex_color_number_sign(data: dict) -> dict:
    if 'color' not in data:
        return data
    data = copy.deepcopy(data)
    if data['color'] and data['color'].startswith('#'):
        data['color'] = data['color'][1:]
    return data


def validate_hex_color(hex_color: str):
    if not hex_color:
        return
    if not re.search('^#?[0-9a-fA-F]{6}$', hex_color):
        raise ValidationError('Invalid hexadecimal color code')


def check_punctuation_chars(data: dict, fields: list) -> dict:
    punctuation_chars = string.punctuation.replace('_', '')
    for field in fields:
        if field in data and isinstance(data[field], str) and any(x in punctuation_chars for x in data[field]):
            raise ValidationError(f"'{field}' cannot contain punctuation characters: {punctuation_chars}")
    return data
