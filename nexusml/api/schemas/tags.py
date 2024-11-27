from marshmallow import fields
from marshmallow import pre_dump
from marshmallow import pre_load
from marshmallow import validates

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.base import MutableResourceResponseSchema
from nexusml.api.schemas.base import ResourceRequestSchema
from nexusml.api.schemas.utils import add_hex_color_number_sign
from nexusml.api.schemas.utils import remove_hex_color_number_sign
from nexusml.api.schemas.utils import validate_hex_color


class TagSchema(BaseSchema):
    name = fields.String(required=True, description='Name')
    description = fields.String(description='Description', allow_none=True)
    color = fields.String(description='Color', allow_none=True)

    @validates('color')
    def _validate_hex_color(self, color):
        validate_hex_color(color)


class TagRequest(TagSchema, ResourceRequestSchema):

    @pre_load
    def _remove_hex_color_number_sign(self, data: dict, **kwargs) -> dict:
        return remove_hex_color_number_sign(data=data)


class TagResponse(TagSchema, MutableResourceResponseSchema):

    @pre_dump
    def _add_hex_color_number_sign(self, data: dict, **kwargs) -> dict:
        return add_hex_color_number_sign(data=data)
