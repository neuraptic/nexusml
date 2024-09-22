from marshmallow import fields
from marshmallow_enum import EnumField

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.base import ImmutableResourceResponseSchema
from nexusml.api.schemas.base import MutableResourceResponseSchema
from nexusml.api.schemas.base import PageSchema
from nexusml.api.schemas.base import ResourceRequestSchema
from nexusml.api.schemas.utils import ElementValue
from nexusml.constants import DATETIME_FORMAT
from nexusml.enums import LabelingStatus


class ShapeSchema(BaseSchema):

    class Coordinates(BaseSchema):
        x = fields.Integer(required=True, description='X coordinate')
        y = fields.Integer(required=True, description='Y coordinate')

    class ShapeOutput(ElementValue):
        element = fields.String(required=True, description='Name of the element to which the value refers')
        value = fields.Field(required=True, description='Assigned value')

    element = fields.String(required=True,
                            description='Name of the input image/video-file-type element in which the shape is drawn')
    polygon = fields.List(fields.Nested(Coordinates),
                          allow_none=True,
                          description='JSON containing a list of vertices. Mutually exclusive with '
                          '`path` and `pixels`. Useful for geometric shapes described by a finite number '
                          'of connected straight line segments')
    path = fields.String(allow_none=True,
                         description='String defining an SVG <path> element. Mutually exclusive with '
                         '`polygon` and `pixels`. Useful for free shapes')
    pixels = fields.List(fields.Nested(Coordinates),
                         allow_none=True,
                         description='JSON containing a list of pixels. Mutually exclusive with `polygon` and `path`. '
                         'Useful for segmentation. WARNING: pixels must be connected with each other '
                         'to form a Shape')
    outputs = fields.List(fields.Nested(ShapeOutput), allow_none=True, description='Value for each output element')


class ShapeRequest(ShapeSchema, ResourceRequestSchema):
    pass


class ShapeResponse(ShapeSchema, MutableResourceResponseSchema):
    pass


class SliceSchema(BaseSchema):

    class SliceOutput(ElementValue):
        element = fields.String(required=True, description='Name of the element to which the value refers')
        value = fields.Field(required=True, description='Assigned value')

    element = fields.String(required=True, description='Name of the input element to which the slice refers')
    start_index = fields.Integer(required=True, description='Start index (1-based)')
    end_index = fields.Integer(required=True, description='End index (1-based)')
    outputs = fields.List(fields.Nested(SliceOutput), allow_none=True, description='Value for each output element')


class SliceRequest(SliceSchema, ResourceRequestSchema):
    pass


class SliceResponse(SliceSchema, MutableResourceResponseSchema):
    pass


class ExampleSchema(BaseSchema):
    labeling_status = EnumField(LabelingStatus,
                                description='Labeling status: "unlabeled" | "pending_review" | "labeled" | "rejected"')
    tags = fields.List(fields.String, allow_none=True, description='List of tag IDs')


class ExampleRequest(ExampleSchema, ResourceRequestSchema):
    values = fields.List(fields.Nested(ElementValue), required=True, description='Values assigned to elements')


class ExampleResponse(ExampleSchema, MutableResourceResponseSchema):
    inputs = fields.List(fields.Nested(ElementValue), required=True, description='Values assigned to input elements')
    outputs = fields.List(fields.Nested(ElementValue), required=True, description='Values assigned to output elements')
    metadata = fields.List(fields.Nested(ElementValue),
                           required=True,
                           description='Values assigned to metadata elements')
    activity_at = fields.DateTime(format=DATETIME_FORMAT,
                                  required=True,
                                  description='Last datetime at which some activity took place in the example, '
                                  'including comments')
    shapes = fields.List(fields.Nested(ShapeResponse),
                         allow_none=True,
                         description='Shapes drawn in input image/video files. Ignore it if not facing a visual task')

    slices = fields.List(fields.Nested(SliceResponse),
                         allow_none=True,
                         description='Ranges of values within sequences or time series')


class ExamplesPage(PageSchema):
    data = fields.List(fields.Nested(ExampleResponse),
                       required=True,
                       description='Examples in the requested page (or in the first page, if not specified)')


class ExampleBatchRequest(BaseSchema):
    batch = fields.List(fields.Nested(ExampleRequest), required=True)


class ExampleBatchResponse(BaseSchema):
    batch = fields.List(fields.Nested(ExampleResponse), required=True)


class CommentSchema(BaseSchema):
    message = fields.String(required=True, description='Comment text')


class CommentRequest(CommentSchema, ResourceRequestSchema):
    pass


class CommentResponse(CommentSchema, ImmutableResourceResponseSchema):
    pass
