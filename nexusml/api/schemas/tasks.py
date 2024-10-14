import copy

from marshmallow import fields
from marshmallow import post_load
from marshmallow import pre_dump
from marshmallow import pre_load
from marshmallow import validate
from marshmallow import validates
from marshmallow_enum import EnumField

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.base import MutableResourceResponseSchema
from nexusml.api.schemas.base import ResourceRequestSchema
from nexusml.api.schemas.base import StatusResponseSchema
from nexusml.api.schemas.files import TaskFileResponse
from nexusml.api.schemas.utils import add_hex_color_number_sign
from nexusml.api.schemas.utils import check_punctuation_chars
from nexusml.api.schemas.utils import remove_hex_color_number_sign
from nexusml.api.schemas.utils import validate_hex_color
from nexusml.constants import API_NAME
from nexusml.enums import ElementMultiValue
from nexusml.enums import ElementValueType
from nexusml.enums import TaskTemplate
from nexusml.enums import TaskType

_task_type_field = EnumField(TaskType,
                             allow_none=True,
                             description='Type of the task: ' + ' | '.join(f'"{x.name.lower()}"' for x in TaskType))


class TaskSchema(BaseSchema):

    class Meta:
        include = {'type': _task_type_field}

    name = fields.String(required=True, description='Name of the task')
    description = fields.String(allow_none=True, description='Description of the task')


class TaskRequest(TaskSchema, ResourceRequestSchema):
    icon = fields.String(allow_none=True,
                         description=f'File containing the task icon. '
                         f'WARNING: the file must be previously uploaded to {API_NAME}')


class TaskPOSTRequest(TaskRequest):
    template = EnumField(TaskTemplate,
                         allow_none=True,
                         description=' | '.join(f'"{x.name.lower()}"' for x in TaskTemplate))


class TaskPUTRequest(TaskRequest):
    pass


class TaskResponse(TaskSchema, MutableResourceResponseSchema):
    icon = fields.Nested(TaskFileResponse, allow_none=True, description='File containing the task icon')
    status = fields.Nested(StatusResponseSchema, required=True, description='Task status')
    organization = fields.String(allow_none=True, description='Organization that administers the task')
    num_deployments = fields.Integer(required=True, description='Current number of deployments')
    max_deployments = fields.Integer(required=True, description='Maximum number of deployments')
    space_usage = fields.Integer(description='Space quota usage (in bytes)')
    space_limit = fields.Integer(required=True, description='Space quota limit (in bytes)')


class ElementSchema(BaseSchema):

    class Meta:
        _value_type_names = ' | '.join(f'"{x.name.lower()}"' for x in ElementValueType)

        include = {
            'type':
                EnumField(ElementValueType,
                          required=True,
                          description='Type of the values assigned to the element: ' + _value_type_names)
        }

    name = fields.String(required=True,
                         description='Name of the element (punctuation characters will be replaced with "_")')
    display_name = fields.String(description='Name to be shown', allow_none=True)
    description = fields.String(description='Description of the element', allow_none=True)
    required = fields.Boolean(description='All examples must have a value for the element')
    nullable = fields.Boolean(description='Allow null values')
    multi_value = EnumField(ElementMultiValue,
                            allow_none=True,
                            description='Allow multiple values. Three formats supported: '
                            '"unordered" | "ordered" | "time_series" (same as "ordered", but in this '
                            'case the index of a value represents a time step). If no format is '
                            'provided, the element will not allow multiple values.')

    @pre_load
    def _check_punctuation_chars(self, data: dict, **kwargs):
        return check_punctuation_chars(data=data, fields=['name'])


class ElementRequest(ElementSchema, ResourceRequestSchema):

    @post_load
    def _rename_value_type_field(self, data: dict, **kwargs):
        data = copy.deepcopy(data)
        data['value_type'] = data.pop('type_')
        return data


class ElementResponse(ElementSchema, MutableResourceResponseSchema):

    @pre_dump
    def _rename_value_type_field(self, data: dict, **kwargs):
        data = copy.deepcopy(data)
        data['type'] = data.pop('value_type')
        return data


class CategorySchema(BaseSchema):
    name = fields.String(required=True,
                         description='Name of the category (punctuation characters will be replaced with "_")')
    display_name = fields.String(description='Name to be shown', allow_none=True)
    description = fields.String(description='Description of the category', allow_none=True)
    color = fields.String(description='Color of the category', allow_none=True)

    @validates('color')
    def _validate_hex_color(self, color: str):
        validate_hex_color(color)


class CategoryRequest(CategorySchema, ResourceRequestSchema):

    @pre_load
    def _check_punctuation_chars(self, data: dict, **kwargs):
        return check_punctuation_chars(data=data, fields=['name'])

    @pre_load
    def _remove_hex_color_number_sign(self, data: dict, **kwargs) -> dict:
        return remove_hex_color_number_sign(data=data)


class CategoryResponse(CategorySchema, MutableResourceResponseSchema):

    @pre_dump
    def _add_hex_color_number_sign(self, data: dict, **kwargs) -> dict:
        return add_hex_color_number_sign(data=data)


class TaskSchemaResponse(BaseSchema):
    inputs = fields.List(fields.Nested(ElementResponse), required=True, description='List of input elements')
    outputs = fields.List(fields.Nested(ElementResponse), description='List of output elements')
    metadata = fields.List(fields.Nested(ElementResponse), description='List of metadata elements')
    task_type = _task_type_field


class TaskSettingsSchema(BaseSchema):

    class ServicesSettingsSchema(BaseSchema):

        class InferenceServiceSettingsSchema(BaseSchema):
            enabled = fields.Boolean(description='Enable/Disable Inference Service')

        class CLServiceSettingsSchema(BaseSchema):
            enabled = fields.Boolean(description='Enable/Disable Continual Learning Service')
            min_days = fields.Float(description='Maximum frequency at which the AI can be retrained')
            max_days = fields.Float(description='Minimum frequency at which the AI should be retrained')
            min_sample = fields.Float(description='Minimum sample size to trigger retraining, relative to current '
                                      'number of examples. Value between 0 and 1, representing the '
                                      'percentage of current number of examples.')
            min_cpu_quota = fields.Float(description='Minimum CPU quota to be used (in hours)')
            max_cpu_quota = fields.Float(description='Maximum CPU quota to be used (in hours)')
            cpu_hard_limit = fields.Float(description='CPU hard limit for guaranteeing quality of service (in hours)')
            min_gpu_quota = fields.Float(description='Minimum GPU quota to be used (in hours)')
            max_gpu_quota = fields.Float(description='Maximum GPU quota to be used (in hours)')
            gpu_hard_limit = fields.Float(description='GPU hard limit for guaranteeing quality of service (in hours)')

        class ALServiceSettingsSchema(BaseSchema):
            enabled = fields.Boolean(description='Enable/Disable Active Learning Service')
            query_interval = fields.Integer(description='Number of tasks to be completed before a new query is sent')
            max_examples_per_query = fields.Integer(description='Maximum number of examples to be returned in a query')

        class MonitoringServiceSettingsSchema(BaseSchema):

            class OutOfDistributionSettingsSchema(BaseSchema):
                min_sample = fields.Integer(description='Minimum number of predictions required for running detection.')
                sensitivity = fields.Float(description='Sensitivity to anomalies (value between 0 and 1)',
                                           validate=validate.Range(min=0, max=1))
                smoothing = fields.Float(description='Smoothing factor (value between 0 and 1). Low values result in '
                                         'less smoothing and thus a high responsiveness to variations in '
                                         'predictions.',
                                         validate=validate.Range(min=0, max=1))

            enabled = fields.Boolean(description='Enable/Disable Monitoring Service')
            refresh_interval = fields.Integer(description='Interval in which metrics are refreshed (in number of '
                                              'predictions). Setting it to 1 forces metrics to be '
                                              'refreshed each time a new prediction is made.')
            ood_predictions = fields.Nested(OutOfDistributionSettingsSchema,
                                            description='Detection of out-of-distribution (OOD) predictions')

        inference = fields.Nested(InferenceServiceSettingsSchema, description='Inference Service settings')
        continual_learning = fields.Nested(CLServiceSettingsSchema, description='Continual Learning Service settings')
        active_learning = fields.Nested(ALServiceSettingsSchema, description='Active Learning Service settings')
        monitoring = fields.Nested(MonitoringServiceSettingsSchema, description='Monitoring Service settings')

    services = fields.Nested(ServicesSettingsSchema, required=True, description='Services settings')


class TaskQuotaUsageRequest(BaseSchema):
    cpu_hours = fields.Float(allow_none=True, description='CPU hours', validate=validate.Range(min=0))
    gpu_hours = fields.Float(allow_none=True, description='GPU hours', validate=validate.Range(min=0))


class TaskQuotaUsageResponse(BaseSchema):

    class TaskQuotaUsage(BaseSchema):
        cpu_hours = fields.Float(required=True, description='CPU hours')
        gpu_hours = fields.Float(required=True, description='GPU hours')

    class TaskQuotaLimit(BaseSchema):
        cpu_hours = fields.Float(required=True, description='CPU hours')
        gpu_hours = fields.Float(required=True, description='GPU hours')

    usage = fields.Nested(TaskQuotaUsage, required=True, description='Task quota usage')
    limit = fields.Nested(TaskQuotaLimit, required=True, description='Task quota limit')
