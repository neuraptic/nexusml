from marshmallow import fields

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.base import StatusResponseSchema


class ServiceSchema(BaseSchema):
    name = fields.String(required=True, description='Name')
    display_name = fields.String(required=True, description='Name to be shown')
    description = fields.String(required=True, description='Description')
    status = fields.Nested(StatusResponseSchema, required=True, description='Service status')


class ServicesSchema(BaseSchema):
    inference = fields.Nested(ServiceSchema,
                              required=True,
                              description='Inference Service running the AI responsible for '
                              'making predictions in production')
    continual_learning = fields.Nested(ServiceSchema,
                                       required=True,
                                       description='Continual Learning (CL) Service performing '
                                       'periodic retraining of the AI')
    active_learning = fields.Nested(ServiceSchema,
                                    required=True,
                                    description='Active Learning (AL) Service asking experts for data labels')
    monitoring = fields.Nested(ServiceSchema,
                               required=True,
                               description='Monitoring Service which monitors the Inference Service')


class ServiceNotification(BaseSchema):
    message = fields.String(required=True, description='HTML formatted message')


class OutputMonitoringTemplateSchema(BaseSchema):
    element = fields.String(required=True, description='UUID of the output element')
    template = fields.Field(required=True,
                            description='For numerical outputs, a JSON with the following format:'
                            '\n\n```\n{\n\t"mean": number,\n\t"std": number\n}\n```'
                            '\n\nFor categorical outputs, each category '
                            '(identified by a UUID) has its own template:'
                            '\n\n```\n{\n\t"category": string,\n\t"template": object\n}\n```'
                            '\n\nA category template consists of the mean '
                            'softmax probability for each category, given by:'
                            '\n\n```\n{\n\t"category": string,\n\t"mean": number\n}\n```')


class MonitoringServiceTemplatesSchema(BaseSchema):
    ai_model = fields.String(required=True, description='UUID of the AI model to which templates correspond')
    outputs = fields.List(fields.Nested(OutputMonitoringTemplateSchema),
                          required=True,
                          description='Template for each output element')
