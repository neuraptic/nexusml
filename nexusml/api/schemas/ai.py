from marshmallow import fields
from marshmallow import validate
from marshmallow_enum import EnumField

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.base import ImmutableResourceResponseSchema
from nexusml.api.schemas.base import PageSchema
from nexusml.api.schemas.base import ResourceRequestSchema
from nexusml.api.schemas.files import TaskFileResponse as FileResponse
from nexusml.api.schemas.utils import ElementValue
from nexusml.constants import API_NAME
from nexusml.enums import AIEnvironment
from nexusml.enums import PredictionState
from nexusml.enums import TrainingDevice

##########################
# Training and Inference #
##########################


class TrainingRequest(BaseSchema):
    pass


class ObservedValue(ElementValue):
    element = fields.String(required=True, description='Input/Metadata element to which the value refers')
    value = fields.Field(required=True, allow_none=True, description='Input/Metadata value. For files, provide the ID')


class PredictedValue(ElementValue):
    element = fields.String(required=True, description='Output element to which the value refers')
    value = fields.Field(required=True,
                         allow_none=True,
                         description=('Predicted value.'
                                      '\n\nCategorical values follow a specific format:'
                                      '\n```'
                                      '\n{'
                                      '\n\t"category": "class_1",'
                                      '\n\t"scores": {'
                                      '\n\t\t"class_1": <class_1_score>,'
                                      '\n\t\t"class_2": <class_2_score>,'
                                      '\n\t\t"class_3": <class_3_score>,'
                                      '\n\t}'
                                      '\n}'
                                      '\n```'
                                      '\n* Field "category" contains the category with the highest score.'
                                      '\n* Field "scores" contains the score of each category (float).'))


class Observation(BaseSchema):
    values = fields.List(fields.Nested(ObservedValue), required=True, description='Input/Metadata values')


class Prediction(BaseSchema):
    outputs = fields.List(fields.Nested(PredictedValue), required=True, description='Predicted values')


class InferenceRequest(BaseSchema):
    batch = fields.List(fields.Nested(Observation),
                        required=True,
                        description='Batch of observations on which predictions will be made')


class InferenceResponse(BaseSchema):
    predictions = fields.List(fields.Nested(Prediction),
                              required=True,
                              description='Predictions made on the provided batch of observations. '
                              '\n\nNote: the order in which predictions are returned '
                              'corresponds to the order of observations within the batch.')
    ai_model = fields.String(required=True, description='ID of the AI model which made the predictions')


##############
# AI Testing #
##############


class TestTarget(ElementValue):
    element = fields.String(required=True, description='Output element to which the target value refers')
    value = fields.Field(required=True, allow_none=True, description='Target value. For files, provide the ID')


_test_targets = fields.List(fields.Nested(TestTarget),
                            required=False,
                            allow_none=True,
                            description='Target values to use for metrics and stats.')


class TestRequest(BaseSchema):

    class TestObservation(Observation):
        targets = _test_targets

    batch = fields.List(fields.Nested(TestObservation),
                        required=True,
                        description='Batch of test observations on which predictions will be made')


#############
# AI Models #
#############

# TODO: move this section up


class AIModelSchema(BaseSchema):
    extra_metadata = fields.Dict(allow_none=True, description='Extra metadata stored for the AI model')
    version = fields.String(required=True,
                            validate=validate.Length(min=5, max=16),
                            description='The version of the AI model in X.Y.Z format')


class AIModelRequest(AIModelSchema, ResourceRequestSchema):
    file = fields.String(required=True,
                         description=f'File containing the AI model. '
                         f'WARNING: the file must be previously uploaded to {API_NAME}')
    training_time = fields.Float(required=True, description='Time needed for training the model (in hours)')
    training_device = EnumField(TrainingDevice,
                                required=True,
                                description='Device used for training the model: "cpu" | "gpu"')


class AIModelResponse(AIModelSchema, ImmutableResourceResponseSchema):
    version = fields.String(required=True, description='Version')
    file = fields.Nested(FileResponse, required=True, description='File containing the AI model')
    task_schema = fields.Dict(required=True, description='Task schema for which the AI model was built')


class _DeploymentSchema(BaseSchema):
    environment = EnumField(AIEnvironment, required=True, description='Environment: "production" or "testing"')


class DeploymentRequest(_DeploymentSchema):
    ai_model = fields.String(required=True, description='Identifier of the AI model to be deployed')


class DeploymentResponse(_DeploymentSchema, AIModelResponse):
    pass


###################
# Prediction Logs #
###################


class PredictionLogSchema(BaseSchema):
    state = EnumField(PredictionState,
                      required=True,
                      description=' | '.join(f'"{x.name.lower()}"' for x in PredictionState))
    inputs = fields.List(fields.Nested(ObservedValue), required=True, description='Input values')
    outputs = fields.List(fields.Nested(PredictedValue), allow_none=True, description='Predicted values')
    metadata = fields.List(fields.Nested(ObservedValue), allow_none=True, description='Metadata values')
    targets = _test_targets


class PredictionLogRequest(PredictionLogSchema, ResourceRequestSchema):
    ai_model = fields.String(required=True, description='ID of the AI model which made the predictions')


class PredictionLogResponse(PredictionLogSchema, ImmutableResourceResponseSchema):
    ai_model = fields.String(
        required=True,
        allow_none=True,  # The AI model could have been deleted
        description='ID of the AI model which made the predictions')
    environment = EnumField(AIEnvironment,
                            required=True,
                            description=('"production" if the predictions were made in production '
                                         'or "testing" if the predictions were made in testing environment'))
    invalid_data = fields.Dict(allow_none=True, description='Invalid values')
    removed_elements = fields.Dict(allow_none=True,
                                   description='Values predicted for elements removed from task schema')

    # # TODO: Uncomment this snippet if we want to accept target values only on test predictions
    # @pre_dump
    # def _ensure_no_production_targets(self, data, **kwargs) -> dict:
    #     if data.get('targets') and data['environment'] == AIEnvironment.PRODUCTION:
    #         raise ValidationError('Predictions made in production should not have target values')
    #     return data


class PredictionLogPage(PageSchema):
    data = fields.List(fields.Nested(PredictionLogResponse),
                       required=True,
                       description='Predictions in the requested page (or in the first page, if not specified)')


class PredictionLoggingRequest(BaseSchema):
    predictions = fields.List(fields.Nested(PredictionLogRequest), required=True)
