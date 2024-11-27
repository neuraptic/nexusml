import copy

from marshmallow import fields
from marshmallow import pre_dump
from marshmallow_enum import EnumField

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.organizations import RoleResponseSchema
from nexusml.constants import DATETIME_FORMAT
from nexusml.enums import NotificationEvent
from nexusml.enums import NotificationSource
from nexusml.enums import NotificationType

############
# Settings #
############


class SettingsSchema(BaseSchema):
    notifications = EnumField(NotificationType, required=True, description='"polling" | "push" | "email"')


class _ClientSettingsSchema(BaseSchema):
    client_version = fields.String(required=True, description='Client version')
    settings = fields.Dict(allow_none=True, description='JSON with the user settings for the specified client version')


class ClientSettingsRequestSchema(_ClientSettingsSchema):
    pass


class ClientSettingsResponseSchema(_ClientSettingsSchema):
    client_id = fields.String(required=True, description='Client identifier')


#################
# Notifications #
#################


class _BaseNotificationSchema(BaseSchema):
    # WARNING: notifications don't expose their public ID because
    #          individual and aggregated notifications share the same endpoint but have different database tables.
    _notification_sources = ' | '.join([f'"{x.name.lower()}"' for x in NotificationSource])
    _notification_events = ' | '.join([f'"{x.name.lower()}"' for x in NotificationEvent])

    uuid = fields.String(required=True, description='Universal identifier of the notification')
    source_type = EnumField(NotificationSource, required=True, description=f'Source type: {_notification_sources}')
    event = EnumField(NotificationEvent, required=True, description=f'Event: {_notification_events}')


class NotificationSchema(_BaseNotificationSchema):
    datetime = fields.DateTime(format=DATETIME_FORMAT,
                               required=True,
                               description='Datetime at which the event occurred')
    source_url = fields.String(required=True, description='Source URL')

    @pre_dump
    def _rename_datetime_field(self, data, **kwargs) -> dict:
        data = copy.deepcopy(data)
        data['datetime'] = data.pop('created_at')
        return data


class AggregatedNotificationSchema(_BaseNotificationSchema):
    since = fields.DateTime(format=DATETIME_FORMAT,
                            required=True,
                            description='Oldest datetime among the aggregated notifications')
    count = fields.Integer(required=True, description='Number of notifications that were aggregated')


#########
# Roles #
#########


class MyAccountRolesSchema(BaseSchema):
    roles = fields.List(fields.Nested(RoleResponseSchema), required=True, allow_none=True, description='Assigned roles')
