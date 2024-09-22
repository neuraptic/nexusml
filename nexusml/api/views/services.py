import copy
from datetime import datetime
import os
from typing import List, Union

from flask import jsonify
from flask import render_template
from flask import request
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs
from flask_mail import Message
from marshmallow import ValidationError
from sqlalchemy import and_ as sql_and

from nexusml.api.ext import mail
from nexusml.api.resources.base import Resource
from nexusml.api.resources.organizations import User
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.base import StatusRequestSchema
from nexusml.api.schemas.base import StatusResponseSchema
from nexusml.api.schemas.services import MonitoringServiceTemplatesSchema
from nexusml.api.schemas.services import ServiceNotification
from nexusml.api.schemas.services import ServiceSchema
from nexusml.api.schemas.services import ServicesSchema
from nexusml.api.views.base import create_view
from nexusml.api.views.core import allowed_clients
from nexusml.api.views.core import error_response
from nexusml.constants import AL_SERVICE_NAME
from nexusml.constants import API_NAME
from nexusml.constants import CL_SERVICE_NAME
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import HTTP_NOT_FOUND_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import INFERENCE_SERVICE_NAME
from nexusml.constants import MONITORING_SERVICE_NAME
from nexusml.constants import SWAGGER_TAG_SERVICES
from nexusml.constants import TESTING_SERVICE_NAME
from nexusml.database.core import db_commit
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.services import Service
from nexusml.engine.services.base import update_service_status
from nexusml.engine.services.monitoring import MonitoringService
from nexusml.enums import LabelingStatus
from nexusml.enums import ServiceType
from nexusml.env import ENV_NOTIFICATION_EMAIL
from nexusml.statuses import CL_TRAINING_STATUS_CODE

################
# Define views #
################

_View = create_view(resource_types=[Task])

_inference_service = {
    'name': INFERENCE_SERVICE_NAME,
    'display_name': 'Inference Service',
    'description': 'Service running the AI responsible for making predictions in production'
}
_cl_service = {
    'name': CL_SERVICE_NAME,
    'display_name': 'Continual Learning Service',
    'description': 'Service performing periodic retraining of the AI'
}
_al_service = {
    'name': AL_SERVICE_NAME,
    'display_name': 'Active Learning Service',
    'description': 'Service asking experts for data labels'
}
_monitoring_service = {
    'name': MONITORING_SERVICE_NAME,
    'display_name': 'Monitoring Service',
    'description': 'Service which monitors the Inference Service'
}
_testing_service = {
    'name': TESTING_SERVICE_NAME,
    'display_name': 'Testing Service',
    'description': 'Service running the AI responsible for making predictions in testing environment'
}
"""
Service info
"""


class ServicesView(_View):
    """
    A view that provides an overview of various services associated with a given task.
    This includes fetching the status of different services such as inference, continual learning,
    active learning, monitoring, and testing services.
    """

    @doc(tags=[SWAGGER_TAG_SERVICES])
    @marshal_with(ServicesSchema)
    def get(self, task_id: str, resources: List[Resource]):
        """
        Handle the GET request to fetch the status of various services associated with a task.

        Steps:
        - Fetch the task object using resources.
        - Retrieve status for inference, continual learning, active learning, monitoring, and testing services.
        - Deep copy service templates and update them with the service status.
        - Construct the response and return it as a JSON object.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.

        Returns:
            Response: The response containing the status of the services.
        """
        task = resources[-1].db_object()

        inference_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.INFERENCE)
        cl_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.CONTINUAL_LEARNING)
        al_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.ACTIVE_LEARNING)
        monitoring_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.MONITORING)
        testing_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.TESTING)

        inference_json = copy.deepcopy(_inference_service)
        cl_json = copy.deepcopy(_cl_service)
        al_json = copy.deepcopy(_al_service)
        monitoring_json = copy.deepcopy(_monitoring_service)
        testing_json = copy.deepcopy(_testing_service)

        inference_json['status'] = inference_service.to_dict()['status']
        cl_json['status'] = cl_service.to_dict()['status']
        al_json['status'] = al_service.to_dict()['status']
        monitoring_json['status'] = monitoring_service.to_dict()['status']
        testing_json['status'] = testing_service.to_dict()['status']

        services = {
            'inference': inference_json,
            'continual_learning': cl_json,
            'active_learning': al_json,
            'monitoring': monitoring_json,
            'testing': testing_json
        }

        return jsonify(services)


class InferenceServiceView(_View):
    """
    A view that provides the status of the Inference Service associated with a given task.
    """

    @doc(tags=[SWAGGER_TAG_SERVICES])
    @marshal_with(ServiceSchema)
    def get(self, task_id: str, resources: List[Resource]):
        """
        Handle the GET request to fetch the status of the Inference Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.

        Returns:
            Response: The response containing the status of the Inference Service.
        """
        task = resources[-1].db_object()
        inference_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.INFERENCE)
        return jsonify({**_inference_service, 'status': inference_service.to_dict()['status']})


class CLServiceView(_View):
    """
    A view that provides the status of the Continual Learning Service associated with a given task.
    """

    @doc(tags=[SWAGGER_TAG_SERVICES])
    @marshal_with(ServiceSchema)
    def get(self, task_id: str, resources: List[Resource]):
        """
        Handle the GET request to fetch the status of the Continual Learning Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.

        Returns:
            Response: The response containing the status of the Continual Learning Service.
        """
        task = resources[-1].db_object()
        cl_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.CONTINUAL_LEARNING)
        return jsonify({**_cl_service, 'status': cl_service.to_dict()['status']})


class ALServiceView(_View):
    """
    A view that provides the status of the Active Learning Service associated with a given task.
    """

    @doc(tags=[SWAGGER_TAG_SERVICES])
    @marshal_with(ServiceSchema)
    def get(self, task_id: str, resources: List[Resource]):
        """
        Handle the GET request to fetch the status of the Active Learning Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.

        Returns:
            Response: The response containing the status of the Active Learning Service.
        """
        task = resources[-1].db_object()
        al_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.ACTIVE_LEARNING)
        return jsonify({**_al_service, 'status': al_service.to_dict()['status']})


class MonitoringServiceView(_View):
    """
    A view that provides the status of the Monitoring Service associated with a given task.
    """

    @doc(tags=[SWAGGER_TAG_SERVICES])
    @marshal_with(ServiceSchema)
    def get(self, task_id: str, resources: List[Resource]):
        """
        Handle the GET request to fetch the status of the Monitoring Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.

        Returns:
            Response: The response containing the status of the Monitoring Service.
        """
        task = resources[-1].db_object()
        monitoring_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.MONITORING)
        return jsonify({**_monitoring_service, 'status': monitoring_service.to_dict()['status']})


class TestingServiceView(_View):
    """
    A view that provides the status of the Testing Service associated with a given task.
    """

    @doc(tags=[SWAGGER_TAG_SERVICES])
    @marshal_with(ServiceSchema)
    def get(self, task_id: str, resources: List[Resource]):
        """
        Handle the GET request to fetch the status of the Testing Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.

        Returns:
            Response: The response containing the status of the Testing Service.
        """
        task = resources[-1].db_object()
        testing_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.TESTING)
        return jsonify({**_testing_service, 'status': testing_service.to_dict()['status']})


##################
# Service status #
##################


def _update_service_status(task: Task, service_type: ServiceType, status: dict) -> Response:
    """
    Update the status of a given service type associated with a task.

    Args:
        task (Task): The task associated with the service.
        service_type (ServiceType): The type of the service to be updated.
        status (dict): The new status to be set for the service.

    Returns:
        Response: The response containing the updated status of the service.

    Raises:
        UnprocessableRequestError: If the service is disabled in the current plan.
        InvalidDataError: If the status is invalid or not found.
        ValueError: For other validation errors.
    """

    @allowed_clients(service_types=[service_type], error_code=HTTP_NOT_FOUND_STATUS_CODE)
    def _update_status(resources) -> Union[Response, None]:
        """
        Internal function that handles the status update request.

        WARNING: don't remove `resources` argument, as `allowed_clients` get the parent task
                 from the resources loaded by the view (injected into `resources`)

        Args:
            resources (List[Resource]): List of resources associated with the request.

        Returns:
            Response: The response containing the updated status or None if not applicable.
        """
        return jsonify(update_service_status(task_db_obj=task.db_object(), service_type=service_type, status=status))

    return _update_status(resources=[task])


class InferenceServiceStatusView(_View):
    """
    A view to update the status of the Inference Service associated with a given task.
    """

    @use_kwargs(StatusRequestSchema, location='json')
    @marshal_with(StatusResponseSchema)
    def put(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the PUT request to update the status of the Inference Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The status details to update.

        Returns:
            Response: The response containing the updated status of the Inference Service.
        """
        return _update_service_status(task=resources[-1], service_type=ServiceType.INFERENCE, status=kwargs)


class CLServiceStatusView(_View):
    """
    A view to update the status of the Continual Learning Service associated with a given task.
    """

    @use_kwargs(StatusRequestSchema, location='json')
    @marshal_with(StatusResponseSchema)
    def put(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the PUT request to update the status of the Continual Learning Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The status details to update.

        Returns:
            Response: The response containing the updated status of the Continual Learning Service.
        """
        task = resources[-1].db_object()
        cl_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.CONTINUAL_LEARNING)
        # Handle status transitions
        if kwargs['code'] != cl_service.status['code']:
            # If finishing training session, register session end datetime and mark trained examples
            if cl_service.status['code'] == CL_TRAINING_STATUS_CODE and kwargs['code'].startswith('02'):
                cl_service.data = {'last_end': datetime.utcnow().strftime(DATETIME_FORMAT), **cl_service.data}
                # Mark all examples with `labeling_status="labeled" and activity_at < last_start` as `trained=True`
                last_start = datetime.strptime(cl_service.data['last_start'], DATETIME_FORMAT)
                trained_examples = (ExampleDB.query().filter(
                    sql_and(ExampleDB.task_id == task.task_id, ExampleDB.labeling_status == LabelingStatus.LABELED,
                            ExampleDB.activity_at < last_start)))
                trained_examples.update({'trained': True})
                db_commit()
            # If starting training session, register session start datetime
            elif kwargs['code'] == CL_TRAINING_STATUS_CODE:
                cl_service.data = {'last_start': datetime.utcnow().strftime(DATETIME_FORMAT), **cl_service.data}
                db_commit()
        # Update service status
        return _update_service_status(task=resources[-1], service_type=ServiceType.CONTINUAL_LEARNING, status=kwargs)


class ALServiceStatusView(_View):
    """
    A view to update the status of the Active Learning Service associated with a given task.
    """

    @use_kwargs(StatusRequestSchema, location='json')
    @marshal_with(StatusResponseSchema)
    def put(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the PUT request to update the status of the Active Learning Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The status details to update.

        Returns:
            Response: The response containing the updated status of the Active Learning Service.
        """
        return _update_service_status(task=resources[-1], service_type=ServiceType.ACTIVE_LEARNING, status=kwargs)


class MonitoringServiceStatusView(_View):
    """
    A view to update the status of the Monitoring Service associated with a given task.
    """

    @use_kwargs(StatusRequestSchema, location='json')
    @marshal_with(StatusResponseSchema)
    def put(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the PUT request to update the status of the Monitoring Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The status details to update.

        Returns:
            Response: The response containing the updated status of the Monitoring Service.
        """
        return _update_service_status(task=resources[-1], service_type=ServiceType.MONITORING, status=kwargs)


class TestingServiceStatusView(_View):
    """
    A view to update the status of the Testing Service associated with a given task.
    """

    @use_kwargs(StatusRequestSchema, location='json')
    @marshal_with(StatusResponseSchema)
    def put(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the PUT request to update the status of the Testing Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The status details to update.

        Returns:
            Response: The response containing the updated status of the Testing Service.
        """
        return _update_service_status(task=resources[-1], service_type=ServiceType.TESTING, status=kwargs)


#########################
# Service notifications #
#########################


def _send_service_notification(task: Task, service_type: ServiceType, payload: dict) -> Response:
    """
    Send a service notification email to users with access to the task.

    Steps:
    - Ensure the request comes from allowed clients for the specified service type.
    - Construct and send email notifications to each user.
    - Return a JSON response with the notification payload.

    Args:
        task (Task): The task associated with the service.
        service_type (ServiceType): The type of the service.
        payload (dict): The notification message payload.

    Returns:
        Response: The response containing the notification payload.
    """

    @allowed_clients(service_types=[service_type], error_code=HTTP_NOT_FOUND_STATUS_CODE)
    def _send_email_notification(resources):
        """
        Internal function that sends notification emails.

        WARNING: don't remove `resources` argument, as `allowed_clients` get the parent task
                 from the resources loaded by the view (injected into `resources`)

        Steps:
        - This function is wrapped with the `allowed_clients` decorator.
        - It constructs the notification email and sends it to the users.

        Args:
            resources (List[Resource]): List of resources associated with the request.
        """
        task = resources[-1]
        service_names = {
            ServiceType.INFERENCE: 'Inference',
            ServiceType.CONTINUAL_LEARNING: 'Continual Learning',
            ServiceType.ACTIVE_LEARNING: 'Active Learning',
            ServiceType.MONITORING: 'Monitoring',
            ServiceType.TESTING: 'Testing'
        }
        sender = (API_NAME, os.environ[ENV_NOTIFICATION_EMAIL])
        recipients = [User.download_auth0_user_data(auth0_id_or_email=x.auth0_id) for x in task.users_with_access()]
        for recipient in recipients:
            try:
                html = render_template('service_notifications.html',
                                       app_name=API_NAME,
                                       service_name=service_names[service_type],
                                       recipient_name=recipient['first_name'],
                                       task_name=task.db_object().name,
                                       task_url=task.url(),
                                       message=payload['message'])
                subject = f'[{API_NAME}] {recipient["first_name"]}, you have new notifications'
                msg = Message(sender=sender, recipients=[recipient['email']], subject=subject, html=html)
                mail.send(msg)
            except Exception:
                pass

    _send_email_notification(resources=[task])

    response = jsonify(payload)
    response.status_code = HTTP_POST_STATUS_CODE
    response.headers['Location'] = request.url

    return response


class InferenceServiceNotificationsView(_View):
    """
    A view to send notifications for the Inference Service associated with a given task.
    """

    @use_kwargs(ServiceNotification, location='json')
    @marshal_with(ServiceNotification)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the POST request to send notifications for the Inference Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The notification details to send.

        Returns:
            Response: The response containing the notification payload.
        """
        return _send_service_notification(task=resources[-1], service_type=ServiceType.INFERENCE, payload=kwargs)


class CLServiceNotificationsView(_View):
    """
    A view to send notifications for the Continual Learning Service associated with a given task.
    """

    @use_kwargs(ServiceNotification, location='json')
    @marshal_with(ServiceNotification)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the POST request to send notifications for the Continual Learning Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The notification details to send.

        Returns:
            Response: The response containing the notification payload.
        """
        return _send_service_notification(task=resources[-1],
                                          service_type=ServiceType.CONTINUAL_LEARNING,
                                          payload=kwargs)


class ALServiceNotificationsView(_View):
    """
    A view to send notifications for the Active Learning Service associated with a given task.
    """

    @use_kwargs(ServiceNotification, location='json')
    @marshal_with(ServiceNotification)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the POST request to send notifications for the Active Learning Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The notification details to send.

        Returns:
            Response: The response containing the notification payload.
        """
        return _send_service_notification(task=resources[-1], service_type=ServiceType.ACTIVE_LEARNING, payload=kwargs)


class MonitoringServiceNotificationsView(_View):
    """
    A view to send notifications for the Monitoring Service associated with a given task.
    """

    @use_kwargs(ServiceNotification, location='json')
    @marshal_with(ServiceNotification)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the POST request to send notifications for the Monitoring Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The notification details to send.

        Returns:
            Response: The response containing the notification payload.
        """
        return _send_service_notification(task=resources[-1], service_type=ServiceType.MONITORING, payload=kwargs)


class TestingServiceNotificationsView(_View):
    """
    A view to send notifications for the Testing Service associated with a given task.
    """

    @use_kwargs(ServiceNotification, location='json')
    @marshal_with(ServiceNotification)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the POST request to send notifications for the Testing Service.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The notification details to send.

        Returns:
            Response: The response containing the notification payload.
        """
        return _send_service_notification(task=resources[-1], service_type=ServiceType.TESTING, payload=kwargs)


#############################
# Monitoring-specific stuff #
#############################


class MonitoringServiceTemplatesView(_View):
    """
    A view to manage templates for the Monitoring Service associated with a given task.
    """

    @staticmethod
    def _verify_templates(templates: dict, task_id: int):
        """
        Verify the validity of the provided templates using the schema and task ID.

        Args:
            templates (dict): Dictionary of templates to verify.
            task_id (int): The ID of the task associated with the templates.

        Returns:
            None

        Raises:
            ValidationError: If the templates do not meet the schema's validation.
            ValueError: If there is a custom validation error.
        """
        templates = MonitoringServiceTemplatesSchema().load(templates)
        MonitoringService.verify_templates(templates=templates, task_id=task_id)

    @allowed_clients(service_types=[ServiceType.MONITORING], error_code=HTTP_NOT_FOUND_STATUS_CODE)
    @marshal_with(MonitoringServiceTemplatesSchema)
    def get(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the GET request to fetch the monitoring service templates.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: Additional arguments, if any.

        Returns:
            Response: The response containing the verified templates.
        """
        task = resources[-1]
        service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.MONITORING)
        try:
            self._verify_templates(templates=service.data, task_id=task.db_object().task_id)
            return jsonify(MonitoringServiceTemplatesSchema().dump(service.data))
        except (ValueError, ValidationError) as e:
            return error_response(code=400, message=f'Templates are corrupted: {e}')

    @allowed_clients(service_types=[ServiceType.CONTINUAL_LEARNING], error_code=HTTP_NOT_FOUND_STATUS_CODE)
    @use_kwargs(MonitoringServiceTemplatesSchema, location='json')
    @marshal_with(MonitoringServiceTemplatesSchema)
    def put(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handle the PUT request to update the monitoring service templates.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of resources associated with the request.
            kwargs: The templates to update.

        Returns:
            Response: The response containing the updated templates.
        """
        task = resources[-1]
        # Verify templates
        try:
            self._verify_templates(templates=kwargs, task_id=task.db_object().task_id)
        except ValueError as e:
            return error_response(code=400, message=f'Invalid templates: {e}')
        # Save templates
        task = resources[-1]
        service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.MONITORING)
        service.data = kwargs
        save_to_db(service)
        # Build response
        response = jsonify(MonitoringServiceTemplatesSchema().dump(service.data))
        response.status_code = HTTP_PUT_STATUS_CODE
        response.headers['Location'] = request.url
        return response
