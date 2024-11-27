# TODO: Try to make this module independent from `nexusml.api`

import os

from flask import render_template
from flask_mail import Message

from nexusml.api.ext import mail
from nexusml.api.resources.organizations import User
from nexusml.api.resources.tasks import Task
from nexusml.constants import API_NAME
from nexusml.enums import ServiceType
from nexusml.env import ENV_NOTIFICATION_EMAIL


def send_email_notification(task: Task, service_type: ServiceType, payload: dict) -> None:
    """
    Sends an email notification to all users with access to the task, with the given payload.

    Args:
        task (Task): The task object.
        service_type (ServiceType): The type of service that is sending the notification.
        payload (dict): The payload to be sent in
    """
    service_names = {
        ServiceType.INFERENCE: 'Inference',
        ServiceType.CONTINUAL_LEARNING: 'Continual Learning',
        ServiceType.ACTIVE_LEARNING: 'Active Learning',
        ServiceType.MONITORING: 'Monitoring',
        ServiceType.TESTING: 'Testing'
    }
    sender = (API_NAME, os.environ[ENV_NOTIFICATION_EMAIL])
    recipients: list = list()
    for user_db_obj in task.users_with_access():
        if user_db_obj.auth0_id:
            recipients.append(User.download_auth0_user_data(auth0_id_or_email=user_db_obj.auth0_id))

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
        except Exception as e:
            print(f'Unable to send email - {str(e)}')
