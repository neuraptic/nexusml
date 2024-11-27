from abc import ABC
from abc import abstractmethod
import builtins
from collections import namedtuple
import copy
from datetime import datetime
import functools
from typing import Callable, Dict, Iterable, List, Optional, Set, Type, Union

from sqlalchemy import and_ as sql_and
from sqlalchemy.exc import DatabaseError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.exc import StatementError
from sqlalchemy.inspection import inspect

from nexusml.api.ext import cache
from nexusml.api.schemas.base import ResourceRequestSchema
from nexusml.api.schemas.base import ResourceResponseSchema
from nexusml.api.utils import API_DOMAIN
from nexusml.api.utils import config
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NULL_UUID
from nexusml.database.base import Entity
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.core import db_query
from nexusml.database.core import db_rollback
from nexusml.database.core import delete_from_db
from nexusml.database.core import retry_on_deadlock
from nexusml.database.core import save_to_db
from nexusml.database.notifications import AggregatedNotificationDB
from nexusml.database.notifications import NotificationDB
from nexusml.database.organizations import Agent
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import ImmutableEntity
from nexusml.database.organizations import MutableEntity
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles as user_roles_table
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.tasks import TaskDB
from nexusml.enums import DBRelationshipType
from nexusml.enums import NotificationEvent
from nexusml.enums import NotificationSource
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceCollectionOperation
from nexusml.enums import ResourceType

__all__ = [
    'Permission',
    'InvalidDataError',
    'PermissionDeniedError',
    'UnprocessableRequestError',
    'ResourceError',
    'ResourceNotFoundError',
    'DuplicateResourceError',
    'ImmutableResourceError',
    'ResourceOutOfSyncError',
    'QuotaError',
    'Resource',
    'dump',
    'filter_effective_permissions',
    'users_permissions',
    'collaborators_permissions',
]

Permission = namedtuple('Permission', ['resource_uuid', 'resource_type', 'action', 'allow'])

##########
# Errors #
##########


class InvalidDataError(ValueError):

    def __init__(self, msg='Invalid request data', *args):
        super().__init__(msg, *args)


class PermissionDeniedError(PermissionError):

    def __init__(self, msg='Permission denied', *args):
        super().__init__(msg, *args)


class UnprocessableRequestError(Exception):

    def __init__(self, msg='Unprocessable request', *args):
        super().__init__(msg, *args)


class ResourceError(Exception):

    def __init__(self, msg='Resource error', resource_id: str = None, *args):
        self.resource_id = resource_id
        if resource_id:
            msg += f': "{resource_id}"'
        super().__init__(msg, *args)


class ResourceNotFoundError(ResourceError):

    def __init__(self, msg='Resource not found', resource_id: str = None, *args):
        super().__init__(msg, resource_id, *args)


class DuplicateResourceError(ResourceError):

    def __init__(self, msg='Resource already exists', resource_id: str = None, *args):
        super().__init__(msg, resource_id, *args)


class ImmutableResourceError(ResourceError):

    def __init__(self, msg='Resource cannot be modified', resource_id: str = None, *args):
        super().__init__(msg, resource_id, *args)


class ResourceOutOfSyncError(ResourceError):

    def __init__(self, msg='Resource out of sync', resource_id: str = None, *args):
        super().__init__(msg, resource_id, *args)


class QuotaError(Exception):

    def __init__(self, msg='Quota exceeded', *args):
        super().__init__(msg, *args)


###################
# Resource Models #
###################


def _permissions_required(action: ResourceAction):
    """ Decorator for `Resource` object-level methods where permissions for the given action are required. """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self._check_permissions(action=action)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class Resource(ABC):
    """ Represents a REST API resource. """

    def __init__(self):
        self._url = None
        self._user = None
        self._client = None
        self._db_object = None
        self._parents = []
        self._cached = False

    @classmethod
    @abstractmethod
    def db_model(cls) -> Type[Entity]:
        """ Database model used for storing resource data. """
        pass

    @classmethod
    @abstractmethod
    def load_schema(cls) -> Type[ResourceRequestSchema]:
        """ Marshmallow schema used for reading JSON data from the client. """
        pass

    @classmethod
    @abstractmethod
    def dump_schema(cls) -> Type[ResourceResponseSchema]:
        """ Marshmallow schema used for writing JSON data to the client. """
        pass

    @classmethod
    @abstractmethod
    def location(cls) -> str:
        """
        Parameterized location (e.g. `/tasks/<task_id>/examples/<example_id>`).
        TODO: should we add a `parent_model()` class method to specify the parent resource model (if any)?
              This would allow us to ensure parent-child relationship consistency at resource level.
        """
        pass

    @classmethod
    def collections(cls) -> dict:
        """ Child collections (key = field name; value = type of `Resource`). """
        return dict()

    @classmethod
    def associations(cls) -> dict:
        """
        Associations with other resources (key = field name; value = type of `Resource`, `Entity`, or `Association`)
        """
        return dict()

    @classmethod
    def permission_resource_type(cls) -> Optional[ResourceType]:
        return

    @classmethod
    def notification_source_type(cls) -> Optional[NotificationSource]:
        return

    @classmethod
    def touch_parent(cls) -> bool:
        """
        Touch direct parent's modification datetime in POST, PUT and DELETE requests.
        WARNING: if `True`, each POST|PUT|DELETE request causes parent resource's data to be uncached.
        """
        return False

    def uuid(self) -> str:
        return str(self.db_object().uuid)

    def public_id(self) -> str:
        return self.db_object().public_id

    def url(self) -> str:
        if self._url:
            return self._url
        api_url = config.get('server')['api_url']
        self._url = API_DOMAIN + api_url + self.location()
        url_params = [x for x in self.location().split('/') if x.startswith('<') and x.endswith('>')]
        assert len(url_params) > 0
        assert len(self.parents()) == len(url_params) - 1
        if self.parents():
            for param, parent in zip(url_params[:-1], self.parents()):
                self._url = self._url.replace(param, parent.uuid())
        self._url = self._url.replace(url_params[-1], self.uuid())
        return self._url

    def user(self) -> UserDB:
        return self._user

    def client(self) -> ClientDB:
        return self._client

    def agent(self) -> Agent:
        return self._user or self._client

    def parents(self) -> list:
        """ Returns the `Resource` parent instances. """
        return self._parents

    def db_object(self) -> Entity:
        return self._db_object

    def cached(self) -> bool:
        """ Indicates whether resource data is cached or not. """
        return self._cached

    @classmethod
    def get(cls,
            agent: Agent,
            db_object_or_id: Union[Entity, str],
            parents: list = None,
            cache: bool = False,
            check_permissions: bool = True,
            check_parents: bool = True,
            remove_notifications: bool = False):
        """ Gets an existing resource

        Args:
            agent (Agent): user or client accessing the resource.
            db_object_or_id (Union[Entity, str]): database object or resource identifier
                                                  (UUID, public ID, or unique value)
            parents (list): parent resources (in order of appearance in the URL), represented by `Resource`
                            instances or tuples of:
                                - Type[Resource]: resource type
                                - str: UUID or public ID of the resource
            cache (bool): cache resource data to speed up its retrieval
            check_permissions (bool): check permissions. If the given user doesn't have enough permissions, a
                                      `PermissionDeniedError` error will be raised.
            check_parents (bool): check parents' permissions
            remove_notifications (bool): remove the notifications created by the specified resource

        Returns
            Resource: resource instance
        """

        resource = cls()
        if isinstance(agent, UserDB):
            resource._user = agent
        else:
            resource._client = agent

        # Set parents
        if parents:
            resource._set_parents(parents)

        # Load resource data from the database or get it from the cache
        if isinstance(db_object_or_id, Entity):
            resource._db_object = db_object_or_id
        else:
            direct_parent = resource.parents()[-1] if resource.parents() else None
            if cache:
                resource._cached = True
                resource._db_object = cls._get_db_object(resource_id=db_object_or_id, parent=direct_parent)
            else:
                resource._db_object = cls._load_db_object(resource_id=db_object_or_id, parent=direct_parent)

        # Check permissions
        # WARNING: permissions cannot be checked before loading resource data, as the UUID is required for checking
        #          resource-level permissions. Since user sync state may be updated during data loading, this may lead
        #          to an inconsistent state (a resource marked as "synced" that couldn't actually be retrieved)
        if check_permissions:
            resource._check_permissions(action=ResourceAction.READ, check_parents=check_parents)

        # Remove all the user's notifications created by the resource
        if remove_notifications and isinstance(agent, UserDB):
            notifications = NotificationDB.filter_by_recipient_source(recipient=agent.user_id,
                                                                      source_uuid=resource.uuid())
            delete_from_db(notifications)

        return resource

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """ Creates a new resource

        Args:
            agent (Agent): user or client creating the resource.
            data (dict): resource data
            parents (list): parent resources (in order of appearance in the URL), represented by `Resource`
                            instances or tuples of:
                                - Type[Resource]: resource type
                                - str: UUID or public ID of the resource
            check_permissions (bool): check permissions. If the given user doesn't have enough permissions, a
                                      `PermissionDeniedError` error will be raised.
            check_parents (bool): check parents' permissions
            notify_to (Iterable): users to which notify about the creation of the resource

        Returns
            Resource: created resource
        """

        resource = cls()
        if isinstance(agent, UserDB):
            resource._user = agent
        else:
            resource._client = agent

        # Set parents
        if parents:
            resource._set_parents(parents)

        # Uncache direct parent's data if it is touched by the resource
        if resource.parents() and resource.touch_parent():
            direct_parent = resource.parents()[-1]
            direct_parent.uncache()

        try:
            # Set resource data
            resource._set_data(data, notify_to)

            # Check permissions
            # WARNING: permissions cannot be checked before loading resource data,
            #          as the UUID is required for checking resource-level permissions
            if check_permissions:
                resource._check_permissions(action=ResourceAction.CREATE, check_parents=check_parents)

            # Set states
            utcnow = datetime.utcnow()
            resource._set_immutable_state(datetime_=utcnow)
            resource._update_mutable_state(datetime_=utcnow)

            # Save resource
            resource.persist()

            # Update direct parent's modification datetime
            if cls.touch_parent():
                resource._touch_direct_parent(notify_to=notify_to)

            # Notify users
            if notify_to:
                resource.notify(NotificationEvent.CREATION, notify_to)

        except Exception as e:
            try:
                resource.delete()
            except Exception:
                pass
            raise e

        return resource

    @_permissions_required(action=ResourceAction.UPDATE)
    def put(self, data: dict, notify_to: Iterable[UserDB] = None):
        """ Updates a resource

        Args:
            data (dict): resource data
            notify_to (Iterable): users to which notify about the resource update
        """
        if not issubclass(self.db_model(), (MutableEntity, OrganizationDB, RoleDB, ClientDB)):
            raise ImmutableResourceError(resource_id=self.public_id())

        # Check sync state
        if issubclass(self.db_model(), MutableEntity):
            if isinstance(self.agent(), UserDB) and self.user().user_id not in self.db_object().synced_by_users:
                raise ResourceOutOfSyncError(resource_id=self.public_id())
            if isinstance(self.agent(), ClientDB) and self.client().client_id not in self.db_object().synced_by_clients:
                raise ResourceOutOfSyncError(resource_id=self.public_id())

        # Set resource data
        self._set_data(data, notify_to)

        # Update mutable state
        self._update_mutable_state(datetime_=datetime.utcnow())

        # Save resource
        self.persist()

        # Update direct parent's modification datetime
        if self.touch_parent():
            self._touch_direct_parent(notify_to=notify_to)

        # Notify users
        if notify_to:
            self.notify(NotificationEvent.UPDATE, notify_to)

    @_permissions_required(action=ResourceAction.DELETE)
    def delete(self, notify_to: Iterable[UserDB] = None):
        """ Deletes the resource

        Args:
            notify_to (Iterable): users to which notify about the deletion of the resource
        """

        self.uncache()
        delete_from_db(self.db_object())

        # Delete resource-level permissions
        if self.permission_resource_type() is not None:
            UserPermission.query().filter_by(resource_uuid=self.uuid()).delete()
            RolePermission.query().filter_by(resource_uuid=self.uuid()).delete()
            db_commit()

        # Update direct parent's modification datetime
        if self.touch_parent():
            self._touch_direct_parent(notify_to=notify_to)

        # Notify users
        if notify_to:
            self.notify(NotificationEvent.DELETION, notify_to)

    def persist(self):
        """
        Saves resource data to the database.
        WARNING: resource data is uncached.
        """
        if self.cached():
            try:
                save_to_db(self.db_object())
            except InvalidRequestError as e:
                err_msg = str(e).lower()
                if ("can't attach instance" in err_msg and 'another instance with key' in err_msg and
                        'is already present in this session' in err_msg):
                    db_commit()
            self.uncache()
        else:
            save_to_db(self.db_object())

    def refresh(self):
        """
        Refreshes resource data with the latest version of the database object.
        WARNING: resource data is uncached.
        """

        def _refresh_db_object():
            if self.cached():
                db.session.refresh(self.db_object())
                self.db_object().force_relationship_loading()
                self.uncache()
            else:
                db.session.expire(self.db_object())

        try:
            _refresh_db_object()
        except InvalidRequestError:
            if inspect(self.db_object()).detached:
                # Cached database object was loaded in another session
                # See https://docs.sqlalchemy.org/en/13/orm/session_state_management.html#unitofwork-merging
                self._db_object = db.session.merge(self.db_object())
                _refresh_db_object()
            pass  # the database object might not have been persisted yet

    def uncache(self):
        """
        Deletes resource data from the cache.
        Note: if the direct parent is touched by the resource, the direct parent's data is also deleted from the cache.
        """
        if not self.cached():
            return
        direct_parent = self.parents()[-1] if self.parents() else None
        # Calls using the public ID or the UUID
        cache.delete_memoized(self._get_db_object, type(self), self.uuid(), direct_parent)
        cache.delete_memoized(self._get_db_object, type(self), self.public_id(), direct_parent)
        # Calls using an ID based on a unique column
        id_col = self.db_model().id_column()
        if id_col is not None:
            resource_id = getattr(self.db_object(), id_col)
            cache.delete_memoized(self._get_db_object, type(self), resource_id, direct_parent)
        # Uncache direct parent's data if it is touched by the resource
        if direct_parent is not None and self.touch_parent():
            direct_parent.uncache()
        # Update cache state flag
        self._cached = False

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=False,
             update_sync_state: bool = True) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps resource data into a JSON object (`serialize=True`) or a dictionary  (`serialize=False`).

        Args:
            serialize (bool): serialize the resource to a JSON object. Set it to `False` when dumping a resource that
                              will later be serialized by its parent
            expand_associations (bool): expand associated resources. By default, only child resources are expanded
                                        (`expand_associations=False`). Note: only those associations providing
                                        a type of `Resource` in `associations()` will be expanded.
                                        Those providing a database table will never be expanded.
            reference_parents (bool): reference parent resources' identifiers
            update_sync_state (bool): mark the resource and its child resources as "synced" for the given user
                                      (only for mutable resources)

        Returns:
            Union[ResourceResponseSchema, dict]: a `ResourceResponseSchema` if `serialize=True`
                                                 or a dictionary  if `serialize=False`
        """

        def _dump_db_collection_resources(db_collection, resource_type) -> list:
            collection_resources = [
                resource_type.get(agent=self.agent(),
                                  db_object_or_id=x,
                                  parents=(self.parents() + [self]),
                                  check_parents=False) for x in db_collection
            ]
            return [
                resource.dump(serialize=False,
                              expand_associations=expand_associations,
                              update_sync_state=update_sync_state) for resource in collection_resources
            ]

        # Refresh and uncache resource data, as lazily loaded database relationships cannot be loaded when the
        # resource database object is detached from current HTTP request's database session. This happens when the
        # resource data was cached in a different HTTP request.
        self.refresh()

        # Get database relationships
        relationship_props = dict(inspect(self.db_model()).relationships.items())
        relationship_types = self.db_model().relationship_types()
        parents = relationship_types[DBRelationshipType.PARENT]
        children = relationship_types[DBRelationshipType.CHILD]
        many_to_many = relationship_types[DBRelationshipType.MANY_TO_MANY]
        assoc_objects = relationship_types[DBRelationshipType.ASSOCIATION_OBJECT]

        # Map from database object to resource representation
        db_object = self.db_object()
        resource_dict = db_object.to_dict()

        # Load parents' identifiers (only if specified)
        if reference_parents:
            for relationship in parents:
                parent = getattr(db_object, relationship)
                if isinstance(parent, Entity):
                    resource_dict[relationship] = getattr(parent, parent.id_column() or 'public_id')

        # Load child collections plus resource associations from database relationships
        for relationship in children + many_to_many + assoc_objects:
            if relationship_props[relationship].lazy == 'dynamic':
                # Note: collections with large number of items are not included in the JSON,
                #       as they are usually paginated in another endpoint
                continue
            # Load related database objects
            if not (relationship in self.collections() or relationship in self.associations()):
                continue
            db_collection = getattr(db_object, relationship)
            if not db_collection:
                resource_dict[relationship] = []
                continue
            # One-to-Many relationships (child resources). Always dump nested resources
            if relationship in children:
                collection_type = self.collections()[relationship]
                resource_dict[relationship] = _dump_db_collection_resources(db_collection=db_collection,
                                                                            resource_type=collection_type)
            # Many-to-Many relationships. Either dump associated resources' whole
            # representation (`expand_associations=True`) or just their ID (`expand_associations=False`)
            elif relationship in many_to_many:
                assoc_type = self.associations()[relationship]
                # If a `Resource` was specified for this relationship, validate its database model
                if issubclass(assoc_type, Resource):
                    assert assoc_type.db_model() == relationship_props[relationship].mapper.class_
                # Expanded mode (dump resources' whole representation)
                if expand_associations and issubclass(assoc_type, Resource):
                    resource_dict[relationship] = _dump_db_collection_resources(db_collection=db_collection,
                                                                                resource_type=assoc_type)
                # Unexpanded mode (dump only IDs)
                else:
                    resource_dict[relationship] = []
                    for related_entity in db_collection:
                        assert isinstance(related_entity, Entity)
                        related_entity_id = getattr(related_entity, related_entity.id_column() or 'public_id')
                        resource_dict[relationship].append(related_entity_id)
            # Association Objects
            else:
                assert relationship in assoc_objects
                resource_dict[relationship] = []
                for assoc_obj in db_collection:
                    assoc_dict = dict()
                    # Get related entities' IDs
                    for assoc_obj_rel in assoc_obj.relationships():
                        related_entity = getattr(assoc_obj, assoc_obj_rel)
                        if not isinstance(related_entity, Entity):
                            continue
                        related_entity_id_col = related_entity.id_column() or 'public_id'
                        assoc_dict[assoc_obj_rel] = getattr(related_entity, related_entity_id_col)
                    # Get additional fields
                    for additional_field in assoc_obj.columns():
                        assoc_dict[additional_field] = getattr(assoc_obj, additional_field)
                    resource_dict[relationship].append(assoc_dict)

        # Dump resource data
        resource_dict['id'] = self.public_id()
        dumped_data = self.dump_data(data=resource_dict, serialize=serialize)

        # Update sync state (only for mutable resources)
        if update_sync_state and isinstance(self.db_object(), MutableEntity):
            sync_state_updated = self.db_object().update_sync_state(agent=self.agent(), commit=False)
            if sync_state_updated:
                self.persist()

        return dumped_data

    @classmethod
    def dump_data(cls, data: dict, serialize=True) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps provided data into a JSON object (`serialize=True`) or a dictionary  (`serialize=False`).

        This function is useful for implementing resource-type-specific dumping logic.

        Args:
            data (dict): data to  be dumped
            serialize (bool): serialize dumped data into a JSON object

        Returns:
            Union[ResourceResponseSchema, dict]: a `ResourceResponseSchema` if `serialize=True`
                                                 or a dictionary  if `serialize=False`
        """
        return cls.dump_schema()().dump(data) if serialize else data

    def notify(self, event: NotificationEvent, recipients: Iterable[UserDB]):
        """
        Notifies an event occurred in the resource.

        TODO: consider removing `recipients` argument and let this function determine the recipients.

        Returns:
            event (int): one of the events specified by `database.models.notifications.NotificationEvent`
            recipients (Iterable): user IDs to be notified
        """

        @retry_on_deadlock
        def _increase_agg_notification_count(task_id: int, recipient: int):
            # Increase the value at SQL-level instead of Python-level to avoid race conditions
            (AggregatedNotificationDB.query().filter_by(task_id=task_id,
                                                        recipient=recipient,
                                                        source_type=self.notification_source_type(),
                                                        event=event).update(
                                                            {'count': AggregatedNotificationDB.count + 1}))

            db_commit()

        if self.notification_source_type() is None:
            return

        if self.db_model() == TaskDB or (self.parents() and self.parents()[0].db_model() == TaskDB):
            task = self if self.db_model() == TaskDB else self.parents()[0]
            task_id = task.db_object().task_id
        else:
            task_id = None

        for recipient in recipients:
            if self.user() is not None and recipient.user_id == self.user().user_id:
                continue
            # If there are too many notifications with the same type of source and event, aggregate them
            agg_notifications = (AggregatedNotificationDB.query().filter_by(task_id=task_id,
                                                                            recipient=recipient.user_id,
                                                                            source_type=self.notification_source_type(),
                                                                            event=event).all())
            assert len(agg_notifications) <= 1  # unique index in (recipient, source_type, event)
            agg_notification = agg_notifications[0] if agg_notifications else None
            # If the notifications have already been aggregated, just increase the count
            if agg_notification is not None:
                _increase_agg_notification_count(task_id=task_id, recipient=recipient.user_id)
            # Otherwise, check the number of notifications with the same type of source and event
            else:
                notifications_group = (NotificationDB.query().filter_by(task_id=task_id,
                                                                        recipient=recipient.user_id,
                                                                        source_type=self.notification_source_type(),
                                                                        event=event).all())
                # If the number of notifications doesn't exceed the limit yet, save the notification individually
                if len(notifications_group) < config.get('notifications')['max_source_events']:
                    notification = NotificationDB(task_id=task_id,
                                                  recipient=recipient.user_id,
                                                  source_type=self.notification_source_type(),
                                                  source_uuid=self.uuid(),
                                                  source_url=self.url(),
                                                  event=event)
                    save_to_db(notification)
                # Otherwise, save a new aggregated notification and delete the individual notifications
                else:
                    since = min(n.created_at for n in notifications_group)  # get the oldest object's creation datetime
                    notification = AggregatedNotificationDB(task_id=task_id,
                                                            recipient=recipient.user_id,
                                                            source_type=self.notification_source_type(),
                                                            event=event,
                                                            since=since,
                                                            count=len(notifications_group) + 1)
                    try:
                        save_to_db(notification)
                        delete_from_db(notifications_group)
                    except IntegrityError:
                        # If some other request aggregated the notifications during this request,
                        # just increase the count
                        db_rollback()
                        _increase_agg_notification_count(task_id=task_id, recipient=recipient.user_id)

    def get_collection(self, collection_name: str, remove_notifications=False) -> list:

        collection_type = self.collections()[collection_name]
        assert issubclass(collection_type, Resource)

        # Convert database objects to resource collections
        db_objects = getattr(self.db_object(), collection_name)
        collection = list()
        for db_object in db_objects:
            resource = collection_type.get(agent=self.agent(),
                                           db_object_or_id=db_object,
                                           parents=(self.parents() + [self]),
                                           check_parents=False,
                                           remove_notifications=remove_notifications)
            collection.append(resource)

        return collection

    def clear_collection(self, collection_name: str, persist: bool = True, notify_to: Iterable[UserDB] = None):

        self._update_collection(collection_name=collection_name,
                                operation=ResourceCollectionOperation.CLEAR,
                                persist=persist,
                                notify_to=notify_to)

    def replace_collection(self,
                           resources: list,
                           collection_name: str,
                           persist: bool = True,
                           notify_to: Iterable[UserDB] = None):

        self._update_collection(collection_name=collection_name,
                                operation=ResourceCollectionOperation.REPLACE,
                                resources=resources,
                                persist=persist,
                                notify_to=notify_to)

    def add_to_collection(self,
                          resources: list,
                          collection_name: str,
                          persist: bool = True,
                          notify_to: Iterable[UserDB] = None):

        self._update_collection(collection_name=collection_name,
                                operation=ResourceCollectionOperation.APPEND,
                                resources=resources,
                                persist=persist,
                                notify_to=notify_to)

    def remove_from_collection(self,
                               resources: list,
                               collection_name: str,
                               persist: bool = True,
                               notify_to: Iterable[UserDB] = None):

        self._update_collection(collection_name=collection_name,
                                operation=ResourceCollectionOperation.REMOVE,
                                resources=resources,
                                persist=persist,
                                notify_to=notify_to)

    @classmethod
    def check_permissions(cls,
                          organization: OrganizationDB,
                          action: ResourceAction,
                          user: UserDB = None,
                          resource=None,
                          check_parents: bool = True):
        """
        Check if the user has enough permissions to perform the action on the given resource or resource type
        (and optionally its parents).

        If the agent performing the action on the resource is a client (not a user), no permissions are checked.
        This function assumes that the token scopes have already been validated.

        If the user doesn't have enough permissions, a `PermissionDeniedError` is raised.

        See `filter_effective_permissions()` for more info on permission precedence.

        WARNING: for the moment, only organization-wide permissions are supported.

        TODO: Double check token scopes at resource level (not only at view level). Use this function for that.

        Args:
            organization (Organization): organization where permissions are being checked
            action (ResourceAction): action to be performed
            user (User): user performing the action.
                         If `resource` is given, the associated user (if any) will be considered.
            resource (Resource): resource on which the action will be performed. It must be an instance of this class
            check_parents (bool): check permissions for accessing parent resources
        """

        # Sanity check
        assert isinstance(resource, cls) or resource is None

        if cls.permission_resource_type() is None:
            return

        if organization is None:
            raise PermissionDeniedError()

        # Get the user accessing the resource
        if resource is not None:
            if user is not None:
                assert resource.user().user_id == user.user_id
            else:
                user = resource.user()

        # If the agent accessing the resource is a client, there are no permissions to check
        if user is None:
            return

        # If the user is an admin or a maintainer, there are no permissions to check
        if any(role.name in [ADMIN_ROLE, MAINTAINER_ROLE] for role in user.roles):
            return

        # Check user's read permissions on parent resources
        # TODO: what if the user doesn't have read permissions on parents but has explicit permissions on the resource?
        if check_parents and resource is not None:
            for parent in resource.parents():
                parent_action = ResourceAction.UPDATE if action != ResourceAction.READ and cls.touch_parent(
                ) else ResourceAction.READ
                parent.check_permissions(organization=organization,
                                         action=parent_action,
                                         resource=parent,
                                         check_parents=False)

        # Get user permissions
        if resource is not None:
            resource_uuid = resource.uuid()
            user_rsrc_uuid_filter = UserPermission.resource_uuid.in_([NULL_UUID, resource_uuid])
            role_rsrc_uuid_filter = RolePermission.resource_uuid.in_([NULL_UUID, resource_uuid])
        else:
            resource_uuid = NULL_UUID
            user_rsrc_uuid_filter = UserPermission.resource_uuid == NULL_UUID
            role_rsrc_uuid_filter = RolePermission.resource_uuid == NULL_UUID

        user_permissions = (UserPermission.query().filter_by(organization_id=organization.organization_id,
                                                             user_id=user.user_id,
                                                             resource_type=cls.permission_resource_type(),
                                                             action=action).filter(user_rsrc_uuid_filter).all())

        # Get role permissions
        role_permissions = (db_query(RolePermission).join(
            user_roles_table, user_roles_table.c.user_id == user.user_id).filter(
                role_rsrc_uuid_filter, RolePermission.resource_type == cls.permission_resource_type(),
                RolePermission.action == action).all())

        # Filter effective permissions
        effective_permissions = filter_effective_permissions(user_permissions=user_permissions,
                                                             role_permissions=role_permissions)

        # Check effective permissions
        allowed = False

        for permission in effective_permissions:
            allowed = permission.allow
            if permission.resource_uuid == resource_uuid:
                break

        if not allowed:
            raise PermissionDeniedError()

    def valid_url(self, url_resource_ids: List[str]) -> bool:
        """ Validates the URL of the request on this resource.
        Args:
            url_resource_ids (List[str]): parent resources' IDs (UUIDs or public IDs)
                                          (in order of appearance in the URL)

        Returns:
            bool: True if the specified parents' IDs match the resource parents' IDs
        """
        if len(url_resource_ids) != len(self.parents()):
            return False
        return all(id_ in [parent.uuid(), parent.public_id()] for id_, parent in zip(url_resource_ids, self.parents()))

    @classmethod
    def _load_db_object(cls, resource_id: str, parent=None, load_relationships=False) -> Entity:
        # Load database object
        parent_db_object = parent.db_object() if parent is not None else None
        db_object = cls.db_model().get_from_id(id_value=resource_id, parent=parent_db_object)
        if db_object is None:
            raise ResourceNotFoundError(resource_id=resource_id)
        # If specified, load non-dynamic relationships eagerly
        if load_relationships:
            db_object.force_relationship_loading()
        return db_object

    @classmethod
    @cache.memoize()
    def _get_db_object(cls, resource_id: str, parent=None) -> Entity:
        """
        WARNING: relationships that are loaded lazily (not eagerly) cannot be loaded from cache,
                 as the cached instance may not be attached to current session.

        NOTE: pass `load_relationships=True` to the commented `database.models.base.DBModel.query()`
              function when it is fixed.
        """
        return cls._load_db_object(resource_id=resource_id, parent=parent, load_relationships=True)

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Sets the resource data with the provided data.

        WARNING: This function replaces all the resource data with the provided data.
                 If an attribute or association (not collection) is not provided, it will be removed from the resource.

        Args:
            data (dict): values for database object's columns and relationships (collections and associations)
            notify_to (Iterable): users to which notify about the update
        """

        def _most_direct_shared_parent(db_model: Type[Entity]) -> Optional[Entity]:

            def _unroll_parents(db_model_: Type[Entity]) -> Set[Type[Entity]]:
                db_model_parent_names = db_model_.relationship_types()[DBRelationshipType.PARENT]
                db_model_parents = [getattr(db_model_, x).mapper.class_ for x in db_model_parent_names]
                unrolled_parents = {x for sublist in [_unroll_parents(p) for p in db_model_parents] for x in sublist}
                return {db_model_}.union(unrolled_parents)

            for parent in reversed(self.parents()):
                shared_parents = [p for p in _unroll_parents(db_model) if parent.db_model() == p]
                if shared_parents:
                    assert len(shared_parents) == 1  # multiple equally direct parents
                    return parent.db_object()

        def _rollback(old_object: Entity):
            db_rollback()
            if old_object is None:
                try:
                    delete_from_db(self.db_object())
                except InvalidRequestError:
                    pass  # the database object might not have been persisted yet
            else:
                db.session.merge(old_object)
                db.session.commit()

        # Save current database object to restore it if something fails
        old_object = copy.deepcopy(self.db_object()) if self.db_object() is not None else None

        # Get info about relationships
        relationship_props = dict(inspect(self.db_model()).relationships.items())
        relationship_types = self.db_model().relationship_types()
        parents = relationship_types[DBRelationshipType.PARENT]
        children = relationship_types[DBRelationshipType.CHILD]
        many_to_many = relationship_types[DBRelationshipType.MANY_TO_MANY]
        assoc_objects = relationship_types[DBRelationshipType.ASSOCIATION_OBJECT]

        # Get provided values for columns
        columns = self.db_model().columns()
        new_object_data = {k: v for k, v in data.items() if k in columns}

        # Keep the values of columns which are not fed by any field defined in the load schema
        if old_object is not None:
            # TODO: Why don't we access `self.fields` instead?
            load_fields = [(f if f not in dir(builtins) else f + '_') for f in self.load_schema()._declared_fields]
            new_object_data.update({k: v for k, v in old_object.to_dict().items() if k not in load_fields})

        # Get the values of parent resources' database object foreign keys
        if self.parents():
            parent_db_object = self.parents()[-1].db_object()
            for pk in parent_db_object.primary_key_columns():
                new_object_data[pk] = getattr(parent_db_object, pk)

        # Get Many-to-One relationships (parent database objects), which must be given by a single identifier
        for relationship in parents:
            rel_db_model = relationship_props[relationship].mapper.class_
            # Check if parent is already set
            if all(pk_col in new_object_data for pk_col in rel_db_model.primary_key_columns()):
                continue
            # Get parent database object
            parent_id = data.get(relationship, '')
            if not isinstance(parent_id, str):
                raise InvalidDataError(f'Invalid ID for related resource "{relationship}"')
            shared_parent = _most_direct_shared_parent(db_model=rel_db_model)
            new_object_data[relationship] = rel_db_model.get_from_id(id_value=parent_id, parent=shared_parent)

        # Create or Update database object
        try:
            # Set column values
            if self.db_object() is None:
                self._db_object = self.db_model()(**new_object_data)
                save_to_db(self.db_object())
            else:
                for column, value in new_object_data.items():
                    setattr(self.db_object(), column, value)

            # Add relationships (collections and associations)
            for relationship in children + many_to_many + assoc_objects:
                # If the relationship is not provided, skip it
                if relationship not in data:
                    if relationship in self.associations():
                        # If the relationship is an association, delete association data from database
                        getattr(self.db_object(), relationship).clear()
                    continue

                # Get relationship's database model
                rel_db_model = relationship_props[relationship].mapper.class_

                # Get provided items. Their content depends on the type of relationship
                # (One-to-Many, Many-to-Many, Association Object)
                items = data[relationship]

                # One-to-Many relationship (child collections). Items must be self-defined resources
                if relationship in children:
                    if relationship not in self.collections():
                        continue
                    if not isinstance(items, Iterable):
                        raise InvalidDataError(f'Invalid collection "{relationship}"')
                    collection_type = self.collections()[relationship]
                    collection = list()
                    for item_data in items:
                        child = collection_type.post(agent=self.agent(),
                                                     data=item_data,
                                                     parents=(self.parents() + [self]),
                                                     check_parents=False,
                                                     notify_to=notify_to)
                        collection.append(child.db_object())
                    setattr(self.db_object(), relationship, collection)

                # Many-to-Many relationship or Association Object. Items can be of two types:
                #     - For Many-to-Many: string with the resource ID (UUID, public ID, or a unique column value)
                #     - For Association Object: dict of {relationship_1: resource_id_1,
                #                                        relationship_N: resource_id_N,
                #                                        field_1: value_1,
                #                                        field_N: value_N}
                else:
                    # Get association type
                    if relationship not in self.associations():
                        continue
                    is_assoc_obj = relationship in assoc_objects
                    association_model = self.associations()[relationship]
                    if is_assoc_obj:
                        assert association_model == rel_db_model
                    else:
                        assert issubclass(association_model, (Entity, Resource))
                    # Load provided items and create/update the corresponding database object
                    associations = []
                    for item in items:
                        # Check item format
                        if is_assoc_obj and isinstance(item, dict) and len(item) >= 2:
                            pass
                        elif not is_assoc_obj and isinstance(item, str):
                            pass
                        else:
                            raise InvalidDataError(f'Invalid resource association "{relationship}"')
                        # Load associated resources
                        if is_assoc_obj:
                            item = dict(item)
                            assoc_id_fields = association_model.relationships()
                            assoc_resource_ids = {field: item.pop(field) for field in assoc_id_fields if field in item}
                            assoc_resource_id = None
                        else:
                            assoc_resource_ids = None
                            assoc_resource_id = item
                        try:
                            # Get associated resources
                            if issubclass(association_model, Resource):
                                # Many-to-Many (resource model specified)
                                assoc_db_model = association_model.db_model()
                                shared_parents = [p for p in self.parents() if p.db_model().is_parent(assoc_db_model)]
                                assoc_resource = association_model.get(agent=self.agent(),
                                                                       db_object_or_id=assoc_resource_id,
                                                                       parents=shared_parents,
                                                                       check_parents=False)
                                assoc_db_object = assoc_resource.db_object()
                            else:
                                # Many-to-Many (resource model not specified)
                                if issubclass(association_model, Entity):
                                    shared_parent = _most_direct_shared_parent(db_model=association_model)
                                    assoc_db_object = association_model.get_from_id(id_value=assoc_resource_id,
                                                                                    parent=shared_parent)
                                # Association Object
                                else:
                                    assoc_pk = dict()
                                    # Get related resources' database objects
                                    related_db_objects = []
                                    for assoc_relationship in inspect(association_model).relationships.values():
                                        if assoc_relationship.key not in assoc_resource_ids:
                                            continue  # related entity's ID not provided
                                        related_model = assoc_relationship.mapper.class_
                                        assert issubclass(related_model, Entity)
                                        related_id = assoc_resource_ids[assoc_relationship.key]
                                        shared_parent = _most_direct_shared_parent(db_model=related_model)
                                        related_object = related_model.get_from_id(id_value=related_id,
                                                                                   parent=shared_parent)
                                        related_db_objects.append(related_object)
                                    # Get related objects' primary keys
                                    for related_object in related_db_objects + [self.db_object()]:
                                        related_object_pk = {
                                            pk_col: getattr(related_object, pk_col)
                                            for pk_col in related_object.primary_key_columns()
                                        }
                                        assoc_pk.update(related_object_pk)
                                    # Get the value of additional primary key columns
                                    additional_pk_cols = association_model.primary_key_columns() - set(assoc_pk.keys())
                                    for pk_col in additional_pk_cols:
                                        if pk_col in item:
                                            assoc_pk[pk_col] = item[pk_col]
                                        else:
                                            # WARNING: if there is no default value for this field, argument
                                            # `required=True` must be passed to corresponding schema's field.
                                            default_value = association_model.__table__.columns[pk_col].default.arg
                                            assert default_value is not None
                                            assoc_pk[pk_col] = default_value
                                    # Get or create Association Object's database object
                                    assoc_db_object = association_model.get(**assoc_pk) or association_model(**assoc_pk)
                                    # Set additional column values
                                    additional_column_values = {c: v for c, v in item.items() if c not in assoc_pk}
                                    for column, value in additional_column_values.items():
                                        setattr(assoc_db_object, column, value)
                        except ResourceNotFoundError:
                            assoc_name = relationship if not relationship.endswith('_') else relationship[:-1]
                            raise ResourceNotFoundError(f'Resource(s) "{assoc_resource_id or assoc_resource_ids}"'
                                                        f'not found in "{assoc_name}"')
                        # Add association's database object to the corresponding relationship collection
                        associations.append(assoc_db_object)
                    setattr(self.db_object(), relationship, associations)
            save_to_db(self.db_object())
        except IntegrityError as e:
            _rollback(old_object=old_object)
            error_code = e.orig.args[0]
            error_msg = e.orig.args[1].lower()
            if error_code == 1062:
                orig_table = f"'{self.db_model().__table__.name}."
                id_column = self.db_model().id_column()
                if orig_table in error_msg and id_column is not None:
                    raise DuplicateResourceError(resource_id=data.get(id_column))
                else:
                    raise DuplicateResourceError()
            else:
                raise InvalidDataError()
        except (DatabaseError, StatementError):
            _rollback(old_object=old_object)
            raise InvalidDataError()  # TODO: try to be more specific
        except Exception as e:
            _rollback(old_object=old_object)
            if '(pymysql.err.IntegrityError) (1062, "Duplicate entry' in str(e):
                # TODO: why is this exception not being raised as an `IntegrityError`?
                raise DuplicateResourceError()
            raise e

    def _set_parents(self, parents: list):
        assert self.agent() is not None

        # Check whether the parents were previously loaded (all or none)
        parents_already_loaded = isinstance(parents[-1], Resource)
        if parents_already_loaded:
            assert all(isinstance(p, Resource) for p in parents)
        else:
            assert all(isinstance(p, tuple) for p in parents)

        # Load parents if they were not previously loaded
        if parents_already_loaded:
            self._parents = parents
        else:
            # TODO: make sure we don't load a resource's database object multiple times
            for p in parents:
                p = p[0].get(agent=self.agent(), db_object_or_id=p[1], parents=list(self._parents), check_parents=False)
                self._parents.append(p)

        # Verify parent-child relationships
        if len(self._parents) > 1:
            try:
                # At resource level
                for prev_parent, parent in zip(self._parents[:-1], self._parents[1:]):
                    assert prev_parent == parent._parents[-1]
                    assert prev_parent._parents == parent._parents[:-1]
                # At database level
                direct_parent = self._parents[-1]
                parent_db_objects = [p.db_object() for p in direct_parent._parents]
                parent_objects, children_objects = (parent_db_objects,
                                                    parent_db_objects[1:] + [direct_parent.db_object()])
                for parent, child in zip(parent_objects, children_objects):
                    assert parent.is_parent(child)
                    assert all(
                        getattr(parent, c, None) == getattr(child, c, None) for c in parent.primary_key_columns())
            except AssertionError:
                raise ResourceNotFoundError()

    def _check_permissions(self, action: ResourceAction, check_parents: bool = True):
        # Get the organization to which the resource belongs
        organization = None
        for resource in [self] + self.parents():
            if isinstance(resource.db_object(), OrganizationDB):
                organization = resource.db_object()
                break
            resource_org = getattr(resource.db_object(), 'organization', None)
            if isinstance(resource_org, OrganizationDB):
                organization = resource_org
                break

        # Check permissions
        self.check_permissions(organization=organization, action=action, resource=self, check_parents=False)

        # Check permissions on parents
        if check_parents and self.parents():
            direct_parent = self.parents()[-1]
            # If it's not a read operation and the parent must be touched, parents require update permissions
            if action != ResourceAction.READ and self.touch_parent():
                # Check update permissions on parents (only if `touch_parent()` is `True`)
                direct_parent.check_permissions(organization=organization,
                                                action=ResourceAction.UPDATE,
                                                resource=direct_parent)
            # Otherwise, parents require only read permissions
            else:
                direct_parent.check_permissions(organization=organization,
                                                action=ResourceAction.READ,
                                                resource=direct_parent)

    def _set_immutable_state(self, datetime_: datetime):
        if not issubclass(self.db_model(), (ImmutableEntity, MutableEntity)):
            return
        assert isinstance(self.db_object(), (ImmutableEntity, MutableEntity))
        self.db_object().created_at = datetime_
        self.db_object().created_by_user = self.user().user_id if isinstance(self.agent(), UserDB) else None
        self.db_object().created_by_client = self.client().client_id if isinstance(self.agent(), ClientDB) else None

    def _update_mutable_state(self, datetime_: datetime):
        if not issubclass(self.db_model(), MutableEntity):
            return
        assert isinstance(self.db_object(), MutableEntity)
        # Modification fields
        self.db_object().modified_at = datetime_
        self.db_object().modified_by_user = self.user().user_id if isinstance(self.agent(), UserDB) else None
        self.db_object().modified_by_client = self.client().client_id if isinstance(self.agent(), ClientDB) else None
        # Sync state
        self.db_object().synced_by_users = [self.user().user_id] if isinstance(self.agent(), UserDB) else []
        self.db_object().synced_by_clients = [self.client().client_id] if isinstance(self.agent(), ClientDB) else []

    def _touch_direct_parent(self, notify_to: Iterable[UserDB] = None):
        """ Updates direct parent's modification datetime. """
        if not self.parents():
            return
        direct_parent = self.parents()[-1]
        direct_parent.refresh()
        if not isinstance(direct_parent.db_object(), MutableEntity):
            return
        direct_parent._update_mutable_state(datetime.utcnow())
        direct_parent.persist()
        if notify_to:
            direct_parent.notify(NotificationEvent.UPDATE, notify_to)

    def _update_collection(self,
                           collection_name: str,
                           operation: ResourceCollectionOperation,
                           resources: List = None,
                           persist: bool = True,
                           notify_to: Iterable[UserDB] = None):

        # Sanity check
        if operation == ResourceCollectionOperation.CLEAR:
            assert not resources
        else:
            assert resources is not None

        # Get collection type
        collection_type = self.collections()[collection_name]
        assert issubclass(collection_type, Resource)

        # Check permissions
        if collection_type.touch_parent():
            self._check_permissions(action=ResourceAction.UPDATE)

        # Get collection
        collection = getattr(self.db_object(), collection_name)

        # Check resource types
        if resources:
            assert all(isinstance(resource, collection_type) for resource in resources)
            if collection:
                assert all(isinstance(resource.db_object(), type(collection[0])) for resource in resources)

        # Perform operation
        if operation == ResourceCollectionOperation.APPEND:
            for resource in resources:
                collection.append(resource.db_object())
        elif operation == ResourceCollectionOperation.REMOVE:
            for resource in resources:
                collection.remove(resource.db_object())
        elif operation == ResourceCollectionOperation.REPLACE:
            setattr(self.db_object(), collection_name, [resource.db_object() for resource in resources])
        else:
            collection.clear()

        # Update mutable state
        if collection_type.touch_parent():
            self._update_mutable_state(datetime_=datetime.utcnow())

        # Persist resource
        if persist:
            self.persist()

        # Notify
        if notify_to:
            self.notify(NotificationEvent.UPDATE, notify_to)


#############
# Functions #
#############


def dump(resources: Iterable[Resource],
         serialize=True,
         expand_associations=False,
         reference_parents=False,
         update_sync_state: bool = True) -> List[dict]:

    return [
        resource.dump(serialize=serialize,
                      expand_associations=expand_associations,
                      reference_parents=reference_parents,
                      update_sync_state=update_sync_state) for resource in resources
    ]


def filter_effective_permissions(user_permissions: List[Permission] = None,
                                 role_permissions: List[Permission] = None) -> List[Permission]:
    """
    Filters effective permissions.

    Permission precedence:
        1) User vs. Role: user takes precedence over role
        2) Generic vs. Resource-level: resource-level (that assigned to a specific resource)
                                       takes precedence over generic (that assigned to a certain type of resource)
        3) Granted vs. Denied: denied permission takes precedence over granted permission

    Args:
        user_permissions (list): permissions assigned to the user directly
        role_permissions (list): permissions inherited from user roles

    Returns:
        list: effective permissions
    """
    if not user_permissions and not role_permissions:
        return []

    user_permissions = user_permissions or []
    role_permissions = role_permissions or []

    gen_scope = lambda x: (x.resource_type, x.action)
    rsrc_scope = lambda x: (x.resource_type, x.resource_uuid, x.action)

    effective_permissions = []

    # 1) User resource-level permissions
    user_rsrc_perms = [x for x in user_permissions if x.resource_uuid != NULL_UUID]
    user_rsrc_scopes = [rsrc_scope(x) for x in user_rsrc_perms]
    effective_permissions += user_rsrc_perms

    # 2) User generic permissions
    user_gen_perms = [x for x in user_permissions if x.resource_uuid == NULL_UUID]
    user_gen_scopes = [gen_scope(x) for x in user_gen_perms]
    effective_permissions += user_gen_perms

    # 3) Role resource-level permissions
    role_rsrc_perms = [x for x in role_permissions if x.resource_uuid != NULL_UUID]
    effective_permissions += [
        x for x in role_rsrc_perms if rsrc_scope(x) not in user_rsrc_scopes and gen_scope(x) not in user_gen_scopes
    ]

    # 4) Role generic permissions
    role_gen_perms = [x for x in role_permissions if x.resource_uuid == NULL_UUID]
    effective_permissions += [x for x in role_gen_perms if gen_scope(x) not in user_gen_scopes]

    # Check if there are conflicts between granted and denied permissions (should only happen in inherited permissions)
    perms_by_scope = dict()

    for perm in set(effective_permissions):
        perm_scope = rsrc_scope(perm)
        if perm_scope not in perms_by_scope:
            perms_by_scope[perm_scope] = []
        perms_by_scope[perm_scope].append(perm)

    if any(len(scope_perms) > 1 for scope_perms in perms_by_scope.values()):

        effective_permissions = []

        for scope_perms in perms_by_scope.values():
            # Case 1: single permission (no conflicts)
            if len(scope_perms) == 1:
                effective_permissions += scope_perms
                continue  # no conflicts
            # Case 2: multiple permissions (possible conflicts). Denied takes precedence over granted
            denied_perms = [x for x in scope_perms if not x.allow]
            effective_permissions += denied_perms if denied_perms else scope_perms

    return list(set(effective_permissions))


def users_permissions(organization: OrganizationDB,
                      user_ids: Iterable[int] = None,
                      resource_type: ResourceType = None,
                      resource_uuid: str = None,
                      action: ResourceAction = None,
                      allow: bool = None,
                      inheritance: bool = True,
                      only_effective: bool = True) -> Dict[UserDB, List[Permission]]:
    """
    Returns the permissions assigned to each user of the organization.

    Args:
        organization (OrganizationDB): organization where the permission has effect
        user_ids (list): database surrogate key of the users
        resource_type (ResourceType): resource type to which the permission refers
        resource_uuid (str): UUID of the resource (only for resource-level permissions)
        action (ResourceAction): action to which the permission refers
        allow (bool): return only granted/denied permissions. If not provided, both types will be considered
        inheritance (bool): include permissions inherited from roles
        only_effective (bool): return only effective permissions.
                               See `filter_effective_permissions()` for more info on permission precedence.

    Returns:
        dict: a dictionary of (<User>: <list_of_permissions>)
    """

    def _filters(db_model: Type[Union[UserPermission, RolePermission]]) -> list:
        filters_ = []

        if resource_type is not None:
            filters_.append(db_model.resource_type == resource_type)

        if resource_uuid:
            filters_.append(db_model.resource_uuid.in_([resource_uuid, NULL_UUID]))

        if action is not None:
            filters_.append(db_model.action == action)

        return filters_

    perms_by_user = dict()

    # Prepare filters
    user_perms_filters = ([UserPermission.organization_id == organization.organization_id] +
                          _filters(db_model=UserPermission))

    if user_ids:
        user_perms_filters.append(UserPermission.user_id.in_(user_ids))

    role_perms_filters = _filters(db_model=RolePermission)

    # Get user permissions in their own organizations
    user_perms = dict()
    for user_perm in UserPermission.query().filter(sql_and(*user_perms_filters)).all():
        perm = Permission(resource_uuid=user_perm.resource_uuid,
                          resource_type=user_perm.resource_type,
                          action=user_perm.action,
                          allow=user_perm.allow)
        if user_perm.user_id not in user_perms:
            user_perms[user_perm.user_id] = []
        user_perms[user_perm.user_id].append(perm)

    # Get role permissions
    role_perms = dict()
    if inheritance:
        user_role_perms = (db_query(user_roles_table, RolePermission).join(
            RoleDB,
            RoleDB.role_id == RolePermission.role_id).filter(user_roles_table.c.role_id == RoleDB.role_id).filter(
                user_roles_table.c.role_id == RolePermission.role_id).filter(
                    RoleDB.organization_id == organization.organization_id).filter(sql_and(*role_perms_filters)))
        if user_ids:
            user_role_perms = user_role_perms.filter(user_roles_table.c.user_id.in_(user_ids))
        for row in user_role_perms.all():
            user_id = row.user_id
            role_perm = row.RolePermission
            perm = Permission(resource_uuid=role_perm.resource_uuid,
                              resource_type=role_perm.resource_type,
                              action=role_perm.action,
                              allow=role_perm.allow)
            if user_id not in role_perms:
                role_perms[user_id] = []
            role_perms[user_id].append(perm)

    # Join user and role permissions
    for user_id in set(user_perms.keys()).union(set(role_perms.keys())):
        user_perms_ = user_perms.get(user_id, [])
        role_perms_ = role_perms.get(user_id, [])
        if only_effective:
            perms_by_user[user_id] = filter_effective_permissions(user_permissions=user_perms_,
                                                                  role_permissions=role_perms_)
        else:
            perms_by_user[user_id] = user_perms_ + role_perms_

    # Include admin/maintainer permissions (immutable, all permissions granted)
    if inheritance and allow is not False:
        admins_maintainers = (db_query(user_roles_table).join(
            RoleDB, RoleDB.role_id == user_roles_table.c.role_id).filter(
                RoleDB.organization_id == organization.organization_id).filter(
                    RoleDB.name.in_([ADMIN_ROLE, MAINTAINER_ROLE])))
        if user_ids:
            admins_maintainers = admins_maintainers.filter(user_roles_table.c.user_id.in_(user_ids))
        for user_role in admins_maintainers.all():
            if resource_type is None and action is None:
                perms_by_user[user_role.user_id] = [
                    Permission(resource_uuid=NULL_UUID, resource_type=resource_type, action=action, allow=True)
                    for resource_type in ResourceType
                    for action in ResourceAction
                ]
            elif resource_type is None:
                perms_by_user[user_role.user_id] = [
                    Permission(resource_uuid=NULL_UUID, resource_type=resource_type, action=action, allow=True)
                    for resource_type in ResourceType
                ]
            else:
                perms_by_user[user_role.user_id] = [
                    Permission(resource_uuid=NULL_UUID, resource_type=resource_type, action=action, allow=True)
                    for action in ResourceAction
                ]

    # Filter granted/denied permissions
    # WARNING: don't do it before, as it may interfere in effective permission filtering
    if allow is not None:
        perms_by_user = {user_id: [x for x in perms if x.allow == allow] for user_id, perms in perms_by_user.items()}

    # Load users
    if user_ids:
        assert set(perms_by_user.keys()).issubset(set(user_ids))

    perms_by_user = {UserDB.get(user_id=user_id): perms for user_id, perms in perms_by_user.items()}

    return perms_by_user


def collaborators_permissions(user_ids: Iterable[int] = None,
                              resource_type: ResourceType = None,
                              resource_uuid: str = None,
                              action: ResourceAction = None,
                              allow: bool = None,
                              only_effective: bool = True) -> Dict[UserDB, Dict[OrganizationDB, List[Permission]]]:
    """
    Returns the permissions assigned to collaborators.

    Args:
        user_ids (list): database surrogate key of the users
        resource_type (ResourceType): resource type to which the permission refers
        resource_uuid (str): UUID of the resource (only for resource-level permissions)
        action (ResourceAction): action to which the permission refers
        allow (bool): return only granted/denied permissions. If not provided, both types will be considered
        only_effective (bool): return only effective permissions.
                               See `filter_effective_permissions()` for more info on permission precedence.

    Returns:
        dict: a dictionary with the following format:
              {
                <user_1>: {
                    <organization_1>: [<list_of_permissions>],
                    ...
                    <organization_N>: [<list_of_permissions>]
                },
                ...
                <user_N>: {...}
              }
    """

    def _filter_org_perms(users_perms: dict, filter_: Callable) -> dict:
        return {
            user: {
                org: filter_(perms) for org, perms in org_perms.items()
            } for user, org_perms in users_perms.items()
        }

    # Prepare filters
    filters = [UserDB.user_id == UserPermission.user_id, UserDB.organization_id != UserPermission.organization_id]

    if user_ids:
        filters.append(UserPermission.user_id.in_(user_ids))

    if resource_type is not None:
        filters.append(UserPermission.resource_type == resource_type)

    if resource_uuid:
        filters.append(UserPermission.resource_uuid == resource_uuid)

    if action is not None:
        filters.append(UserPermission.action == action)

    # Get permissions
    users_perms = dict()
    for user_perm in db_query(UserPermission).filter(sql_and(*filters)).all():
        user_id = user_perm.user_id
        org_id = user_perm.organization_id
        perm = Permission(resource_uuid=user_perm.resource_uuid,
                          resource_type=user_perm.resource_type,
                          action=user_perm.action,
                          allow=user_perm.allow)
        if user_id not in users_perms:
            users_perms[user_id] = dict()
        user_perms = users_perms[user_id]
        if org_id not in user_perms:
            user_perms[org_id] = []
        user_perms[org_id].append(perm)

    # Filter effective permissions
    if only_effective:
        filter_ = lambda perms: filter_effective_permissions(user_permissions=perms)
        users_perms = _filter_org_perms(users_perms=users_perms, filter_=filter_)

    # Filter granted/denied permissions
    # WARNING: don't do it before, as it may interfere in effective permission filtering
    if allow is not None:
        filter_ = (lambda perms: [x for x in perms if x.allow == allow])
        users_perms = _filter_org_perms(users_perms=users_perms, filter_=filter_)

    # Load users and organizations
    if user_ids:
        assert set(users_perms.keys()).issubset(set(user_ids))

    orgs = dict()

    for user_id, orgs_perms in dict(users_perms).items():
        users_perms.pop(user_id)
        user = UserDB.get(user_id=user_id)
        for org_id, org_perms in dict(orgs_perms).items():
            if org_id not in orgs:
                orgs[org_id] = OrganizationDB.get(organization_id=org_id)
            orgs_perms.pop(org_id)
            orgs_perms[orgs[org_id]] = org_perms
        users_perms[user] = orgs_perms

    return users_perms
