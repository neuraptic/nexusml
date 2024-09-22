# Concepts

NexusML is built upon the concepts of *tasks* and *organizations*. This document provides an overview of these concepts 
and their relationships within the platform.

## Tasks

A task in NexusML represents a classification or regression task that the user wants to perform with an AI model.

### Schema

The task schema specifies all the elements that compose the task. There are three types of elements:

- **Inputs**: Elements fed into the AI model to make predictions. Supported types:
    - Image files
    - Document files
    - Audio files
    - Texts
    - Numeric values
- **Outputs**: Elements predicted by the AI model. Supported types:
    - Class labels: Name of the classes the data point belongs to.
    - Numeric values: Continuous values derived from the inputs.
- **Metadata**: Elements storing additional information associated with the data point. They have no effect on 
  model predictions. They are useful for storing information that may be relevant for data management and tracking.

There is no restriction on the number of inputs, outputs, and metadata elements in the task schema, allowing for 
complex task definitions.

### AI Models

An AI model in NexusML is an ML model trained to perform a specific task. There might be multiple versions, each 
representing a snapshot of the AI model at a specific point in time. It is important to note that a model version 
will only be compatible with the task schema it was trained on.

### Examples

An example represents a data point used for training the AI model. It contains the values taken by the data point for 
all the inputs, outputs, and metadata defined in the task schema.

### Services

A service in NexusML is an abstraction of a job that covers a specific stage of the ML lifecycle within the task.

- *Inference Service*: Deploys and serves AI models in production environments.
- *Testing Service*: Deploys and serves AI models in testing environments. It is useful to evaluate the performance of 
  AI models before deploying them in production.
- *Continual Learning Service*: Updates and retrains AI models with new data.
- *Active Learning Service*: Proactively selects the most informative data samples encountered in production for 
  human labeling.
- *Monitoring Service*: Detects anomalies in the AI model behavior.

### Prediction Logs

Prediction logs are records of the predictions made by the Inference Service and the Testing Service.

### Files

A file uploaded to the platform that may be referenced within the task.

## Organizations

Before using NexusML, users must create or join an organization. An organization is a container for tasks, users, 
roles, collaborators, and clients (apps accessing the API). It provides a centralized way to manage and organize all 
the resources within the platform.

An organization may manage multiple tasks, each representing a different predictive task that the organization is 
working on. It is important to note that a task belongs to a single organization, meaning that users from different 
organizations cannot access the task, unless they are invited as collaborators.

### Users

A user is an individual who has access to the organization. Users can have different roles and permissions within the 
organization.

### Roles

A role represents a group of permissions that can be assigned to users within the organization.

### Collaborators

A collaborator is a user from another organization who has been invited to access a specific task owned by the 
organization. A collaborator may have different permissions depending on the task.

### Clients

Clients are external applications that consume the API. Each client is identified by an API key, which is assigned a 
specific set of permissions.

### Files

A file uploaded to the platform that may be referenced within the organization.

### Subscriptions

A subscription is a plan that defines the features and limits available to the organization. An organization can only 
have one active subscription at a time.
