# Architecture of NexusML

NexusML is composed of three main componentes:

- **Engine**: The engine is the core component of NexusML. It is responsible for training and serving AI models and 
  running all the services involved in the ML lifecycle (inference, continual learning, active learning, and 
  monitoring). Two concepts are central to the engine:
  - **Workers**: The workers are responsible for running the engine, either locally, in the cloud, or on the edge. The 
    flexible deployment of the engine allows users to run their AI models in the most suitable environment for 
    their needs. **Note**: While the engine is designed to support multiple types of workers (local, cloud, and edge), 
    the current version of NexusML only implements local workers. Feel free to contribute to the project by 
    implementing cloud or edge workers (the `EngineWorker` class provides the abstract interface for all types of 
    workers).
  - **Services**: A service in NexusML is an abstraction of a job that covers a specific stage of the ML lifecycle 
    within a task.
    - **Inference Service**: Deploys and serves AI models in production environments.
    - **Testing Service**: Deploys and serves AI models in testing environments. It is useful to evaluate the performance of 
      AI models before deploying them in production.
    - **Continual Learning Service**: Updates and retrains AI models with new data.
    - **Active Learning Service**: Proactively selects the most informative data samples encountered in production for 
      human labeling.
    - **Monitoring Service**: Detects anomalies in the AI model behavior.
- **API Server**: The API server provides a RESTful API for interacting with the engine through HTTP requests. It 
  also manages all the resources related to organizations (users, roles, collaborators, apps, permissions, and 
  subscriptions) and tasks (inputs/outputs, data, metadata, and AI models).
- **File Storage Backend**: The file storage backend is responsible for storing all the files related to the 
  organizations and tasks. The current version of NexusML supports two types of file storage backends:
  - **Local**: Stores files in the local file system.
  - **AWS S3**: Stores files in an [AWS S3](https://aws.amazon.com/s3/) bucket.
