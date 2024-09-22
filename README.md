# NexusML

NexusML is a multimodal AutoML platform for classification and regression tasks.

We'll be releasing the first version of NexusML very soon. In the meantime, you can explore the 
[release/0.1.0](https://github.com/neuraptic/nexusml/tree/release/0.1.0) branch.

Please refer to [docs/what-is-nexusml.md](docs/what-is-nexusml.md) and [docs/concepts.md](docs/concepts.md) for an 
overview of NexusML and its key features and concepts.

## Requirements

- Python 3.10
- AWS setup (S3 bucket and credentials) if you want to use S3 as the file storage backend.

## Pending Refactor Note

The engine was originally designed as a standalone RESTful API, operating on a separate infrastructure from the main 
API. As a result, interactions between the engine and the main API relied heavily on JSON objects (Python dictionaries).

We are planning a comprehensive refactor to allow the engine to interact directly with database models. This change 
will streamline and simplify the integration between the engine and the main API.

## Additional Documentation

The [docs](docs) directory contains additional documentation:
- [concepts.md](docs/concepts.md): Describes the concepts used in NexusML.
- [quickstart.md](docs/quickstart.md): Provides a quick start guide for NexusML.
- [states-and-statuses.md](docs/states-and-statuses.md): Describes NexusML's states and statuses.
- [what-is-nexusml.md](docs/what-is-nexusml.md): Provides an overview of NexusML.
