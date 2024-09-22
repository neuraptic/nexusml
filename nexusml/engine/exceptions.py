class SchemaError(ValueError):
    """
    Error that will be raised when an error is found analyzing the schema
    """


class DataError(ValueError):
    """
    Error that will be raised when an error is found managing data
    """


class ExperimentError(ValueError):
    """
    Error that will be raised when an error is found on the experiment environment
    """


class ConfigFileError(ValueError):
    """
    Error that will be raised when an error is found on the config file
    """
