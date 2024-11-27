from abc import ABC
from abc import abstractmethod
from typing import Dict

from matplotlib import pyplot as plt

from nexusml.engine.experiments.utils import flat_dict


class ExperimentLogger(ABC):
    """
    General class that it could be useful for making the log on an experiment with different frameworks
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    @abstractmethod
    def log_param(self, key, value):
        """
        Method to log a param on the created run
        Args:
            key: the key (name) of the param
            value: the value of the param

        Returns:

        """

    @abstractmethod
    def log_metric(self, key, value):
        """
        Method to log a metric on the created run
        Args:
            key: the key (name) of the metric
            value: the value of the metric

        Returns:

        """

    @abstractmethod
    def log_figure(self, key, value):
        """
        Method to log a figure on the created run
        Args:
            key: the key (name) of the figure
            value: the value of the figure

        Returns:

        """

    @abstractmethod
    def log_artifact(self, artifact_path: str):
        """
        Method to log a set of artifacts on the created run
        Args:
            artifact_path: path where all artifacts are stored

        Returns:

        """

    def log_params(self, param_dict: Dict, flat_params: bool = True):
        """
        Method that logs a set of params
        Args:
            param_dict (Dict): dict with the params to be logged
            flat_params (bool): if True, all parameter values that are dict are flattened.
                                For example, the param: 'key_param': {'inner_dict_param': inner_dict_value}
                                will be modified to {'key_param.inner_dict_param': innser_dict_value}

        Returns:

        """
        # If flat_params is True, flat all dict parameters
        if flat_params:
            param_dict = flat_dict(d=param_dict)
        # For each element in the dictionary, call to log_param
        for k, v in param_dict.items():
            self.log_param(key=k, value=v)

    def log_metrics(self, metrics_dict: Dict):
        """
        Method that logs a set of metrics
        Args:
            metrics_dict (Dict): dict with the metrics to be logged

        Returns:

        """
        # For each element in the metrics dictionary, call to log_metric
        for k, v in metrics_dict.items():
            self.log_metric(k, v)

    def log_figures(self, figures_dict: Dict, close_after_log: bool = True):
        """
        Method that logs a set of figures
        Args:
            figures_dict (Dict): dict with the figures to be logged
            close_after_log (bool): if True, the figures will be closed after the logging for efficiency proposes

        Returns:

        """
        # For each figure in the dict
        for k, v in figures_dict.items():
            # Call to log figure
            self.log_figure(k, v)
            # Close figure if 'close_after_log' is True
            if close_after_log:
                plt.close(v)

    @abstractmethod
    def end_run(self):
        """
        End the current run so the MlFlow state is changed to "FINISHED"
        Returns:

        """
