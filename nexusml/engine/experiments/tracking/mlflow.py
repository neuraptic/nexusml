import os
import re
from typing import Any, Tuple

from mlflow.tracking.client import MlflowClient

from nexusml.engine.exceptions import ExperimentError
from nexusml.engine.experiments.tracking.base import ExperimentLogger
from nexusml.engine.experiments.utils import get_metric_name_and_comparator
from nexusml.engine.schema.base import Schema
from nexusml.enums import MLProblemType


class MlFlowExperimentLogger(ExperimentLogger):
    """
    Experiment logger using MlFlow library
    """

    def __init__(self, mlflow_uri: str, experiment_name: str):
        """
        Default constructor
        Args:
            mlflow_uri: Address of local or remote tracking server.
            experiment_name: The experiment name. If exists, log experiment as a run of that experiment.
                            If it does not exist, creates a new experiment with this name
        """
        super().__init__(experiment_name=experiment_name)
        # Store params
        self.mlflow_uri = mlflow_uri
        # Create a MlFlow Client
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        # Try to get or create the experiment, and get its ID
        try:
            self.exp_id = MlFlowExperimentLogger._get_or_create_experiment(mlflow_client=self.client,
                                                                           name=experiment_name)
        except Exception:
            # It could be that two processes running in parallel tries to create the same experiment and only one
            # of them success having an exception "Exception: Yaml file '' exists as ''" by the other
            # So, try to get/create again
            self.exp_id = MlFlowExperimentLogger._get_or_create_experiment(mlflow_client=self.client,
                                                                           name=experiment_name)
        # Create a new run
        self.run = self.client.create_run(experiment_id=self.exp_id)

    @staticmethod
    def _clean_key(key: str) -> str:
        """
        Removes special characters from the key for avoiding MlFlow errors
        Args:
            key (str): original key that can contain special characters

        Returns:
            str: the new key name without special characters
        """
        # Get all characters that are not alphanumeric, underscore (_), point (.), dash (-) and blank space
        regexp = re.compile('[^a-zA-Z\d_.\- ]')
        # Remove all
        return regexp.sub('', key)

    def log_param(self, key: str, value: Any):
        """
        Method to log a param on the created run
        Args:
            key: the key (name) of the param
            value: the value of the param

        Returns:

        """
        # Log the param under the created run
        # First, clean the key
        key = MlFlowExperimentLogger._clean_key(key=key)
        self.client.log_param(run_id=self.run.info.run_id, key=key, value=value)

    def log_metric(self, key: str, value: float):
        """
        Method to log a metric on the created run
        Args:
            key: the key (name) of the metric
            value: the value of the metric

        Returns:

        """
        # Log the metric under the created run
        # First, clean the key
        key = MlFlowExperimentLogger._clean_key(key=key)
        self.client.log_metric(run_id=self.run.info.run_id, key=key, value=value)

    def log_figure(self, key: str, value: Any):
        """
        Method to log a figure on the created run
        Args:
            key: the key (name) of the figure
            value: the value of the figure

        Returns:

        """
        # Log the figure under the created run as PNG file
        # First, clean the key
        key = MlFlowExperimentLogger._clean_key(key=key)
        self.client.log_figure(run_id=self.run.info.run_id, figure=value, artifact_file=f'{key}.png')

    def log_artifact(self, artifact_path: str):
        """
        Method to log a set of artifacts on the created run
        Args:
            artifact_path: path where all artifacts are stored

        Returns:

        """
        # Log the artifacts under the created run
        self.client.log_artifact(run_id=self.run.info.run_id, local_path=artifact_path)

    def end_run(self):
        """
        End the current run so the MlFlow state is changed to "FINISHED"
        Returns:

        """
        self.client.set_terminated(run_id=self.run.info.run_id)

    @staticmethod
    def _get_or_create_experiment(mlflow_client: MlflowClient, name: str):
        """
        Static method that get the ID of the given experiment name, or creates a new one if the experiment doesn't exist
        Args:
            mlflow_client: (MlflowClient): MlFlow client for getting the experiment list
            name (str): the experiment name to search

        Returns:
            Experiment ID of the given experiment name if exists or ID of the created experiment if it doesn't exist
        """
        exp = MlFlowExperimentLogger._get_experiment(mlflow_client=mlflow_client, name=name)
        if exp is None:
            exp_id = mlflow_client.create_experiment(name=name)
        else:
            exp_id = exp.experiment_id
        return exp_id

    @staticmethod
    def _get_experiment(mlflow_client: MlflowClient, name: str):
        """
        Function that returns the Experiment object given the name if exist. If not exist, None will be returned
        Args:
            mlflow_client (MlflowClient): MlFlow client for getting the experiment list
            name (str): the experiment name to search

        Returns:
            Experiment with the given name if exists, or None if it does not exist
        """
        # Search for the experiment with given name
        experiments = mlflow_client.search_experiments(filter_string=f'name = "{name}"')
        # If found, return it
        if len(experiments) != 0:
            return experiments[0]
        else:
            # If not found, return None
            return None


def get_exp_model(mlflow_uri: str, experiment_name: str) -> str:
    """
    Function that get the model of an experiment.
    In some cases, just a single run is done on an experiment, generating a single mlflow
    run. This function gets that run and returns the path to the model of that run.
    If the experiment has more than one run an exception will be raised
    Args:
        mlflow_uri (str): mlflow uri where experiment is stored
        experiment_name (str): name of the experiment

    Returns:
        str indicating the path to the model
    """
    # Get MlFlow client and get the runs of the given experiment
    client = MlflowClient(tracking_uri=mlflow_uri)
    exp = client.get_experiment_by_name(name=experiment_name)
    runs = client.search_runs(experiment_ids=exp.experiment_id)
    if len(runs) != 1:
        raise ExperimentError('Expected a single run experiment')
    return os.path.join(runs[0].info.artifact_uri, 'model.pkl')


def get_best_model(mlflow_uri: str, experiment_name: str) -> Tuple[str, str]:
    """
    Function that is used for getting the best model of a given experiment
    This function will check the metrics of all outputs, getting the model
    that has more outputs with the best metric values
    Args:
        mlflow_uri (str): mlflow uri where experiment is stored
        experiment_name (str): name of the experiment

    Returns:
        Tuple[str, str] with:
         - path to the best model ('model.pkl' artifact of the best run)
         - path to the best model config file ('config.yaml' artifact of the best run)
    """
    # Get MlFlow client and get the runs of the given experiment
    client = MlflowClient(tracking_uri=mlflow_uri)
    exp = client.get_experiment_by_name(name=experiment_name)
    runs = client.search_runs(experiment_ids=exp.experiment_id)
    # To store the performance of each output for each run
    performance_by_run = {}
    # Unify the output list to ensure that all runs worked with the same outputs
    output_list = None
    # For each run
    for i in runs:
        # Get run info
        run_i = client.get_run(run_id=i.info.run_id)
        # Get the schema path
        schema_file = os.path.join(i.info.artifact_uri, 'schema.json')
        if schema_file.startswith('file:///'):
            schema_file = schema_file.replace('file:///', '')
        if not os.path.isfile(schema_file):
            print(f'Not schema file found for run {i.info.run_id}')
            continue

        # To store the performance of each output of the current run
        performance_by_run[i.info.run_id] = {}
        # Create schema
        schema = Schema.create_schema_from_json(json_file=schema_file)
        # Initialize the output list and ensure that is the same always
        if output_list is None:
            output_list = schema.required_outputs()
        else:
            if not all(list(map(lambda x: x in output_list, schema.required_outputs()))):
                raise ExperimentError('Different output elements for the experiments runs')

        # For each output
        for o in output_list:
            # Get problem type
            problem_type_param = f'output_{o["name"]}_problem_type'
            problem_type_param = run_i.data.params[problem_type_param]
            if len(problem_type_param.split('.')) != 2 or problem_type_param.split('.')[0] != 'MLProblemType':
                raise ExperimentError('Parameter "MLProblemType" is not correctly stored')
            problem_type_param = problem_type_param.split('.')[1]
            problem_type_param = MLProblemType[problem_type_param.upper()]
            # Get metric name and comparator
            metric_name, comparator = get_metric_name_and_comparator(problem_type=problem_type_param)
            # Get test metric for the current output
            metric_name = f'test_{o["name"]}_{metric_name}'
            metric_value = run_i.data.metrics[metric_name]
            # Store the metric value and the comparator for the current output for the current run
            performance_by_run[i.info.run_id][o['name']] = (metric_value, comparator)

    # Count votes for each run
    votes_by_run = {}
    # For each output
    for o in output_list:
        # Get the run that have the best metric for this output
        best_value = None
        best_id = None
        for k, v in performance_by_run.items():
            if best_value is None:
                best_value = v[o['name']][0]
                best_id = k
            else:
                if v[o['name']][1](v[o['name']][0], best_value):
                    best_value = v[o['name']][0]
                    best_id = k

        # Add a vote for the run with the best metric
        if best_id not in votes_by_run:
            votes_by_run[best_id] = 0
        votes_by_run[best_id] += 1

    # Get the run that has more votes
    best_run = None
    best_run_votes = None
    for k, v in votes_by_run.items():
        if best_run is None:
            best_run = k
            best_run_votes = v
        else:
            if v > best_run_votes:
                best_run = k
                best_run_votes = v

    # Get the model that has more votes
    best_model_path = os.path.join(client.get_run(best_run).info.artifact_uri, 'model.pkl')
    best_model_config_path = os.path.join(client.get_run(best_run).info.artifact_uri, 'config.yaml')
    if best_model_path.startswith('file:///'):
        best_model_path = best_model_path.replace('file:///', '')
    if best_model_config_path.startswith('file:///'):
        best_model_config_path = best_model_config_path.replace('file:///', '')
    return best_model_path, best_model_config_path
