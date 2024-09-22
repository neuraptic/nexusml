import copy
import itertools
import os
from typing import List

import yaml

from nexusml.engine.exceptions import SchemaError
from nexusml.engine.schema.base import get_pipeline_type
from nexusml.engine.schema.base import Schema
from nexusml.enums import PipelineType


def default_dataframe_transforms():
    """
    Returns the default dataframe transforms for sklearn models.

    Returns:
        list: List of dictionaries with the default dataframe transforms
    """
    return [{
        'class': 'nexusml.engine.data.transforms.sklearn.SelectRequiredElements',
        'args': {
            'shapes': False
        }
    }, {
        'class': 'nexusml.engine.data.transforms.sklearn.DropNaNValues',
        'args': None
    }, {
        'class': 'nexusml.engine.data.transforms.sklearn.SimpleMissingValueImputation',
        'args': None
    }]


def sklearn_default_input_transforms():
    """
    Returns the default input transforms for sklearn models.

    Returns:
        dict: Dictionary with the default input transforms
    """
    return {
        'global': {
            'float': {
                'class': 'nexusml.engine.data.transforms.sklearn.StandardScalerTransform',
                'args': None
            },
            'category': {
                'class': 'nexusml.engine.data.transforms.sklearn.OneHotEncoderTransform',
                'args': None
            },
            'text': {
                'class': 'nexusml.engine.data.transforms.sklearn.TfIdfTransform',
                'args': None
            }
        },
        'specific': None
    }


def sklearn_default_output_transforms():
    """
    Returns the default output transforms for sklearn models.

    Returns:
        dict: Dictionary with the default output transforms
    """
    return {
        'global': {
            'float': {
                'class': 'nexusml.engine.data.transforms.sklearn.MinMaxScalerTransform',
                'args': None
            },
            'category': {
                'class': 'nexusml.engine.data.transforms.sklearn.LabelEncoderTransform',
                'args': None
            }
        },
        'specific': None
    }


def sklearn_embedding_input_transforms(discretize_float: bool, n_bins: int = 5):
    """
    Returns the default input transforms for sklearn models with embeddings.

    Args:
        discretize_float (bool): Whether to discretize the float values
        n_bins (int): Number of bins to discretize the float values

    Returns:
        dict: Dictionary with the default input transforms
    """
    if discretize_float:
        float_transform = {
            'class': 'nexusml.engine.data.transforms.sklearn.KBinsDiscretizerTransform',
            'args': {
                'n_bins': n_bins
            }
        }
    else:
        float_transform = {'class': 'nexusml.engine.data.transforms.sklearn.StandardScalerTransform', 'args': None}
    return {
        'global': {
            'float': float_transform,
            'category': {
                'class': 'nexusml.engine.data.transforms.sklearn.OrdinalEncoderTransform',
                'args': None
            }
        },
        'specific': None
    }


def sklearn_linear_default_model():
    """
    Returns the default configuration for a sklearn linear model.

    Returns:
        dict: Dictionary with the default configuration for a sklearn linear model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'sklearn.linear_model.LogisticRegression',
                'regression_model_class': 'sklearn.linear_model.LinearRegression'
            }
        }
    }


def sklearn_svm_default_model():
    """
    Returns the default configuration for a sklearn SVM model.

    Returns:
        dict: Dictionary with the default configuration for a sklearn SVM model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'sklearn.svm.SVC',
                'classification_model_args': {
                    'probability': True
                },
                'regression_model_class': 'sklearn.svm.SVR'
            }
        }
    }


def sklearn_tree_default_model():
    """
    Returns the default configuration for a sklearn Decision Tree model.

    Returns:
        dict: Dictionary with the default configuration for a sklearn Decision Tree model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'sklearn.tree.DecisionTreeClassifier',
                'regression_model_class': 'sklearn.tree.DecisionTreeRegressor'
            }
        }
    }


def sklearn_knn_default_model():
    """
    Returns the default configuration for a sklearn KNN model.

    Returns:
        dict: Dictionary with the default configuration for a sklearn KNN model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'sklearn.neighbors.KNeighborsClassifier',
                'regression_model_class': 'sklearn.neighbors.KNeighborsRegressor'
            }
        }
    }


def sklearn_random_forest_default_model():
    """
    Returns the default configuration for a sklearn Random Forest model.

    Returns:
        dict: Dictionary with the default configuration for a sklearn Random Forest model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'sklearn.ensemble.RandomForestClassifier',
                'regression_model_class': 'sklearn.ensemble.RandomForestRegressor'
            }
        }
    }


def sklearn_gradient_boosting_tree_default_model():
    """
    Returns the default configuration for a sklearn Gradient Boosting Tree model.

    Returns:
        dict: Dictionary with the default configuration for a sklearn Gradient Boosting Tree model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'sklearn.ensemble.GradientBoostingClassifier',
                'classification_model_args': {
                    'verbose': 0
                },
                'regression_model_class': 'sklearn.ensemble.GradientBoostingRegressor',
                'regression_model_args': {
                    'verbose': 0
                },
            }
        }
    }


def sklearn_mlp_default_model():
    """
    Returns the default configuration for a sklearn MLP model.

    Returns:
        dict: Dictionary with the default configuration for a sklearn MLP model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'sklearn.neural_network.MLPClassifier',
                'regression_model_class': 'sklearn.neural_network.MLPRegressor'
            }
        }
    }


def xgboost_default_model():
    """
    Returns the default configuration for a XGBoost model.

    Returns:
        dict: Dictionary with the default configuration for a XGBoost model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'xgboost.XGBClassifier',
                'classification_model_args': {
                    'use_label_encoder': False
                },
                'regression_model_class': 'xgboost.XGBRegressor'
            }
        }
    }


def catboost_default_model():
    """
    Returns the default configuration for a CatBoost model.

    Returns:
        dict: Dictionary with the default configuration for a
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'catboost.CatBoostClassifier',
                'classification_model_args': {
                    'verbose': False
                },
                'regression_model_class': 'catboost.CatBoostRegressor',
                'regression_model_args': {
                    'verbose': False
                }
            }
        }
    }


def lightgbm_default_model():
    """
    Returns the default configuration for a LightGBM model.

    Returns:
        dict: Dictionary with the default configuration for a LightGBM model
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_from_config',
            'setup_args': {
                'classification_model_class': 'lightgbm.LGBMClassifier',
                'regression_model_class': 'lightgbm.LGBMRegressor'
            }
        }
    }


def sklearn_linear_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a sklearn linear model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'sklearn.linear_model.LogisticRegression',
                'classification_model_params': [{
                    'penalty': ['l1'],
                    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'solver': ['liblinear'],
                    'max_iter': [1000]
                }, {
                    'penalty': ['l2'],
                    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'solver': ['lbfgs'],
                    'max_iter': [1000]
                }, {
                    'penalty': ['elasticnet'],
                    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'l1_ratio': [0.0, 0.5, 1.0],
                    'solver': ['saga'],
                    'max_iter': [1000]
                }, {
                    'penalty': ['none'],
                    'solver': ['lbfgs'],
                    'max_iter': [1000]
                }],
                'regression_model_class': 'sklearn.linear_model.ElasticNet',
                'regression_model_params': {
                    'alpha': [0.0, 0.5, 1.0, 2.0],
                    'l1_ratio': [0.0, 0.5, 1.0],
                    'max_iter': [1000]
                }
            }
        }
    }


def sklearn_svm_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a sklearn SVM model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs':
                    n_jobs,
                'cv':
                    cv,
                'classification_model_class':
                    'sklearn.svm.SVC',
                'classification_model_params': [
                    {
                        'probability': [True],
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['linear']
                    },
                    {
                        'probability': [True],
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['poly'],
                        'degree': [2, 3, 4],
                        'gamma': ['scale', 'auto']
                    },
                    {
                        'probability': [True],
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['rbf'],
                        'gamma': ['scale', 'auto']
                    },
                    {
                        'probability': [True],
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['sigmoid'],
                        'gamma': ['scale', 'auto']
                    },
                ],
                'regression_model_class':
                    'sklearn.svm.SVR',
                'regression_model_params': [
                    {
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['linear']
                    },
                    {
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['poly'],
                        'degree': [2, 3, 4],
                        'gamma': ['scale', 'auto']
                    },
                    {
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['rbf'],
                        'gamma': ['scale', 'auto']
                    },
                    {
                        'C': [0.2, 1.0, 2.0, 5.0],
                        'kernel': ['sigmoid'],
                        'gamma': ['scale', 'auto']
                    },
                ],
            }
        }
    }


def sklearn_tree_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a sklearn Decision Tree model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'sklearn.tree.DecisionTreeClassifier',
                'classification_model_params': {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 25],
                    'min_samples_split': [2, 5, 10, 0.1],
                    'min_samples_leaf': [1, 2, 5, 10, 0.1],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                },
                'regression_model_class': 'sklearn.tree.DecisionTreeRegressor',
                'regression_model_params': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 25],
                    'min_samples_split': [2, 5, 10, 0.1],
                    'min_samples_leaf': [1, 2, 5, 10, 0.1],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                },
            }
        }
    }


def sklearn_knn_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a sklearn KNN model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'sklearn.neighbors.KNeighborsClassifier',
                'classification_model_params': {
                    'n_neighbors': [1, 3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2, 3]
                },
                'regression_model_class': 'sklearn.neighbors.KNeighborsRegressor',
                'regression_model_params': {
                    'n_neighbors': [1, 3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2, 3]
                },
            }
        }
    }


def sklearn_random_forest_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a sklearn Random Forest model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'sklearn.ensemble.RandomForestClassifier',
                'classification_model_params': {
                    'n_estimators': [100, 500, 1000],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 25],
                    'min_samples_split': [2, 5, 10, 0.1],
                    'min_samples_leaf': [1, 5, 10, 0.1],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                },
                'regression_model_class': 'sklearn.ensemble.RandomForestRegressor',
                'regression_model_params': {
                    'n_estimators': [100, 500, 1000],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 25],
                    'min_samples_split': [2, 5, 10, 0.1],
                    'min_samples_leaf': [1, 5, 10, 0.1],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                },
            }
        }
    }


def sklearn_gradient_boosting_tree_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a sklearn Gradient Boosting Tree model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'sklearn.ensemble.GradientBoostingClassifier',
                'classification_model_params': {
                    'loss': ['deviance', 'exponential'],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'n_estimators': [100, 500, 1000],
                    'subsample': [0.75, 1.0],
                    'criterion': ['friedman_mse', 'squared_error'],
                    'min_samples_split': [2, 5, 10, 0.1],
                    'min_samples_leaf': [1, 5, 10, 0.1],
                    'max_depth': [3, 5, 15],
                    'verbose': [0]
                },
                'regression_model_class': 'sklearn.ensemble.GradientBoostingRegressor',
                'regression_model_params': {
                    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'n_estimators': [100, 500, 1000],
                    'subsample': [0.75, 1.0],
                    'criterion': ['friedman_mse', 'squared_error'],
                    'min_samples_split': [2, 5, 10, 0.1],
                    'min_samples_leaf': [1, 5, 10, 0.1],
                    'max_depth': [3, 5, 15],
                    'verbose': [0]
                }
            }
        }
    }


def sklearn_mlp_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a sklearn MLP model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'sklearn.neural_network.MLPClassifier',
                'classification_model_params': {
                    'hidden_layer_sizes': [(100,), (250,), (100, 50), (250, 100), (500, 250, 100), (250, 100, 50)],
                    'activation': ['relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.01, 0.001, 0.0001],
                    'batch_size': [32, 128, 'auto'],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'max_iter': [1000]
                },
                'regression_model_class': 'sklearn.neural_network.MLPRegressor',
                'regression_model_params': {
                    'hidden_layer_sizes': [(100,), (250,), (100, 50), (250, 100), (500, 250, 100), (250, 100, 50)],
                    'activation': ['relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.01, 0.001, 0.0001],
                    'batch_size': [32, 128, 'auto'],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'max_iter': [1000]
                }
            }
        }
    }


def xgboost_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a XGBoost model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'xgboost.XGBClassifier',
                'classification_model_params': {
                    'n_estimators': [50, 500, 1000],
                    'max_depth': [3, 5, 15],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'gamma': [0.0, 0.05, 0.1],
                    'min_child_weight': [1, 5, 9],
                    'subsample': [0.75, 1.0],
                    'reg_alpha': [0, 0.5, 1.0, 2.0],
                    'reg_lambda': [0, 0.5, 1.0, 2.0],
                    'verbosity': [0],
                    'use_label_encoder': [False]
                },
                'regression_model_class': 'xgboost.XGBRegressor',
                'regression_model_params': {
                    'n_estimators': [50, 500, 1000],
                    'max_depth': [3, 5, 15],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'gamma': [0.0, 0.05, 0.1],
                    'min_child_weight': [1, 5, 9],
                    'subsample': [0.75, 1.0],
                    'reg_alpha': [0, 0.5, 1.0, 2.0],
                    'reg_lambda': [0, 0.5, 1.0, 2.0],
                    'verbosity': [0]
                }
            }
        }
    }


def catboost_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a CatBoost model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'catboost.CatBoostClassifier',
                'classification_model_params': {
                    'iterations': [50, 500, 1000],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'l2_leaf_reg': [0, 0.5, 1.0, 2.0],
                    'subsample': [0.75, 1.0],
                    'depth': [3, 10, 25],
                    'min_data_in_leaf': [1, 5, 9],
                    'verbose': [False]
                },
                'regression_model_class': 'catboost.CatBoostRegressor',
                'regression_model_params': {
                    'iterations': [50, 500, 1000],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'l2_leaf_reg': [0, 0.5, 1.0, 2.0],
                    'subsample': [0.75, 1.0],
                    'depth': [3, 10, 25],
                    'min_data_in_leaf': [1, 5, 9],
                    'verbose': [False]
                }
            }
        }
    }


def lightgbm_gridsearch_model(n_jobs: int = 5, cv: int = 5):
    """
    Returns the configuration for a LightGBM model with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the configuration
    """
    return {
        'class': 'nexusml.engine.models.tabular.sklearn.SKLearnModel',
        'args': {
            'setup_function': 'nexusml.engine.models.tabular.sklearn._setup_sklearn_grid_search_from_config',
            'setup_args': {
                'n_jobs': n_jobs,
                'cv': cv,
                'classification_model_class': 'lightgbm.LGBMClassifier',
                'classification_model_params': {
                    'num_leaves': [15, 35],
                    'max_depth': [-1, 3, 10, 25],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'n_estimators': [50, 500, 1000],
                    'min_child_samples': [5, 15, 30],
                    'subsample': [0.75, 1.0],
                    'reg_alpha': [0, 0.5, 1.0, 2.0],
                    'reg_lambda': [0, 0.5, 1.0, 2.0],
                    'verbosity': [-1]
                },
                'regression_model_class': 'lightgbm.LGBMRegressor',
                'regression_model_params': {
                    'num_leaves': [15, 35],
                    'max_depth': [-1, 3, 10, 25],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'n_estimators': [50, 500, 1000],
                    'min_child_samples': [5, 15, 30],
                    'subsample': [0.75, 1.0],
                    'reg_alpha': [0, 0.5, 1.0, 2.0],
                    'reg_lambda': [0, 0.5, 1.0, 2.0],
                    'verbosity': [-1]
                }
            }
        }
    }


def get_sklearn_default_configs():
    """
    Returns the default configurations for sklearn models.

    Returns:
        dict: Dictionary with the default configurations for sklearn models
    """
    input_transforms = sklearn_default_input_transforms()
    output_transforms = sklearn_default_output_transforms()
    dataframe_transforms = default_dataframe_transforms()
    model_config_getters = {
        'linear': sklearn_linear_default_model,
        'svm': sklearn_svm_default_model,
        'decision_tree': sklearn_tree_default_model,
        'knn': sklearn_knn_default_model,
        'random_forest': sklearn_random_forest_default_model,
        'gradient_boosting_tree': sklearn_gradient_boosting_tree_default_model,
        'mlp': sklearn_mlp_default_model,
        'xgboost': xgboost_default_model,
        'catboost': catboost_default_model,
        'lightgbm': lightgbm_default_model
    }
    configs = {
        k: {
            'transforms': {
                'input_transforms': input_transforms,
                'output_transforms': output_transforms
            },
            'dataframe_transforms': dataframe_transforms,
            'model': v()
        } for k, v in model_config_getters.items()
    }
    return configs


def get_sklearn_gridsearch_configs(n_jobs: int = 5, cv: int = 5):
    """
    Returns the default configurations for sklearn models with grid search.

    Args:
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds

    Returns:
        dict: Dictionary with the default configurations for sklearn models with grid search
    """
    input_transforms = sklearn_default_input_transforms()
    output_transforms = sklearn_default_output_transforms()
    dataframe_transforms = default_dataframe_transforms()
    model_config_getters = {
        'linear_gs': sklearn_linear_gridsearch_model,
        'svm_gs': sklearn_svm_gridsearch_model,
        'decision_tree_gs': sklearn_tree_gridsearch_model,
        'knn_gs': sklearn_knn_gridsearch_model,
        'random_forest_gs': sklearn_random_forest_gridsearch_model,
        'gradient_boosting_tree_gs': sklearn_gradient_boosting_tree_gridsearch_model,
        'mlp_gs': sklearn_mlp_gridsearch_model,
        'xgboost_gs': xgboost_gridsearch_model,
        'catboost_gs': catboost_gridsearch_model,
        'lightgbm_gs': lightgbm_gridsearch_model
    }
    configs = {
        k: {
            'transforms': {
                'input_transforms': input_transforms,
                'output_transforms': output_transforms
            },
            'dataframe_transforms': dataframe_transforms,
            'model': v(n_jobs=n_jobs, cv=cv)
        } for k, v in model_config_getters.items()
    }
    return configs


def get_data_section_config(train_data: str, test_data: str):
    """
    Returns the data section configuration.

    Args:
        train_data (str): Path to the training data
        test_data (str): Path to the test data

    Returns:
        dict: Dictionary with the data section configuration
    """
    return {'train_data': train_data, 'test_data': test_data}


def get_experiment_tracking_section_config(experiment_name: str, experiment_save_path: str, save_figures: bool,
                                           save_predictions: bool):
    """
    Returns the experiment tracking section configuration.

    Args:
        experiment_name (str): Name of the experiment
        experiment_save_path (str): Path to save the experiment
        save_figures (bool): Whether to save the figures
        save_predictions (bool): Whether to save the predictions

    Returns:
        dict: Dictionary with the experiment tracking section configuration
    """
    return {
        'name': experiment_name,
        'mlflow_uri': experiment_save_path,
        'save_figures': save_figures,
        'save_predictions': save_predictions
    }


def generate_sklearn_default_configs(output_path: str,
                                     schema_path: str,
                                     train_data: str,
                                     test_data: str,
                                     experiment_name: str,
                                     experiment_save_path: str,
                                     save_figures: bool,
                                     save_predictions: bool,
                                     categories_path: str = None,
                                     seed: int = 0):
    """
    Generates the default configurations for sklearn models.

    Args:
        output_path (str): Path to save the configurations
        schema_path (str): Path to the schema
        train_data (str): Path to the training data
        test_data (str): Path to the test data
        experiment_name (str): Name of the experiment
        experiment_save_path (str): Path to save the experiment
        save_figures (bool): Whether to save the figures
        save_predictions (bool): Whether to save the predictions
        categories_path (str): Path to the categories
        seed (int): Seed for reproducibility

    Returns:
        None
    """

    # Check that the Pipeline Type is tabular classification/regression
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.TABULAR_CLASSIFICATION_REGRESSION:
        raise SchemaError("The schema does not follow the 'tabular_classification_regression' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)
    os.makedirs(output_path, exist_ok=True)
    for k, v in get_sklearn_default_configs().items():
        output_file = f'{os.path.join(output_path, k)}.yaml'
        if os.path.exists(output_file):
            print(f'The output file "{output_file} exist. Not overwritten')
        else:
            save_config = copy.deepcopy(v)
            save_config['data'] = data_config
            save_config['experiment'] = experiment_config
            save_config['schema'] = schema_path
            save_config['seed'] = seed
            if categories_path is not None:
                save_config['categories'] = categories_path
            with open(output_file, 'w') as f:
                yaml.dump(save_config, f)


def generate_sklearn_gridsearch_configs(output_path: str,
                                        schema_path: str,
                                        train_data: str,
                                        test_data: str,
                                        experiment_name: str,
                                        experiment_save_path: str,
                                        save_figures: bool,
                                        save_predictions: bool,
                                        categories_path: str = None,
                                        n_jobs: int = 5,
                                        cv: int = 5,
                                        seed: int = 0):
    """
    Generates the grid search configurations for sklearn models.

    Args:
        output_path (str): Path to save the configurations
        schema_path (str): Path to the schema
        train_data (str): Path to the training data
        test_data (str): Path to the test data
        experiment_name (str): Name of the experiment
        experiment_save_path (str): Path to save the experiment
        save_figures (bool): Whether to save the figures
        save_predictions (bool): Whether to save the predictions
        categories_path (str): Path to the categories
        n_jobs (int): Number of jobs
        cv (int): Number of cross-validation folds
        seed (int): Seed for reproducibility

    Returns:
        None
    """
    # Check that the Pipeline Type is tabular classification/regression
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.TABULAR_CLASSIFICATION_REGRESSION:
        raise SchemaError("The schema does not follow the 'tabular_classification_regression' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)
    os.makedirs(output_path, exist_ok=True)
    for k, v in get_sklearn_gridsearch_configs(n_jobs=n_jobs, cv=cv).items():
        output_file = f'{os.path.join(output_path, k)}.yaml'
        if os.path.exists(output_file):
            print(f'The output file "{output_file} exist. Not overwritten')
        else:
            save_config = copy.deepcopy(v)
            save_config['data'] = data_config
            save_config['experiment'] = experiment_config
            save_config['schema'] = schema_path
            save_config['seed'] = seed
            if categories_path is not None:
                save_config['categories'] = categories_path
            with open(output_file, 'w') as f:
                yaml.dump(save_config, f)


def pytorch_embedding_linear_model_config(discretize_float: bool = False,
                                          n_bins: int = 5,
                                          emb_size: int = 512,
                                          features_per_layer: List[int] = (100, 50, 10),
                                          batch_norm: bool = True,
                                          dropout_p: float = 0.5):
    """
    Returns the configuration for a pytorch embedding linear

    Args:
        discretize_float (bool): Whether to discretize the float values
        n_bins (int): Number of bins
        emb_size (int): Embedding size
        features_per_layer (List[int]): Number of features per layer
        batch_norm (bool): Whether to use batch normalization
        dropout_p (float): Dropout probability

    Returns:
        dict: Dictionary with the configuration
    """
    input_transforms = sklearn_embedding_input_transforms(discretize_float=discretize_float, n_bins=n_bins)
    output_transforms = sklearn_default_output_transforms()
    dataframe_transforms = default_dataframe_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms
        },
        'dataframe_transforms': dataframe_transforms,
        'model': {
            'class': 'nexusml.engine.models.tabular.pytorch.PytorchTabularModel',
            'args': {
                'setup_function': 'nexusml.engine.models.tabular.pytorch.create_pytorch_embedding_module',
                'setup_args': {
                    'emb_size': emb_size,
                    'features_per_layer': features_per_layer,
                    'batch_norm': batch_norm,
                    'dropout_p': dropout_p
                }
            }
        }
    }
    return config


def pytorch_embedding_transformer_model_config(discretize_float: bool = False,
                                               n_bins: int = 5,
                                               emb_size: int = 512,
                                               n_heads: int = 8,
                                               dim_feedforward: int = 1024,
                                               num_encoder_layers: int = 2):
    """
    Returns the configuration for a pytorch embedding transformer

    Args:
        discretize_float (bool): Whether to discretize the float values
        n_bins (int): Number of bins
        emb_size (int): Embedding size
        n_heads (int): Number of heads
        dim_feedforward (int): Dimension of the feedforward network model
        num_encoder_layers (int): Number of encoder layers

    Returns:
        dict: Dictionary with the configuration
    """
    input_transforms = sklearn_embedding_input_transforms(discretize_float=discretize_float, n_bins=n_bins)
    output_transforms = sklearn_default_output_transforms()
    dataframe_transforms = default_dataframe_transforms()

    config = {
        'transforms': {
            'input_transforms': input_transforms,
            'output_transforms': output_transforms
        },
        'dataframe_transforms': dataframe_transforms,
        'model': {
            'class': 'nexusml.engine.models.tabular.pytorch.PytorchTabularModel',
            'args': {
                'setup_function': 'nexusml.engine.models.tabular.pytorch.create_pytorch_transformer_module',
                'setup_args': {
                    'emb_size': emb_size,
                    'n_heads': n_heads,
                    'dim_feedforward': dim_feedforward,
                    'num_encoder_layers': num_encoder_layers
                }
            }
        }
    }
    return config


def get_pytorch_training_args(epochs: int = 50, lr: float = 0.005, batch_size: int = 64):
    """
    Returns the training arguments for pytorch models.

    Args:
        epochs (int): Number of epochs
        lr (float): Learning rate
        batch_size (int): Batch size

    Returns:
        dict: Dictionary with the training arguments
    """
    return {
        'loss_function': {
            'class': 'nexusml.engine.models.tabular.pytorch.BasicLossFunction',
            'args': {
                'classification_cost_sensitive': True
            }
        },
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size
    }


def generate_pytorch_model_configs(output_path: str,
                                   schema_path: str,
                                   train_data: str,
                                   test_data: str,
                                   experiment_name: str,
                                   experiment_save_path: str,
                                   save_figures: bool,
                                   save_predictions: bool,
                                   categories_path: str = None,
                                   seed: int = 0):
    """
    Generates the default configurations for pytorch models.

    Args:
        output_path (str): Path to save the configurations
        schema_path (str): Path to the schema
        train_data (str): Path to the training data
        test_data (str): Path to the test data
        experiment_name (str): Name of the experiment
        experiment_save_path (str): Path to save the experiment
        save_figures (bool): Whether to save the figures
        save_predictions (bool): Whether to save the predictions
        categories_path (str): Path to the categories
        seed (int): Seed for reproducibility

    Returns:
        None
    """
    # Check that the Pipeline Type is tabular classification/regression
    schema = Schema.create_schema_from_json(json_file=schema_path)
    if get_pipeline_type(schema) != PipelineType.TABULAR_CLASSIFICATION_REGRESSION:
        raise SchemaError("The schema does not follow the 'tabular_classification_regression' PipelineType")

    # Generate configs
    data_config = get_data_section_config(train_data=train_data, test_data=test_data)
    experiment_config = get_experiment_tracking_section_config(experiment_name=experiment_name,
                                                               experiment_save_path=experiment_save_path,
                                                               save_figures=save_figures,
                                                               save_predictions=save_predictions)

    training_args = get_pytorch_training_args()

    os.makedirs(output_path, exist_ok=True)

    embedding_linear_model_configs = {
        'discretize_float': [False, True],
        'n_bins': [5],
        'emb_size': [512],
        'features_per_layer': [(128,), (256, 128)],
        'batch_norm': [True],
        'dropout_p': [0.5]
    }

    embedding_linear_model_configs = [[(k, v) for v in vs] for k, vs in embedding_linear_model_configs.items()]
    embedding_linear_model_configs = list(map(dict, itertools.product(*embedding_linear_model_configs)))
    for i, model_params in enumerate(embedding_linear_model_configs):
        output_file = os.path.join(output_path, f'torch_emb_linear_{i + 1}.yaml')
        if os.path.exists(output_file):
            print(f'The output file "{output_file} exist. Not overwritten')
        else:
            save_config = pytorch_embedding_linear_model_config(**model_params)
            save_config['data'] = data_config
            save_config['experiment'] = experiment_config
            save_config['schema'] = schema_path
            save_config['seed'] = seed
            save_config['training'] = training_args
            if categories_path is not None:
                save_config['categories'] = categories_path
            with open(output_file, 'w') as f:
                yaml.dump(save_config, f)

    embedding_transformer_model_configs = {
        'discretize_float': [True],
        'n_bins': [5],
        'emb_size': [512],
        'n_heads': [8],
        'dim_feedforward': [1024],
        'num_encoder_layers': [2]
    }

    embedding_transformer_model_configs = [
        [(k, v) for v in vs] for k, vs in embedding_transformer_model_configs.items()
    ]
    embedding_transformer_model_configs = list(map(dict, itertools.product(*embedding_transformer_model_configs)))
    i = 1
    for model_params in embedding_transformer_model_configs:
        # Ensure that emb_size is divisible by n_heads in transformer model
        if (model_params['emb_size'] % model_params['n_heads']) == 0:
            output_file = os.path.join(output_path, f'torch_tab_transformer_{i}.yaml')
            if os.path.exists(output_file):
                print(f'The output file "{output_file} exist. Not overwritten')
            else:
                save_config = pytorch_embedding_transformer_model_config(**model_params)
                save_config['data'] = data_config
                save_config['experiment'] = experiment_config
                save_config['schema'] = schema_path
                save_config['seed'] = seed
                save_config['training'] = training_args
                if categories_path is not None:
                    save_config['categories'] = categories_path
                with open(output_file, 'w') as f:
                    yaml.dump(save_config, f)
            i += 1
