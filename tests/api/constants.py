from platformdirs import user_data_dir

from nexusml.constants import API_NAME
from nexusml.constants import API_VERSION

# FILES
WORKERS_LOGFILE = 'workers_log.txt'

# CONFIG
TEST_CONFIG = {
    'engine': {
        'worker': {
            'type': 'local'
        },
        'services': {
            'inference': {
                'enabled': True,
            },
            'continual_learning': {
                'enabled': True,
                'min_days': 7.0,
                'max_days': 28.0,
                'min_sample': 0.2,
                'min_cpu_quota': 600.0,
                'max_cpu_quota': 900.0,
                'cpu_hard_limit': 2000.0,
                'min_gpu_quota': 300.0,
                'max_gpu_quota': 450.0,
                'gpu_hard_limit': 1000.0,
            },
            'active_learning': {
                'enabled': True,
                'query_interval': 7,
                'max_examples_per_query': 50,
            },
            'monitoring': {
                'enabled': True,
                'refresh_interval': 100,
                'ood_predictions': {
                    'min_sample': 100,
                    'sensitivity': 0.5,
                    'smoothing': 0.8,
                }
            },
            'testing': {
                'enabled': True,
            },
        },
    },
    'general': {
        'enable_demo_tasks': True
    },
    'jobs': {
        'abort_upload_after': 7,
        'billing_time': '02:00',
        'max_workers': 5,
    },
    'limits': {
        'organizations': {
            'num_organizations': 1000,
            'picture_size': 400 * 1024,  # 400 KB
            'waitlist': 10**4,
        },
        'quotas': {
            'default_plan': {
                'max_apps': 3,
                'max_collaborators': 3,
                'max_cpu_hours': 0,
                'max_deployments': 0,
                'max_gpu_hours': 0,
                'max_predictions': 1000,
                'max_roles': 5,
                'max_tasks': 1,
                'max_users': 10,
                'max_examples': 10**4,
                'space_limit': 50 * 1024**2,  # 50 MB
            },
        },
        'requests': {
            'cache_timeout': 60,
            'max_payload': 800 * 1024,  # 800 KB
            'requests_per_day': 10**7,
            'requests_per_hour': 10**6,
            'requests_per_minute': 10**4,
            'requests_per_second': 10**3,
        },
        'tasks': {
            'picture_size': 400 * 1024,  # 400 KB
            'max_preloaded_categories': 100,  # TODO: this parameter should be removed when the cache works correctly
        },
    },
    'notifications': {
        'interval': 600,
        'max_source_events': 3,
        'mail_port': 587,
        'use_tls': True,
        'use_ssl': False
    },
    'security': {
        'api_keys': {
            'expiration': 60 * 60 * 24 * 30,  # 30 days
        },
        'public_id': {
            'min_length': 8,
        },
    },
    'server': {
        'api_url': f'/v{API_VERSION}',
    },
    'storage': {
        'database': {
            'type': 'mysql',
            'uri': 'mysql+pymysql://<user>:<password>@localhost:3306/<database>',
        },
        'files': {
            'backend': 's3',
            'local': {
                'max_upload_size': 100 * 1024**2,
                'root_path': user_data_dir(API_NAME),
                'url_expiration': 600,
            },
            's3': {
                'max_upload_size': 100 * 1024**2,
                'url_expiration': 600,
            },
        }
    },
    'views': {
        'default_items_per_page': 6,
        'max_items_per_page': 100,
    },
}

# MOCK CLIENT
CLIENT_AUTH0_HOST = 'localhost'
CLIENT_AUTH0_SERVER = 'http://' + CLIENT_AUTH0_HOST
CLIENT_AUTH0_PORT = 3000
CLIENT_AUTH0_CALLBACK = '/oauth2/callback'
CLIENT_SCOPES = ('organizations.create organizations.read '
                 'organizations.update organizations.delete '
                 'tasks.create tasks.read tasks.update tasks.delete '
                 'files.create files.read files.update files.delete '
                 'models.create models.read models.update models.delete '
                 'examples.create examples.read examples.update examples.delete '
                 'predictions.create predictions.read predictions.delete')
# Note: by default, `requests.adapters.HTTPAdapter()` keeps a maximum of 10 `ConnectionPool` instances.
#       See https://docs.python-requests.org/en/master/api/#requests.adapters.HTTPAdapter
CLIENT_MAX_THREADS = 1

# AUTH0
AUTH0_URL = '/oauth2'
AUTH0_AUTHORIZE_URL = AUTH0_URL + '/authorize'
AUTH0_TOKEN_URL = AUTH0_URL + '/token'

# MISC
BACKEND_JSON_FIELDS = {'id', 'uuid', 'created_at', 'created_by', 'modified_at', 'modified_by'}
