"""
This script defines the names of all required environment variables.
"""
#######
# API #
#######

# General
ENV_API_DOMAIN = 'NEXUSML_API_DOMAIN'  # Domain of the RESTful API server (without "https://" or "http://")
ENV_RSA_KEY_FILE = 'NEXUSML_API_RSA_KEY_FILE'  # Private RSA key file used to sign JWT tokens

# Official clients
ENV_WEB_CLIENT_ID = 'NEXUSML_API_WEB_CLIENT_ID'  # UUID of the official web client

# Mail
ENV_MAIL_SERVER = 'NEXUSML_API_MAIL_SERVER'  # SMTP server used to send emails
ENV_MAIL_USERNAME = 'NEXUSML_API_MAIL_USERNAME'  # Username for the SMTP server
ENV_MAIL_PASSWORD = 'NEXUSML_API_MAIL_PASSWORD'  # Password for the SMTP server
ENV_NOTIFICATION_EMAIL = 'NEXUSML_API_NOTIFICATION_EMAIL'  # Email address used to send notifications
ENV_WAITLIST_EMAIL = 'NEXUSML_API_WAITLIST_EMAIL'  # Email address used to send waitlist notifications
ENV_SUPPORT_EMAIL = 'NEXUSML_API_SUPPORT_EMAIL'  # Email address used for support

############
# DATABASE #
############

ENV_DB_NAME = 'NEXUSML_DB_NAME'  # Name of the MySQL database
ENV_DB_USER = 'NEXUSML_DB_USER'  # MySQL database username
ENV_DB_PASSWORD = 'NEXUSML_DB_PASSWORD'  # MySQL database password

######################
# EXTERNAL RESOURCES #
######################

# CELERY (Optional)
ENV_CELERY_BROKER_URL = 'NEXUSML_CELERY_BROKER_URL'  # URL of the Redis server used as the Celery broker
ENV_CELERY_RESULT_BACKEND = 'NEXUSML_CELERY_RESULT_BACKEND'  # URL of the Redis server used as the Celery result backend

# AUTH0 (Optional)
ENV_AUTH0_DOMAIN = 'NEXUSML_AUTH0_DOMAIN'  # Domain of the Auth0 tenant
ENV_AUTH0_CLIENT_ID = 'NEXUSML_AUTH0_CLIENT_ID'  # Auth0 client management ID
ENV_AUTH0_CLIENT_SECRET = 'NEXUSML_AUTH0_CLIENT_SECRET'  # Auth0 client management secret
ENV_AUTH0_JWKS = 'NEXUSML_AUTH0_JWKS'  # URL of the Auth0 JWKS endpoint
ENV_AUTH0_SIGN_UP_REDIRECT_URL = 'NEXUSML_AUTH0_SIGN_UP_REDIRECT_URL'  # URL to redirect users to after signing up
ENV_AUTH0_TOKEN_AUDIENCE = 'NEXUSML_AUTH0_TOKEN_AUDIENCE'  # Recipient of tokens
ENV_AUTH0_TOKEN_ISSUER = 'NEXUSML_AUTH0_TOKEN_ISSUER'  # Issuer of tokens

# AWS (Optional)
# (check https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#guide-configuration)
ENV_AWS_ACCESS_KEY_ID = 'AWS_ACCESS_KEY_ID'  # Access key for the AWS account
ENV_AWS_SECRET_ACCESS_KEY = 'AWS_SECRET_ACCESS_KEY'  # Secret key for the AWS account
ENV_AWS_S3_BUCKET = 'AWS_S3_BUCKET'  # AWS S3 bucket name
