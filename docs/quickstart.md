# Quickstart Guide

Welcome to the NexusML Quickstart Guide! This guide is designed to help you set up and use NexusML, with clear 
instructions to ensure a smooth onboarding experience.

<!-- toc -->

- [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [MySQL Server](#mysql-server)
  - [Redis](#redis)
  - [RSA Key Pair](#rsa-key-pair)
  - [Auth0 (Optional)](#auth0-optional)
  - [AWS S3 (Optional)](#aws-s3-optional)
- [Installation](#installation)
- [Usage](#usage)

## Prerequisites

This guide uses [Ubuntu](https://ubuntu.com/download), but you may choose any Linux distribution that suits your 
needs. However, we can only ensure compatibility with Ubuntu or [Amazon Linux](https://aws.amazon.com/amazon-linux-2). 
Windows is also supported.

### Environment Variables

The [nexusml/env.py](../nexusml/env.py) file contains the environment variables used by NexusML.

Here is the list of required environment variables:

- `NEXUSML_API_DOMAIN`: The domain of the RESTful API server (without "https://" or "http://").
- `NEXUSML_API_RSA_KEY_FILE`: The path to the RSA private key file used to sign JWT tokens.
- `NEXUSML_API_WEB_CLIENT_ID`: The UUID of the official web client.
- `NEXUSML_API_MAIL_SERVER`: The SMTP server used to send emails.
- `NEXUSML_API_MAIL_USERNAME`: The username for the SMTP server.
- `NEXUSML_API_MAIL_PASSWORD`: The password for the SMTP server.
- `NEXUSML_API_NOTIFICATION_EMAIL`: The email address used to send notifications.
- `NEXUSML_API_WAITLIST_EMAIL`: The email address used to send waitlist notifications.
- `NEXUSML_API_SUPPORT_EMAIL`: The email address used for support.
- `NEXUSML_DB_NAME`: The name of the MySQL database.
- `NEXUSML_DB_USER`: The MySQL database username.
- `NEXUSML_DB_PASSWORD`: The MySQL database password.

Here is the list of optional environment variables:

- `NEXUSML_CELERY_BROKER_URL`: The URL of the Redis server used as the Celery broker. Defaults to 
  "redis://localhost:6379/0".
- `NEXUSML_CELERY_RESULT_BACKEND`: The URL of the Redis server used as the Celery result backend. Defaults to 
  "redis://localhost:6379/0".
- `NEXUSML_AUTH0_DOMAIN`: The domain of the Auth0 tenant.
- `NEXUSML_AUTH0_CLIENT_ID`: Auth0 client management ID.
- `NEXUSML_AUTH0_CLIENT_SECRET`: Auth0 client management secret.
- `NEXUSML_AUTH0_JWKS`: The URL of the Auth0 JWKS endpoint.
- `NEXUSML_AUTH0_SIGN_UP_REDIRECT_URL`: The URL to redirect users to after signing up.
- `NEXUSML_AUTH0_TOKEN_AUDIENCE`: The audience of Auth0 tokens.
- `NEXUSML_AUTH0_TOKEN_ISSUER`: The issuer of Auth0 tokens.
- `AWS_ACCESS_KEY_ID`: The AWS access key ID.
- `AWS_SECRET_ACCESS_KEY`: The AWS secret access key.
- `AWS_S3_BUCKET`: The name of the AWS S3 bucket used by the file storage backend.

Note: If you are using Auth0 or AWS S3, you will need to set all the environment variables related to these services.

### MySQL Server

NexusML uses MySQL as the database management system. You can install MySQL Server by following the instructions on 
the [official MySQL website](https://dev.mysql.com/doc/refman/8.4/en/installing.html). Installing MySQL Server on 
Ubuntu is a straightforward process (check the official 
[Ubuntu documentation](https://documentation.ubuntu.com/server/how-to/databases/install-mysql/) for more 
information):

First, update your package index and install MySQL Server:

```sh
sudo apt update
sudo apt install mysql-server
```

After the installation, it's recommended to secure your MySQL installation. This includes removing insecure default 
settings and setting up a root password. Run:

```sh
sudo mysql_secure_installation
```

Follow the on-screen instructions.

To start MySQL and ensure it runs at boot, run the following commands:

```sh
sudo systemctl start mysql
sudo systemctl enable mysql
```

You can check if MySQL is running with:

```sh
sudo systemctl status mysql
```

To log in to the MySQL shell as root, run:

```sh
sudo mysql
```

You should see the MySQL prompt (`mysql>`), indicating that MySQL is installed and running correctly.

### Redis

NexusML uses [Celery](https://docs.celeryq.dev/en/stable/) to handle asynchronous and scheduled jobs and 
[Redis](https://redis.io/) as the message broker for Celery. While Celery can be installed as a Python package, Redis 
requires a separate installation. To install Redis on Ubuntu, please follow the instructions on the official 
[Redis documentation](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/install-redis-on-linux/). 
Here is a quick guide to installing Redis on Ubuntu:

First, update your package index and install Redis:

```sh
sudo apt update
sudo apt install redis-server
```

After installation, configure Redis to run as a background service by editing its configuration file.
 
```sh
sudo nano /etc/redis/redis.conf
```

Change `supervised no` to `supervised systemd`. After making this change, restart the Redis service:

```sh
sudo systemctl restart redis-server
```

Verify that Redis is running:

```sh
sudo systemctl status redis-server
```

Use the Redis CLI to check the server’s response:

```sh
redis-cli
127.0.0.1:6379> ping
```

You should receive a `PONG` response. To exit the Redis CLI, type:

```sh
exit
```

Enable Redis to start at boot:

```sh
sudo systemctl enable redis-server
```

### RSA Key Pair

NexusML uses RSA key pairs to sign and verify JWT tokens, ensuring secure communication between services. Follow these 
steps to generate an RSA key pair:

1. **Generate the RSA Key Pair**: Use `openssl` to generate a private and public key pair.

   ```sh
   openssl genpkey -algorithm RSA -out private_key.pem -pkeyopt rsa_keygen_bits:2048
   openssl rsa -pubout -in private_key.pem -out public_key.pem
   ```
   
   This will create two files:

    - `private_key.pem`: The private key used to sign JWT tokens.
    - `public_key.pem`: The public key used to verify JWT tokens.

2. **Secure Your Keys**:

   - Keep your `private_key.pem` safe and do not share it. This key is used to sign JWT tokens.
   - The `public_key.pem` can be shared with services that need to verify JWT tokens.

3. **Set NexusML Environment Variables**

   - Set the following environment variables:

     ```
     NEXUSML_API_RSA_KEY_FILE=<path-to-private-key>
     ```
     
For more information about RSA key management and JWT tokens, please refer to the [OpenSSL](https://www.openssl.org/) 
and [JWT](https://jwt.io/) documentation.

### Auth0 (Optional)

<div style="text-align: left;">
  <div style="border: 1px solid #f5c6cb; padding: 10px 10px; background-color: rgba(248, 215, 218, 0.5); color: #721c24; display: flex; align-items: flex-start;">
    <span style="margin-right: 10px;">⚠️</span>
    <div>
      While Auth0 is optional when running NexusML in single-tenant, single-client mode, it is highly recommended to set up Auth0 in production environments for security reasons.
    </div>
  </div>
  <p></p>
</div>

NexusML uses [Auth0](https://auth0.com/) for authentication, ensuring secure login and access management. To set up 
Auth0 for NexusML, please refer to the instructions in [auth0.md](auth0.md). After setting up Auth0, you will need to 
set the following environment variables:

```
NEXUSML_AUTH0_DOMAIN=<your-auth0-domain>
NEXUSML_AUTH0_CLIENT_ID=<your-auth0-client-id>
NEXUSML_AUTH0_CLIENT_SECRET=<your-auth0-client-secret>
NEXUSML_AUTH0_JWKS=https://<your-auth0-domain>/.well-known/jwks.json
NEXUSML_AUTH0_SIGN_UP_REDIRECT_URL=<your-redirect-url>
```

### AWS S3 (Optional)

NexusML can use [AWS S3](https://aws.amazon.com/s3/) as the file storage backend. Follow the steps below to 
configure AWS S3 for NexusML:

1. **Create an AWS Account**:

   - Visit the [AWS website](https://aws.amazon.com/) and sign up for an account.
   - Once logged in, go to the **AWS Management Console**.

2. **Create an S3 Bucket**:

   - In the AWS Management Console, search for **S3** and click on the service.
   - Click on **Create bucket** and choose a unique bucket name.

3. **Set Bucket Permissions**:

   - In the bucket settings, navigate to the **Permissions** tab.
   - Under **Bucket Policy**, add a policy to allow NexusML access to your bucket.

4. **Create an IAM User and Get Access Credentials**:

   - In the AWS Management Console, go to **IAM** (Identity and Access Management).
   - Click on **Users** and then **Add User**.
   - Provide a username and select **Programmatic access** to generate access keys.
   - Attach the **AmazonS3FullAccess** policy to the user or create a custom policy with more specific permissions.
   - After creating the user, download or note the **Access Key ID** and **Secret Access Key**.

5. **Set Environment Variables**:

   - Set the following environment variables:

     ```
     AWS_ACCESS_KEY_ID=<your-aws-access-key>
     AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>
     AWS_S3_BUCKET=<your-s3-bucket-name>
     ```

## Installation

To install NexusML, run:

```sh
python -m pip install nexusml
```

Note: The `detectron2` package, required by the engine, might need manual installation. Please refer to the 
[Detectron2](https://github.com/facebookresearch/detectron2) repository for detailed instructions. You may also need 
to install [GCC](https://gcc.gnu.org/).

## Usage

You can use NexusML in two ways:

- **As a Python Package**. You can import NexusML into your Python project and use it as a library. You can either use 
  the engine directly or extend the RESTful API to create custom endpoints.
- **As a Standalone Service**. You can run NexusML as a standalone RESTful API. This allows you to interact with 
  NexusML using HTTP requests. To start the RESTful API Server, run:

  ```sh
  nexusml-server
  ```

If you are not running the RESTful API server in production, you can use the default API key to access the API. This 
API key allows you to access the API without authentication, avoiding the need to set up Auth0. To get the default 
API key, either copy the message shown by the `nexusml-server` command at startup or open the `default_api_key.txt` 
file located in the `nexusml` package directory.

<div style="text-align: left;">
  <div style="border: 1px solid #f5c6cb; padding: 10px 10px; background-color: rgba(248, 215, 218, 0.5); color: #721c24;">
    ⚠️ Make sure that the default API key is not enabled in production
  </div>
  <p></p>
</div>

It is important to make sure that the default API key is not enabled in production. To disable the default API key, 
update the following line in the `config.yaml` file located in the `nexusml` package directory:

```yaml
general:
  default_api_key_enabled: false
```
