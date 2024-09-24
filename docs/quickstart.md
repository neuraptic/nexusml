# Quickstart Guide

Welcome to the NexusML Quickstart Guide! This guide will help you set up and start using NexusML in a few simple steps.

<!-- toc -->

- [Prerequisites](#prerequisites)
  - [MySQL Server](#mysql-server)
  - [Redis](#redis)
  - [Auth0](#auth0)
  - [RSA Key Pair](#rsa-key-pair)
  - [AWS S3 (Optional)](#aws-s3-optional)
- [Installation](#installation)
- [Usage](#usage)

## Prerequisites

This guide uses [Ubuntu](https://ubuntu.com/download), but you may choose any Linux distribution that suits your 
needs. However, we can only ensure compatibility with Ubuntu or [Amazon Linux](https://aws.amazon.com/amazon-linux-2). 
Windows is also supported.

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

### Auth0

NexusML uses [Auth0](https://auth0.com/) for authentication, ensuring secure login and access management. To set up 
Auth0 for NexusML, follow these steps:

1. **Create an Auth0 Account**:

   - Visit [Auth0's website](https://auth0.com/) and sign up for a free account.
   - Once logged in, go to the **Dashboard**.

2. **Create a New Application**:

   - In the Dashboard, click on **Applications** in the sidebar and select **Create Application**.
   - Choose **Regular Web Application** or **Machine to Machine Application**, depending on your needs.
   - Name your application and choose the platform that best suits your setup.

3. **Configure Application Settings**:

   - After creating your application, navigate to its settings.
   - Add the **callback URLs** that will be used for authentication. Example:

     ```
     http://localhost:3000/callback
     ```

4. **Get Client ID and Client Secret**:

    - Scroll down to find the **Client ID** and **Client Secret** (you will need these to configure NexusML).

5. **Set NexusML Environment Variables**

    - Set the following environment variables:

      ```
      NEXUSML_AUTH0_DOMAIN=<your-auth0-domain>
      NEXUSML_AUTH0_CLIENT_ID=<your-auth0-client-id>
      NEXUSML_AUTH0_CLIENT_SECRET=<your-auth0-client-secret>
      NEXUSML_AUTH0_JWKS=https://<your-auth0-domain>/.well-known/jwks.json
      NEXUSML_AUTH0_SIGN_UP_REDIRECT_URL=<your-redirect-url>
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
