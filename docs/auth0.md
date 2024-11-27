# Auth0 Setup Guide

<!-- toc -->

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Create a Custom API](#create-a-custom-api)
- [Create a Machine-to-Machine (M2M) Application](#create-a-machine-to-machine-m2m-application)
- [Create a Web Application](#create-a-web-application)
- [Create a Login Flow](#create-a-login-flow)

## Introduction

This guide provides a comprehensive walkthrough for configuring Auth0 to integrate with NexusML. It covers the complete 
setup process, from initial prerequisites to the creation of NexusML-related scopes and permissions. For additional 
resources, visit the [Auth0 Quickstarts](https://auth0.com/docs/quickstarts).

## Prerequisites

Before you start, ensure you have:

- An Auth0 account.
- Administrator access to your Auth0 tenant.

To follow the steps below, log in to the [Auth0 Dashboard](https://manage.auth0.com/).

## Create a Custom API

### 1. Create a New API

1. In the left-hand sidebar, navigate to **Applications**.
2. Click on **APIs** under the **APIs** section.
3. Click on the **+ Create API** button.

### 2. Configure the API

1. **Name Your API**: Enter a name for your API, e.g., "NexusML API".
2. **Identifier**: This is a unique identifier for your API. It can be a URL or a unique string, e.g., 
   `https://nexusml.example.com`.
3. **Signing Algorithm**: Choose the signing algorithm for your API tokens. The default and recommended option is 
   **RS256**.
4. Click **Create** to proceed.

### 3. Define NexusML Scopes/Permissions

1. After creating the API, you'll be redirected to the API's settings page.
2. Click on the **Permissions** tab to define the scopes for your API.
3. Click **+ Add Permission** to start adding the scopes required by NexusML:
    - `examples.create` - Create examples
    - `examples.delete` - Delete examples
    - `examples.read` - Read examples
    - `examples.update` - Update examples
    - `files.create` - Create files
    - `files.delete` - Delete files
    - `files.read` - Read files
    - `files.update` - Update files
    - `models.create` - Create AI models
    - `models.delete` - Delete AI models
    - `models.read` - Read AI models
    - `models.update` - Update AI models
    - `organizations.create` - Create organizations
    - `organizations.delete` - Delete organizations
    - `organizations.read` - Read organizations
    - `organizations.update` - Update organizations
    - `predictions.read` - Read predictions
    - `tasks.create` - Create tasks
    - `tasks.delete` - Delete tasks
    - `tasks.read` - Read tasks
    - `tasks.update` - Update tasks

## Create a Machine-to-Machine (M2M) Application

### 1. Create a New Application

1. In the left-hand sidebar, navigate to **Applications**.
2. Click on **Applications** under the **Applications** section.
3. Click on the **+ Create Application** button.

### 2. Configure the Application

1. **Name Your Application**: Provide a name for your M2M application, e.g., "API".
2. **Choose Application Type**: Select **Machine to Machine Applications**.
3. Click **Create** to proceed.

### 3. Configure API Access

1. After creating the application, you'll be redirected to a screen where you can configure API access.
2. Under **Select an API**, choose the API you want this application to access. If you haven’t created an API yet, you 
   can do so under the **APIs** section in the left-hand sidebar.
3. After selecting the API, you will see the scopes that the application can request access to. Check the appropriate 
   scopes based on your needs.
4. Click **Authorize** to grant access.

### 4. View Application Credentials

1. Navigate to the **Settings** tab of your application.
2. Here, you'll find your application’s **Client ID** and **Client Secret**. These credentials are necessary for 
   authenticating your application when making API requests.

### 5. Set Up Environment Variables

```bash
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret
AUTH0_DOMAIN=your-auth0-domain
```

## Create a Web Application

### 1. Create a New Application

1. In the left-hand sidebar, navigate to **Applications**.
2. Click on **Applications** under the **Applications** section.
3. Click on the **+ Create Application** button.

### 2. Configure the Application

1. **Name Your Application**: Enter a name for your React application, e.g., "Web Application".
2. **Choose Application Type**: Select **Single Page Web Applications**.
3. Click on **Create** to proceed.

### 3. Configure the Application Settings

1. After creating the application, you will be redirected to the application's settings page.
2. Under the **Settings** tab, configure the following:
   - **Allowed Callback URLs**: Add the URL where Auth0 should redirect after authentication. For development, this is 
     typically `http://localhost:3000`.
   - **Allowed Logout URLs**: Add the URL where Auth0 should redirect after logout. For development, this is also 
     `http://localhost:3000`.
   - **Allowed Web Origins**: Add the origin of your React application, e.g., `http://localhost:3000`.

3. Click **Save Changes** to apply these settings.

## Create a Login Flow

### 1. Create a Role

1. In the left-hand sidebar, navigate to **User Management** > **Roles**.
2. Click **+ Create Role** if you haven't created the role yet. Otherwise, identify the existing role you want to 
   assign automatically.
3. Name the role (e.g., "NexusML Basic User").
4. Under **Permissions**, assign all the scopes defined previously in the API section for your application.
5. Save the role and note down the **Role ID**. This will be used as `NEXUSML_BASIC_ROL_ID`.

### 2. Create an Action to Assign a Role on Login

1. Click on **Triggers** under **Actions**.
2. Select **post-login**.
3. In the menu on the right side, beside **Add Action**, click on the **+** icon and select **Build from scratch**.
4. Name your action (e.g., "Assign Role on Login") and select the recommended Node.js version under **Runtime**.
5. Introduce the following secrets:
   - `NEXUSML_BASIC_ROL_ID`: The ID of the role you want to assign automatically.
   - `NEXUSML_CLIENT_ID`: The Client ID of your application.
   - `NEXUSML_ACTIONS_CLIENT_ID`: The Client ID used for Auth0 Actions.
   - `NEXUSML_ACTIONS_DOMAIN`: Your Auth0 domain (e.g., `https://your-domain.auth0.com`).
   - `NEXUSML_ACTIONS_CLIENT_SECRET`: The Client Secret used for Auth0 Actions.

6. Add the following JavaScript code to the action script:

    ```javascript
    /**
    * Handler that will be called during the execution of a PostLogin flow.
    *
    * @param {Event} event - Details about the user and the context in which they are logging in.
    * @param {PostLoginAPI} api - Interface whose methods can be used to change the behavior of the login.
    */
    
    exports.onExecutePostLogin = async (event, api) => {
      // Verify if login comes from NexusML
      if(event.client.client_id === event.secrets.NEXUSML_CLIENT_ID) {    
        // Assign roles on first login (when user signs up)
        // It's done here because Pre/Post User Registration
        // doesn't contain event.client.client_id
        const not_first_login = event.stats.logins_count > 1 ||
                            event.transaction?.protocol === 'oauth2-refresh-token' ||
                            event.request.query?.prompt === 'none';
    
        if(!not_first_login) {
          // Create management API client instance
          const ManagementClient = require("auth0").ManagementClient;
    
          // Application info of NEXUSML_Actions app (Machine to Machine)
          const management = new ManagementClient({
            domain: event.secrets.NEXUSML_ACTIONS_DOMAIN,
            clientId: event.secrets.NEXUSML_ACTIONS_CLIENT_ID,
            clientSecret: event.secrets.NEXUSML_ACTIONS_CLIENT_SECRET,
          });
    
          const params =  { id : event.user.user_id };
          const data = { "roles" : [event.secrets.NEXUSML_BASIC_ROL_ID] };
    
          try {
            await management.users.assignRoles(params, data);
            console.log('Role assigned');
          } catch (e) {
            console.log(e);
          }
        }
      }
    };
    ```

7. Save the action by clicking on **Deploy**.

### 3. Attach the Action to the Login Flow

1. Go to **Triggers** under **Actions** and select **post-login**.
2. Drag and drop the newly created action under **Custom** section, "Assign Role on Login", into the flow, ideally 
   right after the user logs in.
3. Save the flow by clicking **Apply**.
