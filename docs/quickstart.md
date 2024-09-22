# Quickstart Guide

Welcome to the NexusML Quickstart Guide! This guide will help you set up and start using NexusML in a few simple steps.

## Prerequisites

1. **Install a Linux Distribution**

   We'll be using Ubuntu, but feel free to choose any Linux distribution that suits you. However, we can only ensure 
   compatibility with Ubuntu or Amazon Linux. Windows is also supported. To continue with the installation, you will 
   need an Ubuntu distribution to run NexusML. If you do not already have it, install Ubuntu on your host machine. 
   Ensure that you have the 64-bit version of one of the following Ubuntu releases:

   - Ubuntu Noble 24.04 (LTS)
   - Ubuntu Mantic 23.10 (EOL: July 12, 2024)
   - Ubuntu Jammy 22.04 (LTS)
   - Ubuntu Focal 20.04 (LTS)

2. **Install MySQL**

   TODO: Complete this section.

3. **Install REDIS**

   TODO: Complete this section.

## Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/yourusername/project-name.git
    ```

2. **Navigate to the Project Directory**

    ```sh
    cd project-name
    ```

3. **Install Dependencies**
   NexusML uses `setup.py` to install all required dependencies and set up the basic configuration. Some dependencies may 
   require additional steps before installation.

    ```sh
    python -m pip install nexusml
    ```

    If you are a contributor and want to have an editable installation, run:

    ```sh
    python -m pip install nexusml -e
    ```
   
   For the API, you need to install specific requirements:

   ```sh
   python -m pip install nexusml[api]
   ```
   
   For `detectron2`, you may need to install it manually along with GCC. Visit the 
   [Detectron2 repository](https://github.com/facebookresearch/detectron2) for more information.



## Usage

TODO: Complete this section.
