# Table of Contents

<!-- toc -->

- [Requirements](#requirements)
- [Coding Style](#coding-style)
  - [Project-Specific Conventions](#project-specific-conventions)
  - [Docstring Formatting](#docstring-formatting)
  - [Linters and Formatters](#linters-and-formatters)
- [Git](#git)
  - [Branch Naming](#branch-naming)
  - [Commits](#commits)
  - [Pull Requests](#pull-requests)
- [Release Versioning](#release-versioning)
- [Release Notes](#release-notes)
- [Abbreviation Guide](#abbreviation-guide)

## Requirements

Check the [dev-requirements.txt](dev-requirements.txt) file for the development dependencies.

## Coding Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and the 
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) along with some additional 
project-specific conventions. Check [pylintrc](pylintrc) and [.style.yapf](.style.yapf) for the specific formatting 
rules.

### Project-Specific Conventions

- **Use Type Hints**: Define functions and declare variables using type hints.

  For example:

  ```
  def compute_result(x: int, y: int) -> int:
      return x + y
  
  result: int = compute_result(x=5, y=10)
  ```
  
  Instead of:
  
  ```
  def compute_result(x, y):
      return x + y
  
  result = compute_result(x=5, y=10)
  ```

- **Use Type Checking**: Use type checking for typing only imports. When code needs to be seen by a type checker but shouldn't be executed at runtime, 
  use the `TYPE_CHECKING` constant from the `typing` module. 
  Check the official documentation on ["runtime or type checking"](https://peps.python.org/pep-0484/#runtime-or-type-checking) 
  for more details. 
  
  For example:

  ```
  import typing

  if typing.TYPE_CHECKING:
      import expensive_mod

  def a_func(arg: 'expensive_mod.SomeClass') -> None:
      a_var = arg  # type: expensive_mod.SomeClass
      ...
    ```
  
- **Use Keyword Arguments**: Call functions passing keyword arguments instead of positional arguments.

  For example:
  
  ```
  save_result(result=3, verbose=True)
  ```
  
  Instead of:
  
  ```
  save_result(3, True)
  ```

- **Use [f-string](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)**: Use f-strings for string 
  formatting.

  For example:
  
  ```
  name = 'John'
  print(f'Hello, {name}!')
  ```
  
  Instead of:
  
  ```
  name = 'John'
  print('Hello, {}!'.format(name))
  ```

  or

  ```
  name = 'John'
  print('Hello, %s!' % name)
  ```

- **Use the three-double-quote `"""` format only for docstrings**: Do not use it for section or block comments.

  For example:
  
  ```
  ##################
  # Set basic info #
  ##################
  
  info = dict()
  
  # Personal info
  info['name'] = 'John'
  info['age'] = 30
  
  # Professional info
  info['company'] = 'Neuraptic AI'
  info['position'] = 'Data Scientist'
  
  #########################
  # Save info to database #
  #########################
  
  # Connect to database server
  host = 'localhost'
  port = 3000
  db = connect_to_db(host, port)
  
  # Save info
  db.save(info)
  ```
  
  Instead of:
  
  ```
  """
  Set basic info
  """
  
  info = dict()
  
  # Personal info
  info['name'] = 'John'
  info['age'] = 30
  
  # Professional info
  info['company'] = 'Neuraptic AI'
  info['position'] = 'Data Scientist'
  
  """
  Save info to database
  """
  
  # Connect to database server
  host = 'localhost'
  port = 3000
  db = connect_to_db(host, port)
  
  # Save info
  db.save(info)
  ```

- **Break long function argument lists**: If a function call or definition has many arguments, break it into 
  multiple lines, with each line containing only one argument. Check 
  ["3.19.2 Line Breaking"](https://android.googlesource.com/platform/external/google-styleguide/+/refs/tags/android-s-beta-2/pyguide.md#3_19_2-line-breaking)
  section of the Google Python Style Guide for more details.

  For example:
  
  ```
  def send_email(
      subject: str,
      body: str,
      to: str,
      cc: Optional[str] = None,
      bcc: Optional[str] = None,
      attachments: Optional[List[str]] = None
  ) -> None:
      ...
  ```
  
  Instead of:
  
  ```
  def send_email(subject: str, body: str, to: str, cc: Optional[str] = None, bcc: Optional[str] = None, attachments: Optional[List[str]] = None) -> None:
      ...
  ```
  
- **Use Parenthesis for Line Breaks**: Break lines with parenthesis (`()`) instead of backslashes (`\`).

  For example:
  
  ```
  from transformers import (CLIPModel, ViTModel, 
                            RobertaModel, RobertaTokenizer)
  ```
  
  Instead of:
  
  ```
  from transformers import CLIPModel, ViTModel, \ 
    RobertaModel, RobertaTokenizer
  ```

- **String Quote Format**: Use single quotes for strings unless they contain single quotes.
  For strings containing single quotes, use double quotes to avoid escaping.

  For example:

  ```
  use_single = 'This is a string with no single quotes inside.'
  use_double = "It's easier to use double quotes here."
  ```

- **Import Rules**:

  - **For Python Built-in Libraries**: Always import the entire library. However, some well-known individual elements may be imported directly.

    For example:

    ```
    import os
    ```
   
    Instead of:

    ```
    from os import environ
    ```

    Common exceptions:

    ```
    from datetime import datetime
    from typing import List
    from collections import Counter
    from abc import abstractmethod
    from dataclasses import dataclass
    from pathlib import Path
    from threading import Thread
    ```
    
  - **For Third-Party and Project-Specific Modules**: Always use absolute imports instead of relative paths and explicitly import the classes and
    functions needed, except for well-known libraries.
    
    For example:

    ```
    from my_project.module import MyClass
    ```
   
    Instead of:

    ```
    from .module import MyClass
    ```

    Common exceptions:

    ```
    import boto3
    import cv2
    import docker
    import jwt
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    ```

### Docstring Formatting

We use 
[Pytorch's docstring formatting](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#docstring-type-formatting), 
which is based on the [Google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#381-docstrings).

### Linters and Formatters
To ensure code consistency, we use the following tools:

1. [isort](https://pycqa.github.io/isort/): Organizes the imports. Run:

   ```
   isort nexusml tests --profile google --line-length 120
   ```

2. [YAPF](https://github.com/google/yapf): Formats the code. Run:

   ```
   yapf --recursive --in-place nexusml tests
   ```

3. [Ruff](https://github.com/astral-sh/ruff): Quickly lints and fixes the code. Run:

   ```
   ruff check nexusml tests --ignore F405,F403,E731,F841,E741 --fix
   ```

   Note: Warnings F405, F403, E731, F841, and E741 are currently disabled due to coding practices under review. 
   These will be addressed in future releases.

4. [Pylint](https://pypi.org/project/pylint/): Thoroughly lints the code. We apply tailored configurations for different components, including 
   the main package, tests, and specific modules such as `nexusml/api/schemas`, `nexusml/database`, `nexusml/engine`, 
   and `nexusml/engine/models`. These configurations help avoid Pytest-related false positives and disable warnings 
   that don’t apply to these areas. Additionally, certain warnings are disabled at the file, class, 
   or line level when necessary. This level of granularity should only be used to address irrelevant warnings, 
   not to bypass best practices.
    
   To inspect any package with a `pylintrc` file, run:
   ```
   pylint --rcfile=<package-route>/pylintrc <package-route>
   ```
   Specifically, to inspect the main [`nexusml`](nexusml) package, run:
   ```
   pylint --rcfile=pylintrc nexusml
   ```
   To inspect the [`nexusml/api/schemas`](nexusml/api/schemas) package, run:
   ```
   pylint --rcfile=nexusml/api/schemas/pylintrc nexusml/api/schemas
   ```
   To inspect the [`nexusml/database`](nexusml/database) package, run:
   ```
   pylint --rcfile=nexusml/database/pylintrc nexusml/database
   ```
   To inspect the [`nexusml/engine`](nexusml/engine) package, run:
   ```
   pylint --rcfile=nexusml/engine/pylintrc nexusml/engine  
   ```
   To inspect the [`nexusml/engine/models`](nexusml/engine/models) package, run:
   ```
   pylint --rcfile=nexusml/engine/models/pylintrc nexusml/engine/models
   ```
   To inspect the [`tests`](tests) package, run: 
   ```
   pylint --rcfile=tests/pylintrc tests
   ```

   Note: Some warnings are currently disabled due to coding practices under review. 
   These will be addressed in future releases. Refer to each `pylintrc` for more details.

## Git

A well-structured Git repository and a clean history can help other developers understand the context, reason, and 
impact of changes made over time.

### Branch Naming

We follow the GitFlow workflow 
(detailed at [Atlassian GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)) 
for branching, with some additional prefixes to cover more cases.

Branch names should use the format `prefix/description`, where `prefix` reflects the type of work being done. 
The most common prefixes are:

- `feature/`: For new features or major changes. 
- `fix/`: For small bug fixes that are not critical. 
- `hotfix/`: For immediate fixes to production issues. 
- `release/`: For preparing a new production release. 
- `docs/`: For changes or updates to documentation. 
- `refactor/`: For code refactoring and improving code structure without changing its behavior. 
- `test/`: For adding or modifying tests.

Examples:

```
feature/add-user-authentication
bugfix/fix-password-reset-logic
docs/update-readme-with-installation-guide
refactor/restructure-auth-module
test/add-auth-module-tests
```

### Commits

We use the following message format:

```
<type>(<scope>): <short summary>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

1. **Type**
    - A brief label of the kind of change, e.g.:
        - `feat`: New feature.
        - `fix`: A bug fix.
        - `docs`: Documentation only changes.
        - `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc).
        - `refactor`: Refactoring code.
        - `perf`: A code change that improves performance.
        - `test`: Adding or modifying tests.
        - `chore`: Maintenance tasks, updates build tasks, etc.

2. **Scope (optional)**
    - A scope can be anything specifying the place of the commit change, e.g., `auth`, `moduleX`, `core`, etc.

3. **Short Summary (Header)**
    - Should be no longer than 50 characters.
    - Begin with a capital letter.
    - Use the imperative mood (e.g., "Add" instead of "Added" or "Adds").
    - Do not end with a period.

4. **Body**
    - Use the body to explain the "what" and "why" of a change, as opposed to the "how".
    - Wrap text at 72 characters.
    - Use bullets if needed.

5. **Footer**
    - Include any additional information or metadata, like breaking changes or references to issue tracker IDs.

Example commit message:

```
feat(auth): Add 2-factor authentication support
Implements 2FA to enhance security for user accounts. This will require
users to enter a code sent to their mobile device during login.
Resolves: #12345
```

Also, make sure to:

- **Use Imperative Mood**: Write in a command or instruction form. For example, "Add", "Fix", "Remove", "Update", 
  rather than "Added", "Fixed", "Removed", "Updated".
- **Be Clear and Descriptive**: Avoid vague messages like "Fix bugs" or "Update code". Instead, describe briefly what 
  you did and why.
- **Separate Subject from Body with a Blank Line**: If you're adding a body to the commit message, always separate it 
  from the summary with a blank line.
- **Reference Relevant Issues**: If the commit relates to an issue, reference it in the footer so it's clear what 
  problem or feature the commit addresses.
- **Avoid Side Effects**: Each commit should represent one logical change. If a commit fixes a bug and refactors code, 
  it would be better to split those changes into two separate commits.

### Pull Requests

The title and description of Pull Requests (PRs) should be informative and easy to understand at a glance.

#### Title

Format:

```
[Scope]: Short, descriptive title (less than 50 characters)
```

- **Scope**: Can be a module name, a filename, a broad area of the code, or a ticket/issue number. It provides context 
  for where the changes are happening or what they're related to.
  
- **Short, descriptive title**: Should briefly summarize the main change introduced by the PR. Keep it concise while 
  ensuring it remains meaningful.

Also, make sure to:

1. **Be Concise**: Aim for brevity while capturing the essence of the change.
2. **Use Imperative Mood**: Write in a command or instruction form. For example, "Add", "Fix", "Remove", "Update", 
   rather than "Added", "Fixed", "Removed", "Updated".
3. **Avoid Generic Phrases**: Avoid titles like "Fix bug" or "Update code". Be more specific.
4. **Reference Issues/ Tickets**: If the PR is related to a specific issue, mention the issue number in the title.
5. **Avoid Special Characters**: Avoid using special characters or punctuation marks.

Examples:

* `LoginModule: Fix password reset bug`
* `[#1234] Update user profile UI`
* `Docs: Add API endpoint documentation`
* `Backend: Migrate database to Postgres`

#### Description

Make sure you include at least the following sections in the Pull Request (PR) description:

1. *Background*: provide context information.
2. *Changes*: describe the changes proposed in the PR.

## Release Versioning

We use [Semantic Versioning](https://semver.org/), which gives a version number MAJOR.MINOR.PATCH incrementing the:

- MAJOR version when we make significant, breaking changes.
- MINOR version when we add functionality in a backwards compatible manner.
- PATCH version when we make backwards compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format. 
More specifically:

- `alpha`: the code is actively in development after the previous release and being tested internally by the developers.
- `beta`: the code is feature-complete and being tested by selected customers. *Beta testers* are people who actively 
  report issues of beta software. They are usually customers or representatives of prospective customers of the 
  organization that develops the software. Beta testers tend to volunteer their services free of charge but often 
  receive versions of the product they test, discounts on the release version, or other incentives.
- `rc`: a release candidate is a beta version with potential to be a stable product, which is ready to release unless 
  significant bugs emerge.
  
Pre-release numbers start from 0. For example: `0.1.0-alpha.0`, `0.1.0-alpha.1`, `0.1.0-alpha.2`, `0.1.0-beta.0`, 
`0.1.0-beta.1`, `0.1.0-beta.2`, `0.1.0-rc.0`, `0.1.0-rc.1`, `0.1.0-rc.2`.

## Release Notes

The [RELEASE.md](RELEASE.md) file contains the release notes, which are a crucial part of this project documentation. 
These notes serve as a comprehensive record of all changes, updates, and improvements made in each version of this 
project. They provide essential insights for both users and contributors, helping them understand the evolution and 
current state of the project. For consistency and clarity, the release notes are organized into distinct sections, 
each addressing a specific type of change. It is imperative that these sections are presented in the specific order 
shown below.

1. **Major Features**: This section highlights the most significant new features and enhancements introduced in the 
   release. These are substantial updates that have a notable impact on the project's functionality and user experience.
2. **Additional Features**: Here, we list other new features and improvements that have been added. While these may 
   not be as impactful as major features, they offer additional value and enhancements to the project.
3. **Breaking Changes**: Critical for users upgrading from previous versions, this section documents any changes that 
   break backward compatibility, including modifications that may require changes in existing setups or usage patterns.
4. **Behavioral Changes**: Detail any alterations in the software's behavior, such as changes in the business logic, 
   defaults, or functionality. These are non-breaking changes but may still affect the user experience.
5. **Bug Fixes**: Provide a list of resolved bugs in this release. Detailing the nature of the bugs and their 
   resolutions helps users understand the improvements and fixes made.
6. **Refactoring**: Describe code refactoring efforts, including reorganization, optimization, and other internal 
   improvements. These changes typically do not affect the software's external functionality but improve code quality and 
   maintainability.
7. **Style**: Outline changes made to the coding style or standards. This is crucial for contributors to maintain 
   consistency and quality in the codebase.
8. **Tests**: Report on the addition of new tests, modifications to existing tests, or enhancements in the testing 
   framework. This section underscores the commitment to software reliability and stability.
9. **Other Changes**: Include any miscellaneous changes that do not fit into the above categories but are significant, 
   such as documentation updates, dependency changes, or other minor modifications.

Each section should be meticulously updated with each new release to accurately reflect the latest developments. This 
structured approach ensures a clear and comprehensive documentation of the project's ongoing progress and changes.

## Abbreviation Guide

*Consider creating an abbreviation guide like [fastai Abbreviation Guide](https://docs.fast.ai/dev/abbr.html).*
