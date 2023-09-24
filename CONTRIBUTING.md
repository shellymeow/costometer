# Setting up an environment for local development

1. First, set up a virtual environment in your preferred way. 
This should work on any Mac/OSX system with virtualenv on it. See here for how to install virtualenv/commands on Windows: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/.

```
#create virtual environment
python3 -m venv env
#activate virtual environment
source env/bin/activate
```

2. Install the dev requirements (includes editable version of package):

```
python -m pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:

```
pre-commit install
```

4. Make your changes and commit, making any change suggested by flake8 and other hooks.

5. Test that everything passes (or is skipped) with tox before pushing to the remote:

```
tox
```

# How to submit changes

Send a pull request, after following the steps to set up an environment for local development! We don't have a template for this, but providing details of what changed will help others approve your changes more quickly.

# How to report a bug or submit a feature request

Please follow the issue templates found in `.github/ISSUE_TEMPLATE`.
