# Installation Guide

## Quick Installation

octopi is available on PyPI and can be installed using pip:

```bash
pip install octopi
```

## Development Installation

If you want to contribute to octopi or need the latest development version, you can install from source:

```bash
git clone https://github.com/chanzuckerberg/octopi.git
cd octopi
pip install -e .
```

## MLflow Setup

To use MLflow for experiment tracking, create a `.env` file in your project root with the following content:

```bash
MLFLOW_TRACKING_USERNAME = <Your_CZ_email>
MLFLOW_TRACKING_PASSWORD = <Your_mlflow_access_token>
```

You can get a CZI MLflow access token from [here](https://mlflow.cw.use4-prod.si.czi.technology/api/2.0/mlflow/users/access-token).

## Verification

To verify your installation, run:

```bash
python -c "import octopi; print(octopi.__version__)"
```

## Next Steps

- [Import Your Data](data-import.md) - Learn how to import your tomograms into a copick project.
- [Quick Start Guide](quickstart.md) - Run your first particle picking experiment. 
- [Learn the API](../user-guide/api-tutorial.md) - Integrate Octopi into your Python workflows. 
