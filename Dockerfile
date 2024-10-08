FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Setting working directory
WORKDIR /usr/app

# Base utilities
RUN apt update && apt install -y python3-pip git
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip and Poetry
RUN pip install --upgrade pip
RUN pip install git+https://github.com/copick/copick-utils.git
ENV POETRY_VERSION=1.8.3
RUN python3 -m pip install --no-cache-dir poetry==$POETRY_VERSION

# Install dependencies
COPY pyproject.toml ./
# Set Poetry to not use virtual environments
ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry install

# Install package
COPY . ./
RUN pip install .

# run cmd after the build
CMD ["python3", "src/model_explore/train.py"]
