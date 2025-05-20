FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Setting working directory
WORKDIR /usr/app

# Base utilities
RUN apt update && \
    apt install -y python3.10-venv python3-pip git &&\
    apt-get clean
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip and the package
RUN pip install --upgrade pip
COPY . ./
RUN pip install .

#RUN python src/octopi/segmentations_from_picks.py
ENTRYPOINT ["python3", "src/octopi/optuna_pl_ddp.py"]