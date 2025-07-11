FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Setting working directory
WORKDIR /usr/app

# Base utilities
RUN apt update && \
    apt install -y python3.10-venv python3-pip git &&\
    apt-get clean
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip and the package
# for now pip 25 is not supported 
RUN pip install --upgrade "pip<25" 
COPY . ./
RUN pip install .

#RUN python octopi/entry_points/run_optuna.py
ENTRYPOINT ["python3", "octopi/entry_points/run_optuna.py"]