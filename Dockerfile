# Load python container as basis
ARG PYTHON_VERSION=3.11.0
FROM python:${PYTHON_VERSION}-slim as base

# Set working directory
WORKDIR /repo

# Update package data base (install quarto, packages)
RUN apt-get update 
RUN apt-get install -y make wget libfontconfig zip
RUN wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.55/quarto-1.5.55-linux-amd64.deb
RUN dpkg -i quarto-1.5.55-linux-amd64.deb 
RUN apt-get install -f
RUN quarto install tinytex

# Install Python packages
RUN pip3 install jupyter

# Copy project files into the container
COPY . /repo

# Run make
CMD ls /repo
