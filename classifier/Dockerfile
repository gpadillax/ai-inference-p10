# Copyright 2022 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM ubi8/ubi-minimal
ARG MINICONDA_REPO="https://repo.anaconda.com/miniconda"
ARG MINICONDA_VERSION="Miniconda3-py39_4.12.0-Linux-ppc64le.sh"
ARG RESNET_MODEL_REPO="https://github.com/onnx/models/raw/main/vision/classification/resnet/model"
ARG SYNSET_REPO="https://raw.githubusercontent.com/onnx/models/main/vision/classification"

RUN microdnf install -y \
    wget  \
    bzip2  \
    && microdnf clean all

WORKDIR /opt/miniconda_installer

# Install conda, we use miniconda because it uses a smaller set of 
# dependencies. We also install in slient mode using these directions
# https://docs.anaconda.com/anaconda/install/silent-mode/
RUN wget "$MINICONDA_REPO/$MINICONDA_VERSION" -O installer.sh && \
    chmod u+x ./installer.sh && \
    ./installer.sh -b -p /opt/miniconda && \
    rm ./installer.sh 

WORKDIR /opt/resnet

# Create the Environment
COPY onnxruntime_env.yaml .
RUN bash -c 'eval "$(/opt/miniconda/bin/conda shell.bash hook)" &&\
             conda init  && \
             conda env create -f onnxruntime_env.yaml && \
             conda clean --all -f -y'

# Download the model file and class labels
RUN wget "${RESNET_MODEL_REPO}/resnet101-v2-7.onnx"
RUN wget "${SYNSET_REPO}/synset.txt"

# Update python files
COPY python/*.py .
COPY *.sh .

# The container should not run under root
# Openshift will run the container under a non-root 
# user id that will have group 0 (inside the container).
# Here we make our application files readable by the
# root group, and owned by user 1001.
# We set the container to run under 1001, to be 
# compatitable with other K8S distributions.
RUN chown -R 1001:0 /opt && \
    chmod -R g=u /opt

USER 1001

# 5000 is the port specified in app.py
EXPOSE 5000

# Sets the command for the container
CMD ["/opt/resnet/startup.sh"]
