# vim: ft=mako
# FROM 9bde45dfef284488a8adeb17a1b173d4.azurecr.io/mttl:576b7db as base

FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 as base
FROM singularitybase.azurecr.io/installer/base/singularity-installer:adsbrain20230905T133621821 as installer
FROM singularitybase.azurecr.io/validations/base/singularity-tests:20231108T120134067 as validator

FROM base

# install DLN package and setup data
WORKDIR /app
RUN git clone -b deep-language-networks-soft-prompts https://github.com/microsoft/deep-language-networks
WORKDIR /app/deep-language-networks
RUN pip install -e .
RUN bash scripts/setup_data.sh

# detect whether someone ran the installer already
RUN [ ! -d /opt/microsoft/_singularity/installations ] || touch /tmp/adapted_for_sing && echo "Looks like this image already has the singularity installer, not running it again."

# get the installation scripts
COPY --from=installer /installer /opt/microsoft/_singularity/installations/

# install components required for running this image in Singularity
RUN [ -e /tmp/adapted_for_sing ] || /opt/microsoft/_singularity/installations/singularity/installer.sh

# get the validation scripts
COPY --from=validator /validations /opt/microsoft/_singularity/validations/

# if you modify /etc/profile.d e.g. to activate a conda environment automatically,
# you should condition activation on the following variable *not* being set
ARG SINGULARITY_DOCKER_BUILD=1


# set validation arguments for expected use of this image
ENV SINGULARITY_IMAGE_ACCELERATOR=NVIDIA

# run the validation
RUN /opt/microsoft/_singularity/validations/validator.sh
