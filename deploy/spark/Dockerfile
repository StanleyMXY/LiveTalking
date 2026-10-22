# LiveTalking on the DGX Spark (aarch64, GB10 / Blackwell, driver 580 = CUDA 13.0).
#
# The base image carries the stock PyTorch 2.10.0+cu130 wheels (cuDNN 9.15). NVIDIA's NGC builds
# 26.02/26.03 (cuDNN 9.19/9.20) were measured on this machine and have no cuDNN engine for
# transposed convolutions, which wav2lip uses, so they are not used here.
ARG BASE_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:v0.19.0-cu130-ubuntu2404
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1

# The base image starts a vLLM server by default.
ENTRYPOINT []

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY deploy/spark/constraints.txt deploy/spark/requirements.txt deploy/spark/sadtalker-requirements.txt /tmp/
# Flask needs blinker>=1.9; the base image has the Debian-packaged 1.7.0, which pip cannot uninstall.
# basicsr 1.4.2 (imported by gfpgan, which SadTalker imports) uses torchvision.transforms.functional_tensor,
# removed from torchvision; the sed points it at the function's new home.
RUN pip install --ignore-installed "blinker>=1.9"  && pip install -c /tmp/constraints.txt -r /tmp/requirements.txt  && pip install -c /tmp/constraints.txt -r /tmp/sadtalker-requirements.txt  && pip install --no-build-isolation -c /tmp/constraints.txt basicsr==1.4.2 gfpgan==1.3.8  && sed -i "s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/"       "$(python3 -c 'import importlib.util;print(importlib.util.find_spec("basicsr").submodule_search_locations[0])')/data/degradations.py"

WORKDIR /app
COPY . /app

CMD ["python3", "app.py", "--transport", "webrtc", "--model", "wav2lip", "--avatar_id", "wav2lip256_avatar1"]
