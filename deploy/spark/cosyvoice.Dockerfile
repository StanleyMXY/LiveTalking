# CosyVoice3 voice-cloning TTS server (cosyvoice_api_server.py, port 8091) on the DGX Spark.
# Same base as the LiveTalking image: stock PyTorch 2.10.0+cu130 (NVIDIA's NGC cuDNN 9.19/9.20 builds
# fail on this GPU, see Dockerfile). Only the Python dependencies live in the image; the CosyVoice
# checkout and the model weights are mounted (see compose.yaml).
ARG BASE_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:v0.19.0-cu130-ubuntu2404
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1

ENTRYPOINT []

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ffmpeg libsndfile1 sox \
 && rm -rf /var/lib/apt/lists/*

COPY cosyvoice-constraints.txt cosyvoice-requirements.txt /tmp/
RUN pip install --ignore-installed "blinker>=1.9" \
 && pip install -c /tmp/cosyvoice-constraints.txt -r /tmp/cosyvoice-requirements.txt

# pyworld pins numpy<2, which would pull numpy back to 1.26 and break pyarrow (needs numpy>=2);
# it is built against the installed numpy, so it is added without its dependencies.
RUN pip install --no-deps pyworld

WORKDIR /opt/cosyvoice
CMD ["python3", "cosyvoice_api_server.py", "--port", "8091", "--model_dir", "/model"]
