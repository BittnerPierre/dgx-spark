# Dockerfile.vllm012
FROM nvcr.io/nvidia/vllm:25.11-py3

RUN pip install --upgrade pip && \
    pip install -U "vllm>=0.12.0" \
      --extra-index-url https://wheels.vllm.ai/nightly && \
    pip install \
      "mistral-common>=1.8.6" \
      "transformers==4.57.3" \
      -U "huggingface_hub"

EXPOSE 8000
WORKDIR /workspace
CMD ["bash"]