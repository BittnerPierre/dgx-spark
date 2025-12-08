docker build -t custom-vllm-0.12 .

docker run --rm -it --gpus all custom-vllm-0.12 bash

python -c "import vllm; print(vllm.**version**)"

-> shoud display 0.12.0

python -c "import mistral_common, transformers; print('mistral_common', mistral_common.**version**, 'transformers', transformers.**version**)"

-> mistral_common 1.8.6 transformers 4.57.3

docker run --rm --gpus all \ --ipc=host \ --ulimit memlock=-1 \ --ulimit stack=67108864 \ -p 8001:8000 \ custom-vllm-0.12 \ vllm serve mistralai/Ministral-3-3B-Instruct-2512 \ --gpu-memory-utilization 0.9
