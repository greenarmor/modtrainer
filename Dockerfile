
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip3 install torch transformers datasets accelerate peft bitsandbytes trl sentencepiece
WORKDIR /app
COPY . /app
CMD ["python3", "training/finetune_lora.py"]
