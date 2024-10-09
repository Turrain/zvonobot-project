FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN python3.9 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
COPY main.py .

EXPOSE 8002
CMD ["python", "main.py"]