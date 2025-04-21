FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-tk \
    openssh-client \
    build-essential \
    git

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1 -y

RUN pip install gradio==5.0.2 \
                litserve==0.2.3 \
                opencv-python \
                pydantic==2.9.2 \
                "sam-2 @ git+https://github.com/facebookresearch/segment-anything-2@c2ec8e14a185632b0a5d8b161928ceb50197eddc" 'supervision==0.23.0; python_full_version>"3.10"' \
                transformers==4.44.2 \ 
                uvloop==0.20.0 \ 
                torch==2.4.1 \
                torchvision==0.19.1 \
                scikit-learn

COPY . /lang-segment-anything
WORKDIR /lang-segment-anything
EXPOSE 8000

# Entry point
# CMD ["python3", "app.py"]