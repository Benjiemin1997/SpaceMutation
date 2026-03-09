FROM python:3.10-slim

WORKDIR /workspace

COPY fault_container_demo/requirements.txt /workspace/fault_container_demo/requirements.txt
RUN pip install --no-cache-dir -r /workspace/fault_container_demo/requirements.txt

COPY model_unet.py /workspace/model_unet.py
COPY UNet.pth /workspace/UNet.pth
COPY mut /workspace/mut
COPY fault_container_demo/app /workspace/fault_container_demo/app

ENV PROJECT_ROOT=/workspace
ENV MODEL_PATH=/workspace/UNet.pth

CMD ["python", "-m", "fault_container_demo.app.runner"]