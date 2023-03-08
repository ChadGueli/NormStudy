FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN pip install boto3[crt] optuna pandas psutil pytorch-lightning \
    torchmetrics torchvision

COPY code /home/code
WORKDIR /home/code

RUN mkdir -p /data/in
RUN mkdir /data/out