FROM tensorflow/tensorflow:latest-gpu
RUN pip3 install matplotlib tqdm scipy pandas numpy sklearn
RUN mkdir -p /home/deepffr
WORKDIR /home/deepffr
