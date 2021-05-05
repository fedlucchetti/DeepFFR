FROM tensorflow/tensorflow:latest-gpu
RUN apt-get install -y python3.8
RUN apt-get update -y
RUN pip3 install matplotlib tqdm scipy pandas numpy sklearn
RUN mkdir -p /home/deepffr
WORKDIR /home/deepffr
