# FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow:nightly-gpu
RUN pip3 install matplotlib tqdm scipy pandas numpy sklearn
RUN mkdir -p /home/FFRNeuralNet

WORKDIR /home/FFRNeuralNet
