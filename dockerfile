FROM ubuntu:jammy
RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install build-essential -y
WORKDIR /
COPY Machine-Learning-Final /Machine-Learning-Final
WORKDIR /Machine-Learning-Final
RUN pip3 install notebook
RUN pip3 install -r requirements.txt