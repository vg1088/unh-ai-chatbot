#syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

WORKDIR /

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
VOLUME /chatbot_data/
EXPOSE 1896:1896


CMD [ "python3",  "chatbot_fat.py"]