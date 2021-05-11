FROM python:3.7.8

RUN apt-get update
RUN pip install --upgrade pip

RUN apt-get install nano

WORKDIR /app

COPY ./requirements.txt requirements.txt

COPY . /app

EXPOSE 80 443
