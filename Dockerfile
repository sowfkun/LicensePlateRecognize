FROM python:3.7.8

RUN apt-get update
RUN pip install --upgrade pip

RUN apt-get install nano

WORKDIR /app
COPY . /app

COPY ./requirements.txt requirements.txt
#COPY ./install_req.sh install_req.sh
RUN pip install -r ./requirements.txt

EXPOSE 80 443

#ENTRYPOINT ["./install_req.sh"]
