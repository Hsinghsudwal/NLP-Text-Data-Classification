FROM python:3.11-alpine

COPY . /app

WORKDIR /app

RUN pip install -r requirement.txt

CMD python application.py
