FROM python:3.12.0

RUN apt-get update && apt-get install -y libgl1

WORKDIR /app

COPY ../requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY ../ /app/