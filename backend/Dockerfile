FROM python:3.10-slim

RUN apt update && apt install -y binutils

WORKDIR /app
ADD ./app/requirements.txt /app/
RUN pip install -r requirements.txt
ADD ./app/. /app/


CMD ["python3", "main.py"]
