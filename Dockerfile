# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /fetch-ml

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir

COPY . .
COPY /templates/app.html /templates/app.html

RUN python3 train_linear_model.py

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]