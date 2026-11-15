FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_xgboost.bin", "./"]

EXPOSE 9000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9000", "predict:app"]