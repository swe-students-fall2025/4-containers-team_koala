FROM python:3.10-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv && pipenv install --system --deploy

COPY . .

CMD ["python", "app.py"]
