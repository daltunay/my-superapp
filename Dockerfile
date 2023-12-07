FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install "poetry==1.7.0" \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-dev

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]
