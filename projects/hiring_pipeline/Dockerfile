FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Initialize database
RUN python -c "import database; database.init_db()"

EXPOSE 8000

CMD ["uvicorn", "api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
