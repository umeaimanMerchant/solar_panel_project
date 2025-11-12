# Dockerfile - builds a simple image that runs a FastAPI app (expects main.py with `app = FastAPI()`)

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install runtime deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app sources
COPY . .

EXPOSE 8000

# Expect a FastAPI instance named `app` in main.py (module path: main:app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]