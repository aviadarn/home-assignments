FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY streamlin_app.py .

CMD ["streamlit", "run", "streamlin_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
