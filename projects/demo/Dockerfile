FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY demo.py .
COPY data.json .

EXPOSE 8501

CMD streamlit run demo.py
