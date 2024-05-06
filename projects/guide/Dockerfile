FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt .

RUN apt update && apt install -y git
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY guided_search.py .
COPY templates ./templates

CMD streamlit run app.py
