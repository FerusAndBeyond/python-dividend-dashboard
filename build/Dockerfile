FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src .
CMD ["solara", "run", "app.py", "--host=0.0.0.0", "--production"]