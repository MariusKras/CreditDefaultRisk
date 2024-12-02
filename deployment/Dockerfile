FROM python:3.12
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
COPY model_for_deployment.pkl /app/model_for_deployment.pkl
EXPOSE 8080
CMD ["uvicorn", "fastapi_model_server:app", "--host", "0.0.0.0", "--port", "8080"]