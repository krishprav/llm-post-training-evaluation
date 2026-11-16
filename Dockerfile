FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY post_training_lab post_training_lab
COPY eval_harness eval_harness
COPY tests tests
COPY configs configs
COPY data data
RUN mkdir -p reports artifacts
CMD ["python", "-m", "pytest", "-q"]
