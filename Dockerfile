FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY diplomatic_crisis_env/ ./diplomatic_crisis_env/
RUN pip install -e .
EXPOSE 7860
CMD ["uvicorn", "diplomatic_crisis_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
