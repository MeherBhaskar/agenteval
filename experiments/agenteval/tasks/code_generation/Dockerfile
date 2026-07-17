FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    pytest \
    black \
    isort \
    flake8 \
    mypy \
    requests \
    httpx

COPY tasks/code_generation /app/tasks/code_generation
RUN pip install -e /app/tasks/code_generation

WORKDIR /app/tasks/code_generation

CMD ["python", "run_server.py"]
