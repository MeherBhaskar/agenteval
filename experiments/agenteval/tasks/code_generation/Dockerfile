FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pytest

COPY tasks/code_generation/server.py /app/tasks/code_generation/
COPY tasks/code_generation/task.json /app/tasks/code_generation/

WORKDIR /app/tasks/code_generation

CMD ["python", "server.py"]