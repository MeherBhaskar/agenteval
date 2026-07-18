FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/api_integration/server.py /app/tasks/api_integration/
COPY tasks/api_integration/task.json /app/tasks/api_integration/

WORKDIR /app/tasks/api_integration

CMD ["python", "server.py"]
