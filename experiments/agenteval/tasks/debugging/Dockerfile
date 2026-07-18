FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/debugging/server.py /app/tasks/debugging/
COPY tasks/debugging/task.json /app/tasks/debugging/

WORKDIR /app/tasks/debugging

CMD ["python", "server.py"]
