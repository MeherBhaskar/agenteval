FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/sql_generation/server.py /app/tasks/sql_generation/
COPY tasks/sql_generation/task.json /app/tasks/sql_generation/

WORKDIR /app/tasks/sql_generation

CMD ["python", "server.py"]
