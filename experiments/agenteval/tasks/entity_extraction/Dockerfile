FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/entity_extraction/server.py /app/tasks/entity_extraction/
COPY tasks/entity_extraction/task.json /app/tasks/entity_extraction/

WORKDIR /app/tasks/entity_extraction

CMD ["python", "server.py"]
