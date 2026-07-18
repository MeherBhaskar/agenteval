FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/text_generation/server.py /app/tasks/text_generation/
COPY tasks/text_generation/task.json /app/tasks/text_generation/

WORKDIR /app/tasks/text_generation

CMD ["python", "server.py"]
