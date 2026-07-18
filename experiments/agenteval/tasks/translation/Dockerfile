FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/translation/server.py /app/tasks/translation/
COPY tasks/translation/task.json /app/tasks/translation/

WORKDIR /app/tasks/translation

CMD ["python", "server.py"]
