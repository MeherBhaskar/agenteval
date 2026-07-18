FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/summarization/server.py /app/tasks/summarization/
COPY tasks/summarization/task.json /app/tasks/summarization/

WORKDIR /app/tasks/summarization

CMD ["python", "server.py"]
