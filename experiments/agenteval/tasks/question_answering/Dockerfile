FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/question_answering/server.py /app/tasks/question_answering/
COPY tasks/question_answering/task.json /app/tasks/question_answering/

WORKDIR /app/tasks/question_answering

CMD ["python", "server.py"]
