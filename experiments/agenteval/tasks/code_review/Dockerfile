FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/code_review/server.py /app/tasks/code_review/
COPY tasks/code_review/task.json /app/tasks/code_review/

WORKDIR /app/tasks/code_review

CMD ["python", "server.py"]
