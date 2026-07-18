FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/data_analysis/server.py /app/tasks/data_analysis/
COPY tasks/data_analysis/task.json /app/tasks/data_analysis/

WORKDIR /app/tasks/data_analysis

CMD ["python", "server.py"]
