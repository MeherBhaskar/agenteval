FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    gymnasium \
    numpy \
    pandas \
    matplotlib \
    tqdm \
    flask

COPY tasks/retail_operations/server.py /app/tasks/retail_operations/
COPY tasks/retail_operations/task.json /app/tasks/retail_operations/

WORKDIR /app/tasks/retail_operations

CMD ["python", "server.py"]