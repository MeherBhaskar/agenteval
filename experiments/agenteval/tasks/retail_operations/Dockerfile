FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    gymnasium \
    numpy \
    pandas \
    matplotlib \
    tqdm

COPY tasks/retail_operations /app/tasks/retail_operations
RUN pip install -e /app/tasks/retail_operations

WORKDIR /app/tasks/retail_operations

CMD ["python", "run_server.py"]
