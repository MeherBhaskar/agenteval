FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    datasets \
    beir \
    sentence-transformers \
    rank-bm25 \
    faiss-cpu \
    flask

COPY tasks/fact_qa/server.py /app/tasks/fact_qa/
COPY tasks/fact_qa/task.json /app/tasks/fact_qa/

WORKDIR /app/tasks/fact_qa

CMD ["python", "server.py"]