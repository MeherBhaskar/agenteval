FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    datasets \
    beir \
    sentence-transformers \
    rank-bm25 \
    faiss-cpu

COPY tasks/fact_qa /app/tasks/fact_qa

WORKDIR /app/tasks/fact_qa

CMD ["python", "run_server.py"]
