FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy

COPY tasks/sentiment_analysis/server.py /app/tasks/sentiment_analysis/
COPY tasks/sentiment_analysis/task.json /app/tasks/sentiment_analysis/

WORKDIR /app/tasks/sentiment_analysis

CMD ["python", "server.py"]
