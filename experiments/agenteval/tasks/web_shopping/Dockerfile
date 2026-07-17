FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    flask \
    beautifulsoup4 \
    lxml

COPY tasks/web_shopping/server.py /app/tasks/web_shopping/
COPY tasks/web_shopping/task.json /app/tasks/web_shopping/

WORKDIR /app/tasks/web_shopping

CMD ["python", "server.py"]