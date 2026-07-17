FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    playwright \
    beautifulsoup4 \
    lxml

COPY tasks/web_shopping /app/tasks/web_shopping
RUN pip install -e /app/tasks/web_shopping

WORKDIR /app/tasks/web_shopping

CMD ["python", "run_server.py"]
