FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    plotly

COPY tasks/data_analysis /app/tasks/data_analysis
RUN pip install -e /app/tasks/data_analysis

WORKDIR /app/tasks/data_analysis

CMD ["python", "run_server.py"]
