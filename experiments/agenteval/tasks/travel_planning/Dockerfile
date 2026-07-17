FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    ortools \
    requests \
    pandas \
    numpy

COPY tasks/travel_planning /app/tasks/travel_planning
RUN pip install -e /app/tasks/travel_planning

WORKDIR /app/tasks/travel_planning

CMD ["python", "run_server.py"]
