FROM agenteval/base:latest

RUN pip install --no-cache-dir \
    ortools \
    requests \
    pandas \
    numpy \
    flask

COPY tasks/travel_planning/server.py /app/tasks/travel_planning/
COPY tasks/travel_planning/task.json /app/tasks/travel_planning/

WORKDIR /app/tasks/travel_planning

CMD ["python", "server.py"]