#!/bin/bash
# Quick verification script for AgentEval infrastructure

set -e

echo "=== AgentEval Infrastructure Check ==="

# 1. Check tasks load
echo "1. Testing task loading..."
python3 -c "
import yaml
from pathlib import Path
tasks_file = Path('tasks/core/tasks.yaml')
docs = list(yaml.safe_load_all(open(tasks_file)))
print(f'   Loaded {len(docs)} tasks:')
for doc in docs:
    print(f'   - {doc[\"task_id\"]} ({doc[\"category\"]})')
"

# 2. Check runner imports
echo ""
echo "2. Testing runner imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
from runner import TaskSpec, ReActAgent, EvaluationRunner
print('   All imports successful')
"

# 3. Check Docker availability
echo ""
echo "3. Checking Docker..."
docker --version
if docker info >/dev/null 2>&1; then
    echo "   Docker daemon running ✓"
else
    echo "   Docker daemon NOT running - start Docker Desktop or 'sudo systemctl start docker'"
fi

# 4. Check base images (will need to be built)
echo ""
echo "4. Base images needed:"
echo "   - agenteval/base:latest"
echo "   - agenteval/fact-qa:v1.0"
echo "   - agenteval/retail-sim:v1.0"
echo "   - agenteval/code-gen:v1.0"
echo "   - agenteval/web-shopping:v1.0"
echo "   - agenteval/travel-planning:v1.0"
echo ""
echo "   Build with: docker build -t agenteval/base:latest environments/base/"
echo "   (Dockerfiles not included in this minimal setup)"

echo ""
echo "=== Infrastructure Check Complete ==="
echo ""
echo "Next steps:"
echo "1. Start Docker daemon if not running"
echo "2. Build base Docker images"
echo "3. Run: python runner.py --agent react --model gpt-4o-mini --split dev --episodes 1"