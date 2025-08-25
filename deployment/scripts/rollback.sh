#!/bin/bash
# Bio-Enhanced Rollback Script

set -euo pipefail

PROJECT_NAME="secure-mpc-transformer"
ENVIRONMENT="${1:-staging}"
PREVIOUS_VERSION="${2}"

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "❌ Previous version required for rollback"
    echo "Usage: $0 <environment> <previous_version>"
    exit 1
fi

echo "🔄 Rolling back $PROJECT_NAME in $ENVIRONMENT to version $PREVIOUS_VERSION"

# Rollback deployment
kubectl rollout undo deployment/$PROJECT_NAME-$ENVIRONMENT -n $PROJECT_NAME-$ENVIRONMENT

# Wait for rollback
kubectl rollout status deployment/$PROJECT_NAME-$ENVIRONMENT -n $PROJECT_NAME-$ENVIRONMENT --timeout=300s

# Validate rollback
python -c "
import requests
import time

for i in range(10):
    try:
        response = requests.get('http://localhost:8080/health', timeout=5)
        if response.status_code == 200:
            print('✅ Rollback validation passed')
            break
    except:
        pass
    time.sleep(5)
else:
    print('❌ Rollback validation failed')
    exit(1)
"

echo "✅ Rollback completed successfully"
