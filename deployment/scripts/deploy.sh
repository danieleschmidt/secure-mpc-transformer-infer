#!/bin/bash
# Bio-Enhanced Production Deployment Script

set -euo pipefail

PROJECT_NAME="secure-mpc-transformer"
VERSION="1.0.0"
ENVIRONMENT="${1:-staging}"

echo "🚀 Starting Bio-Enhanced Deployment for $ENVIRONMENT"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    echo "❌ Invalid environment: $ENVIRONMENT"
    echo "Valid options: development, staging, production"
    exit 1
fi

# Pre-deployment checks
echo "🔍 Running pre-deployment quality gates..."
python bio_comprehensive_quality_gates.py
QUALITY_EXIT_CODE=$?

if [ $QUALITY_EXIT_CODE -ne 0 ]; then
    echo "❌ Quality gates failed. Deployment aborted."
    exit $QUALITY_EXIT_CODE
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t $PROJECT_NAME:$VERSION .

# Apply Kubernetes manifests
echo "📋 Applying Kubernetes manifests for $ENVIRONMENT..."
kubectl apply -f k8s/$ENVIRONMENT/

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/$PROJECT_NAME-$ENVIRONMENT -n $PROJECT_NAME-$ENVIRONMENT --timeout=600s

# Health checks
echo "🏥 Running post-deployment health checks..."
python -c "
import asyncio
import time
import requests

async def health_check():
    for i in range(30):
        try:
            response = requests.get('http://localhost:8080/health', timeout=10)
            if response.status_code == 200:
                print('✅ Health check passed')
                return True
        except:
            pass
        print(f'⏳ Health check {i+1}/30...')
        await asyncio.sleep(10)
    return False

if not asyncio.run(health_check()):
    print('❌ Health checks failed')
    exit(1)
"

# Run bio-evolution validation
echo "🧬 Validating bio-evolution system..."
python pure_python_bio_demo.py

echo "✅ Deployment completed successfully!"
echo "🔗 Application URL: http://localhost:8080"
echo "📊 Monitoring: http://localhost:9090"
