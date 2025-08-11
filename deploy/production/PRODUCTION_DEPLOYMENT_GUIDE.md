# Production Deployment Guide

This guide provides comprehensive instructions for deploying the Secure MPC Transformer with Quantum-Inspired Task Planning to production environments.

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+ with Docker Compose v2
- OR Kubernetes 1.24+ with kubectl configured
- SSL certificates (Let's Encrypt recommended)
- Monitoring infrastructure (optional but recommended)

### Docker Compose Deployment (Recommended for smaller deployments)

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/secure-mpc-transformer-infer.git
cd secure-mpc-transformer-infer/deploy/production

# Create secrets directory
mkdir -p secrets
echo "your_secure_db_password" > secrets/db_password.txt
echo "your_secure_grafana_password" > secrets/grafana_password.txt

# Create configuration directory
mkdir -p config nginx/ssl monitoring postgres

# Deploy the stack
docker compose -f docker-compose.production.yml up -d
```

### Kubernetes Deployment (Recommended for larger deployments)

```bash
# Apply the production manifest
kubectl apply -f production_deployment_manifest.yaml

# Verify deployment
kubectl get pods -n secure-mpc-transformer
kubectl get services -n secure-mpc-transformer
```

## 📋 Detailed Deployment Options

### Option 1: Docker Compose with Full Monitoring Stack

This deployment includes:
- Secure MPC Transformer API (3 replicas)
- PostgreSQL database with automated backups
- Redis for caching and session storage
- NGINX reverse proxy with SSL termination
- Prometheus + Grafana for monitoring
- AlertManager for alerting
- Loki + Promtail for log aggregation
- Jaeger for distributed tracing

**Advantages:**
- Complete monitoring and observability stack
- Easy to manage and update
- Suitable for medium-scale deployments (10-1000 concurrent users)
- Lower infrastructure complexity

**Resource Requirements:**
- Memory: 16GB+ RAM
- CPU: 8+ cores
- Storage: 100GB+ SSD
- Network: 1Gbps+

### Option 2: Kubernetes with Auto-Scaling

This deployment includes:
- Horizontal Pod Autoscaler (3-50 replicas)
- Network policies for security
- Pod disruption budgets for availability
- Ingress with SSL termination
- Service mesh integration ready

**Advantages:**
- Enterprise-grade scalability
- Built-in high availability
- Advanced security policies
- Cloud-native deployment
- Suitable for large-scale deployments (1000+ concurrent users)

**Resource Requirements (per node):**
- Memory: 32GB+ RAM
- CPU: 16+ cores
- Storage: 500GB+ SSD
- Network: 10Gbps+

### Option 3: Hybrid Cloud Deployment

For maximum security and compliance:

```bash
# Deploy control plane in private cloud
kubectl apply -f production_deployment_manifest.yaml --context=private-cluster

# Deploy compute nodes in public cloud with specific taints
kubectl taint nodes gpu-node-1 high-compute=true:NoSchedule
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `QUANTUM_PLANNING_ENABLED` | Enable quantum-inspired optimization | `true` | Yes |
| `ENHANCED_SECURITY_ENABLED` | Enable advanced security features | `true` | Yes |
| `AUTO_SCALING_ENABLED` | Enable automatic scaling | `true` | No |
| `PERFORMANCE_OPTIMIZATION_ENABLED` | Enable ML-based optimization | `true` | No |
| `LOG_LEVEL` | Logging verbosity | `INFO` | No |
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |

### Security Configuration

#### SSL/TLS Setup

1. **Production SSL Certificates:**
```bash
# Using Let's Encrypt (recommended)
certbot certonly --dns-route53 -d api.your-domain.com

# Copy certificates
cp /etc/letsencrypt/live/api.your-domain.com/fullchain.pem nginx/ssl/
cp /etc/letsencrypt/live/api.your-domain.com/privkey.pem nginx/ssl/
```

2. **Security Headers (NGINX):**
```nginx
# nginx/nginx.conf
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

#### Database Security

```sql
-- postgres/init.sql
CREATE USER mpc_user WITH ENCRYPTED PASSWORD 'your_secure_password';
CREATE DATABASE mpc_transformer OWNER mpc_user;
GRANT CONNECT ON DATABASE mpc_transformer TO mpc_user;
REVOKE ALL ON SCHEMA public FROM public;
GRANT ALL ON SCHEMA public TO mpc_user;
```

### Performance Tuning

#### Resource Allocation

**For High-Throughput Scenarios (>1000 RPS):**
```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

**For Low-Latency Scenarios (<100ms P95):**
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "6Gi"
    cpu: "3000m"
```

#### Auto-Scaling Parameters

```yaml
# Aggressive scaling for varying load
minReplicas: 5
maxReplicas: 100
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 60  # Scale up earlier
```

## 📊 Monitoring and Observability

### Metrics Endpoints

| Endpoint | Description | Port |
|----------|-------------|------|
| `/metrics` | Prometheus metrics | 9090 |
| `/health` | Health check | 8080 |
| `/health/live` | Liveness probe | 8080 |
| `/health/ready` | Readiness probe | 8080 |

### Key Metrics to Monitor

1. **Performance Metrics:**
   - `http_request_duration_seconds`
   - `http_requests_total`
   - `mpc_computation_time_seconds`
   - `quantum_optimization_efficiency`

2. **Security Metrics:**
   - `security_threats_detected_total`
   - `validation_failures_total`
   - `circuit_breaker_state`

3. **System Metrics:**
   - `process_cpu_seconds_total`
   - `process_memory_bytes`
   - `go_memstats_alloc_bytes`

### Alerting Rules

```yaml
# monitoring/rules/alerts.yml
groups:
- name: mpc-transformer
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, http_request_duration_seconds) > 2
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      
  - alert: SecurityThreatDetected
    expr: increase(security_threats_detected_total[5m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Security threat detected"
```

### Dashboard Setup

Access Grafana at `http://your-domain:3000` with:
- Username: `admin`
- Password: (from `secrets/grafana_password.txt`)

Pre-configured dashboards:
- **MPC Transformer Overview:** System health and performance
- **Security Dashboard:** Threat detection and validation metrics
- **Quantum Planning:** Optimization efficiency and scaling metrics

## 🛡️ Security Considerations

### Network Security

1. **Firewall Rules:**
```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw deny incoming
ufw enable
```

2. **Network Policies (Kubernetes):**
```yaml
# Included in production_deployment_manifest.yaml
# - Ingress traffic only from nginx-ingress
# - Egress traffic restricted to DNS and HTTPS
# - Internal communication allowed within namespace
```

### Secret Management

1. **Docker Secrets:**
```bash
# Use Docker secrets for sensitive data
echo "db_password" | docker secret create db_password -
echo "api_key" | docker secret create api_key -
```

2. **Kubernetes Secrets:**
```bash
# Create secrets from files
kubectl create secret generic mpc-transformer-secrets \
  --from-file=db-password=./secrets/db_password.txt \
  --from-file=api-key=./secrets/api_key.txt
```

### Security Scanning

```bash
# Container vulnerability scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image securempc/transformer-inference:v0.2.0-quantum

# Kubernetes security scanning
kubectl run kube-bench --rm -ti --restart=Never \
  --image=aquasec/kube-bench:latest -- --version 1.24
```

## 🚀 Deployment Verification

### Health Check Script

```bash
#!/bin/bash
# deploy/production/verify-deployment.sh

API_URL="https://api.your-domain.com"

echo "🔍 Verifying deployment..."

# Health check
if curl -f -s "${API_URL}/health" > /dev/null; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# API functionality test
response=$(curl -s -X POST "${API_URL}/api/v1/validate" \
  -H "Content-Type: application/json" \
  -d '{"input": "test input", "type": "text"}')

if echo "$response" | jq -e '.status == "success"' > /dev/null; then
    echo "✅ API functionality verified"
else
    echo "❌ API functionality test failed"
    exit 1
fi

# Performance test
latency=$(curl -w "%{time_total}" -s -o /dev/null "${API_URL}/health")
if (( $(echo "$latency < 0.5" | bc -l) )); then
    echo "✅ Latency check passed (${latency}s)"
else
    echo "⚠️ High latency detected (${latency}s)"
fi

echo "🎉 Deployment verification completed successfully!"
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 https://api.your-domain.com/health

# Using k6 for more advanced testing
k6 run --vus 50 --duration 30s load-test.js
```

## 📈 Scaling Guidelines

### Horizontal Scaling

**Automatic Scaling (HPA):**
- Scales based on CPU, memory, and custom metrics
- Quantum coherence metrics for optimal performance
- Threat level-based scaling for security events

**Manual Scaling:**
```bash
# Docker Compose
docker compose -f docker-compose.production.yml up -d --scale mpc-transformer-api=5

# Kubernetes
kubectl scale deployment mpc-transformer-api --replicas=10
```

### Vertical Scaling

**Memory-Intensive Workloads:**
- Increase memory limits for large model inference
- Enable GPU memory optimization
- Configure swap if necessary (not recommended for production)

**CPU-Intensive Workloads:**
- Increase CPU limits for quantum optimization
- Consider CPU affinity for performance-critical pods
- Enable CPU frequency scaling

### Multi-Region Deployment

```bash
# Deploy to multiple regions for global availability
kubectl config use-context us-east-1
kubectl apply -f production_deployment_manifest.yaml

kubectl config use-context eu-west-1
kubectl apply -f production_deployment_manifest.yaml

kubectl config use-context ap-southeast-1
kubectl apply -f production_deployment_manifest.yaml
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      run: |
        docker build -t securempc/transformer-inference:${{ github.ref_name }} .
        docker push securempc/transformer-inference:${{ github.ref_name }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/mpc-transformer-api \
          mpc-transformer=securempc/transformer-inference:${{ github.ref_name }}
        kubectl rollout status deployment/mpc-transformer-api
```

### Rolling Updates

```bash
# Zero-downtime deployment
kubectl set image deployment/mpc-transformer-api \
  mpc-transformer=securempc/transformer-inference:v0.2.1
  
# Monitor rollout
kubectl rollout status deployment/mpc-transformer-api

# Rollback if necessary
kubectl rollout undo deployment/mpc-transformer-api
```

## 🛠️ Maintenance

### Backup Strategy

**Database Backups:**
```bash
# Automated daily backups
docker exec mpc-postgres pg_dump -U mpc_user mpc_transformer > backup_$(date +%Y%m%d).sql

# Point-in-time recovery setup
# Enable WAL archiving in PostgreSQL configuration
```

**Configuration Backups:**
```bash
# Kubernetes resource backup
kubectl get all -n secure-mpc-transformer -o yaml > k8s-backup-$(date +%Y%m%d).yaml

# Docker Compose backup
cp docker-compose.production.yml docker-compose.backup-$(date +%Y%m%d).yml
```

### Log Rotation

```yaml
# docker-compose.production.yml logging configuration
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "3"
```

### Security Updates

```bash
# Weekly security update script
#!/bin/bash
# scripts/security-update.sh

# Update base images
docker pull securempc/transformer-inference:latest
docker pull postgres:15-alpine
docker pull redis:7-alpine

# Scan for vulnerabilities
trivy image securempc/transformer-inference:latest

# Update deployment if no critical vulnerabilities
if [ $? -eq 0 ]; then
    kubectl set image deployment/mpc-transformer-api \
      mpc-transformer=securempc/transformer-inference:latest
fi
```

## 🚨 Troubleshooting

### Common Issues

**Issue: High Memory Usage**
```bash
# Check memory allocation
kubectl describe pod mpc-transformer-api-xxx
kubectl top pod mpc-transformer-api-xxx

# Solution: Increase memory limits or optimize code
```

**Issue: Database Connection Failures**
```bash
# Check database connectivity
kubectl exec -it mpc-transformer-api-xxx -- nc -zv postgres 5432

# Check database logs
kubectl logs postgres-xxx
```

**Issue: SSL Certificate Expiration**
```bash
# Check certificate expiration
openssl x509 -in nginx/ssl/fullchain.pem -text -noout | grep "Not After"

# Renew Let's Encrypt certificate
certbot renew --dry-run
```

### Performance Debugging

```bash
# Enable debug logging
kubectl set env deployment/mpc-transformer-api LOG_LEVEL=DEBUG

# Profile memory usage
kubectl exec -it mpc-transformer-api-xxx -- python -m memory_profiler

# Analyze quantum optimization performance
kubectl logs mpc-transformer-api-xxx | grep "quantum_optimization"
```

### Emergency Procedures

**Scale Down in Emergency:**
```bash
# Immediate scale down
kubectl scale deployment mpc-transformer-api --replicas=1

# Disable auto-scaling
kubectl delete hpa mpc-transformer-hpa
```

**Database Recovery:**
```bash
# Restore from backup
psql -U mpc_user -d mpc_transformer < backup_20231201.sql

# Point-in-time recovery
# Follow PostgreSQL PITR documentation
```

## 📞 Support and Monitoring

### Monitoring Contacts

- **Production Issues:** production-alerts@your-domain.com
- **Security Issues:** security@your-domain.com
- **Performance Issues:** performance@your-domain.com

### SLA Targets

- **Availability:** 99.9% uptime
- **Latency:** P95 < 200ms, P99 < 500ms
- **Error Rate:** < 0.1%
- **Security Response:** < 15 minutes for critical threats

### Status Page

Set up a status page to communicate with users:
- Use Atlassian Statuspage, StatusPage.io, or similar
- Integrate with monitoring alerts
- Provide maintenance schedules and incident updates

---

## 🎉 Congratulations!

You have successfully deployed the Secure MPC Transformer with Quantum-Inspired Task Planning to production. The system is now ready to handle secure multi-party computation with advanced optimization and comprehensive security features.

For additional support, please refer to:
- [API Documentation](./API_REFERENCE.md)
- [Security Guide](./SECURITY.md)
- [Performance Tuning Guide](./PERFORMANCE.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)