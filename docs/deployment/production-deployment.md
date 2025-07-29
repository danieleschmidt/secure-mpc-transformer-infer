# Production Deployment Guide

This guide covers production deployment of the Secure MPC Transformer system across different environments and configurations.

## Overview

The Secure MPC Transformer can be deployed in several configurations:
- **Single-node CPU deployment** - For testing and light workloads
- **Single-node GPU deployment** - For high-performance inference
- **Multi-party distributed deployment** - For full MPC protocols
- **Kubernetes cluster deployment** - For scalable production use

## Prerequisites

### Hardware Requirements

#### Minimum Configuration
- **CPU**: 4 cores, 3.0GHz+
- **RAM**: 16GB
- **Storage**: 50GB SSD
- **Network**: 1Gbps for multi-party setups

#### Recommended Configuration
- **CPU**: 8+ cores, 3.5GHz+
- **RAM**: 64GB+
- **GPU**: NVIDIA RTX 4090 or A100 (24GB+ VRAM)
- **Storage**: 100GB+ NVMe SSD
- **Network**: 10Gbps for production MPC

### Software Requirements

- **OS**: Ubuntu 22.04 LTS (recommended) or RHEL 8+
- **Docker**: 24.0+ with NVIDIA Container Runtime
- **Python**: 3.10+ (if running without containers)
- **CUDA**: 12.0+ (for GPU deployments)

## Deployment Options

### 1. Docker Deployment (Recommended)

#### Single-Node CPU Deployment

```bash
# Pull the latest CPU image
docker pull ghcr.io/yourusername/secure-mpc-transformer:latest-cpu

# Run with production configuration
docker run -d \
  --name mpc-transformer-prod \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /path/to/config:/app/config \
  -v /path/to/logs:/app/logs \
  -e MPC_PROTOCOL=semi_honest_3pc \
  -e MPC_PARTY_ID=0 \
  -e LOG_LEVEL=INFO \
  ghcr.io/yourusername/secure-mpc-transformer:latest-cpu
```

#### Single-Node GPU Deployment

```bash
# Ensure NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Run GPU-enabled container
docker run -d \
  --name mpc-transformer-gpu \
  --restart unless-stopped \
  --gpus all \
  -p 8080:8080 \
  --shm-size=4g \
  -v /path/to/config:/app/config \
  -v /path/to/logs:/app/logs \
  -e MPC_PROTOCOL=aby3 \
  -e MPC_PARTY_ID=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e GPU_MEMORY_FRACTION=0.9 \
  ghcr.io/yourusername/secure-mpc-transformer:latest-gpu
```

#### Multi-Party Docker Compose Deployment

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  mpc-party-0:
    image: ghcr.io/yourusername/secure-mpc-transformer:latest-gpu
    container_name: mpc-party-0
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      - MPC_PARTY_ID=0
      - MPC_NUM_PARTIES=3
      - MPC_PROTOCOL=aby3
      - MPC_PEERS=mpc-party-1:50051,mpc-party-2:50051
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./config/party-0:/app/config
      - ./logs/party-0:/app/logs
      - ./certs:/app/certs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - mpc-network

  mpc-party-1:
    image: ghcr.io/yourusername/secure-mpc-transformer:latest-gpu
    container_name: mpc-party-1
    restart: unless-stopped
    ports:
      - "8081:8080"
      - "50052:50051"
    environment:
      - MPC_PARTY_ID=1
      - MPC_NUM_PARTIES=3
      - MPC_PROTOCOL=aby3
      - MPC_PEERS=mpc-party-0:50051,mpc-party-2:50051
      - CUDA_VISIBLE_DEVICES=1
    volumes:
      - ./config/party-1:/app/config
      - ./logs/party-1:/app/logs
      - ./certs:/app/certs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - mpc-network

  mpc-party-2:
    image: ghcr.io/yourusername/secure-mpc-transformer:latest-gpu
    container_name: mpc-party-2
    restart: unless-stopped
    ports:
      - "8082:8080"
      - "50053:50051"
    environment:
      - MPC_PARTY_ID=2
      - MPC_NUM_PARTIES=3
      - MPC_PROTOCOL=aby3
      - MPC_PEERS=mpc-party-0:50051,mpc-party-1:50051
      - CUDA_VISIBLE_DEVICES=2
    volumes:
      - ./config/party-2:/app/config
      - ./logs/party-2:/app/logs
      - ./certs:/app/certs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - mpc-network

  nginx:
    image: nginx:alpine
    container_name: mpc-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - mpc-party-0
      - mpc-party-1
      - mpc-party-2
    networks:
      - mpc-network

networks:
  mpc-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

Deploy with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 2. Kubernetes Deployment

#### Namespace and RBAC

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mpc-transformer
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: mpc-transformer
  name: mpc-transformer-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mpc-transformer-rolebinding
  namespace: mpc-transformer
subjects:
- kind: ServiceAccount
  name: mpc-transformer-sa
  namespace: mpc-transformer
roleRef:
  kind: Role
  name: mpc-transformer-role
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mpc-transformer-sa
  namespace: mpc-transformer
```

#### ConfigMaps and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mpc-config
  namespace: mpc-transformer
data:
  protocol.yaml: |
    protocol: aby3
    security_level: 128
    num_parties: 3
    timeout: 30
  logging.yaml: |
    level: INFO
    format: json
    output: stdout
---
apiVersion: v1
kind: Secret
metadata:
  name: mpc-tls-certs
  namespace: mpc-transformer
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-certificate>
  tls.key: <base64-encoded-private-key>
  ca.crt: <base64-encoded-ca-certificate>
```

#### Deployment Manifests

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mpc-transformer
  namespace: mpc-transformer
spec:
  serviceName: mpc-service
  replicas: 3
  selector:
    matchLabels:
      app: mpc-transformer
  template:
    metadata:
      labels:
        app: mpc-transformer
    spec:
      serviceAccountName: mpc-transformer-sa
      containers:
      - name: mpc-transformer
        image: ghcr.io/yourusername/secure-mpc-transformer:latest-gpu
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 50051
          name: grpc
        env:
        - name: MPC_PARTY_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['mpc.party.id']
        - name: MPC_NUM_PARTIES
          value: "3"
        - name: MPC_PROTOCOL
          value: "aby3"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: tls-certs
          mountPath: /app/certs
        - name: data
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: mpc-config
      - name: tls-certs
        secret:
          secretName: mpc-tls-certs
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
---
apiVersion: v1
kind: Service
metadata:
  name: mpc-service
  namespace: mpc-transformer
spec:
  selector:
    app: mpc-transformer
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: grpc
    port: 50051
    targetPort: 50051
  clusterIP: None
---
apiVersion: v1
kind: Service
metadata:
  name: mpc-lb
  namespace: mpc-transformer
spec:
  type: LoadBalancer
  selector:
    app: mpc-transformer
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: https
    port: 443
    targetPort: 8080
```

#### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mpc-ingress
  namespace: mpc-transformer
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - mpc-transformer.yourdomain.com
    secretName: mpc-tls-secret
  rules:
  - host: mpc-transformer.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mpc-service
            port:
              number: 80
```

## Configuration Management

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MPC_PROTOCOL` | MPC protocol to use | `semi_honest_3pc` | No |
| `MPC_PARTY_ID` | Party identifier (0, 1, 2) | `0` | Yes |
| `MPC_NUM_PARTIES` | Number of parties | `3` | No |
| `MPC_PEERS` | Comma-separated peer addresses | None | Yes for multi-party |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | `all` | No |
| `GPU_MEMORY_FRACTION` | GPU memory fraction | `0.9` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `TLS_CERT_FILE` | TLS certificate path | `/app/certs/cert.pem` | No |
| `TLS_KEY_FILE` | TLS key path | `/app/certs/key.pem` | No |

### Configuration Files

#### `config/protocol.yaml`
```yaml
protocol: aby3
security_level: 128
num_parties: 3
party_id: 0
peers:
  - host: party-1.internal
    port: 50051
  - host: party-2.internal
    port: 50051

timeouts:
  connection: 30
  computation: 300
  heartbeat: 10

crypto:
  field_size: 128
  key_length: 2048
  use_preprocessing: true
```

#### `config/logging.yaml`
```yaml
version: 1
formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: detailed
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /app/logs/mpc-transformer.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    level: DEBUG
    formatter: detailed
loggers:
  secure_mpc_transformer:
    level: DEBUG
    handlers: [console, file]
root:
  level: INFO
  handlers: [console]
```

## Security Considerations

### TLS/SSL Configuration

Generate certificates for production:

```bash
# Generate CA certificate
openssl genrsa -out ca-key.pem 4096
openssl req -new -x509 -days 365 -key ca-key.pem -sha256 -out ca.pem

# Generate server certificate
openssl genrsa -out server-key.pem 4096
openssl req -subj "/CN=mpc-transformer" -sha256 -new -key server-key.pem -out server.csr
openssl x509 -req -days 365 -sha256 -in server.csr -CA ca.pem -CAkey ca-key.pem -out server-cert.pem

# Generate client certificates for each party
for i in {0..2}; do
  openssl genrsa -out client-${i}-key.pem 4096
  openssl req -subj "/CN=party-${i}" -new -key client-${i}-key.pem -out client-${i}.csr
  openssl x509 -req -days 365 -sha256 -in client-${i}.csr -CA ca.pem -CAkey ca-key.pem -out client-${i}-cert.pem
done
```

### Network Security

#### Firewall Rules

```bash
# Allow inbound HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow MPC communication (restrict to known IPs)
sudo ufw allow from 10.0.1.0/24 to any port 50051

# Allow SSH (restrict to management network)
sudo ufw allow from 10.0.0.0/24 to any port 22

# Enable firewall
sudo ufw enable
```

#### Network Segmentation

- **Management Network**: `10.0.0.0/24` - SSH, monitoring
- **Application Network**: `10.0.1.0/24` - HTTP/HTTPS traffic
- **MPC Network**: `10.0.2.0/24` - Inter-party communication
- **Storage Network**: `10.0.3.0/24` - Database and file storage

### Access Control

#### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: mpc-transformer
  name: mpc-operator
rules:
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch", "update", "patch"]
```

## Monitoring and Observability

### Health Checks

The application exposes several health check endpoints:

- `GET /health` - Basic health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics
- `GET /debug/pprof` - Performance profiling (debug builds only)

### Metrics Collection

Configure Prometheus scraping:

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'mpc-transformer'
  static_configs:
  - targets: ['mpc-party-0:8080', 'mpc-party-1:8080', 'mpc-party-2:8080']
  metrics_path: /metrics
  scrape_interval: 10s
```

### Log Aggregation

Configure log forwarding to centralized logging:

```yaml
# fluentd-config.yaml
<source>
  @type tail
  path /app/logs/*.log
  pos_file /var/log/fluentd-mpc.log.pos
  tag mpc.transformer
  format json
</source>

<match mpc.transformer>
  @type elasticsearch
  host elasticsearch.logging.svc.cluster.local
  port 9200
  index_name mpc-transformer
  type_name _doc
</match>
```

## Backup and Recovery

### Data Backup Strategy

1. **Configuration Backup**: Daily backup of configuration files
2. **Certificate Backup**: Secure backup of TLS certificates
3. **Log Archival**: Weekly archival of application logs
4. **State Backup**: Backup of protocol state (if stateful)

#### Backup Script

```bash
#!/bin/bash
# backup.sh - Production backup script

BACKUP_DIR="/backup/mpc-transformer/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configurations
cp -r /app/config "$BACKUP_DIR/"

# Backup certificates (encrypted)
tar -czf "$BACKUP_DIR/certs.tar.gz" -C /app/certs .
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
    --s2k-digest-algo SHA512 --s2k-count 65536 --symmetric \
    --output "$BACKUP_DIR/certs.tar.gz.gpg" "$BACKUP_DIR/certs.tar.gz"
rm "$BACKUP_DIR/certs.tar.gz"

# Archive logs
find /app/logs -name "*.log" -mtime +1 | \
  tar -czf "$BACKUP_DIR/logs.tar.gz" -T -

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR" s3://mpc-transformer-backups/ --recursive

# Cleanup old backups (keep 30 days)
find /backup/mpc-transformer -type d -mtime +30 -exec rm -rf {} \;
```

### Disaster Recovery

#### Recovery Procedure

1. **Infrastructure Recovery**:
   ```bash
   # Restore from infrastructure as code
   terraform apply -var="restore_from_backup=true"
   ```

2. **Application Recovery**:
   ```bash
   # Restore configuration
   aws s3 cp s3://mpc-transformer-backups/latest/config/ /app/config/ --recursive
   
   # Restore certificates
   aws s3 cp s3://mpc-transformer-backups/latest/certs.tar.gz.gpg .
   gpg --decrypt certs.tar.gz.gpg | tar -xzf - -C /app/certs/
   
   # Restart services
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Verification**:
   ```bash
   # Test health endpoints
   curl -f http://localhost:8080/health
   
   # Verify inter-party communication
   ./scripts/test-mpc-connectivity.sh
   ```

## Performance Optimization

### Resource Allocation

#### CPU Optimization
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# Set process affinity
taskset -c 0-7 ./mpc-transformer
```

#### GPU Optimization
```bash
# Set GPU performance mode
nvidia-smi -pm 1

# Set maximum performance
nvidia-smi -ac memory_clock,graphics_clock

# Monitor GPU utilization
nvidia-smi -l 1
```

#### Memory Optimization
```bash
# Increase shared memory
echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
echo 'kernel.shmmax=68719476736' >> /etc/sysctl.conf
sysctl -p
```

### Load Balancing

#### NGINX Configuration

```nginx
# nginx.conf
upstream mpc_backend {
    least_conn;
    server mpc-party-0:8080 weight=1 max_fails=3 fail_timeout=30s;
    server mpc-party-1:8080 weight=1 max_fails=3 fail_timeout=30s;
    server mpc-party-2:8080 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name mpc-transformer.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name mpc-transformer.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://mpc_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        proxy_pass http://mpc_backend;
        access_log off;
    }
}
```

## Troubleshooting

### Common Issues

#### GPU Not Available
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

#### Network Connectivity Issues
```bash
# Test inter-party connectivity
nc -zv party-1.internal 50051
nc -zv party-2.internal 50051

# Check firewall rules
sudo ufw status verbose

# Verify DNS resolution
nslookup party-1.internal
```

#### Performance Issues
```bash
# Monitor system resources
htop
iotop
nvidia-smi -l 1

# Check container resources
docker stats

# Profile application
python -m cProfile -o profile.stats ./mpc-transformer
```

### Log Analysis

#### Useful Log Queries

```bash
# Error analysis
grep -i error /app/logs/*.log | tail -100

# Performance monitoring
grep "latency" /app/logs/*.log | awk '{print $NF}' | sort -n

# Security events
grep -i "auth\|security\|cert" /app/logs/*.log
```

#### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages
- **WARNING**: Potential issues that don't prevent operation
- **ERROR**: Error conditions that may affect functionality
- **CRITICAL**: Serious errors that may cause system failure

## Maintenance

### Regular Maintenance Tasks

#### Daily
- [ ] Check system health and alerts
- [ ] Review error logs
- [ ] Monitor resource utilization
- [ ] Verify backup completion

#### Weekly
- [ ] Update security patches
- [ ] Rotate log files
- [ ] Review performance metrics
- [ ] Test disaster recovery procedures

#### Monthly
- [ ] Update container images
- [ ] Certificate renewal (if needed)
- [ ] Capacity planning review
- [ ] Security audit

### Update Procedures

#### Rolling Updates

```bash
# Update single party (zero-downtime)
docker-compose -f docker-compose.prod.yml up -d --no-deps mpc-party-0

# Wait for health check
while ! curl -f http://localhost:8080/health; do sleep 5; done

# Update remaining parties
docker-compose -f docker-compose.prod.yml up -d --no-deps mpc-party-1 mpc-party-2
```

#### Emergency Updates

```bash
# Emergency stop all services
docker-compose -f docker-compose.prod.yml down

# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Start with health checks
docker-compose -f docker-compose.prod.yml up -d

# Verify functionality
./scripts/verify-deployment.sh
```

This production deployment guide ensures secure, scalable, and maintainable deployment of the Secure MPC Transformer system across various environments and configurations.