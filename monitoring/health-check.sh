#!/bin/bash
# Health check script for MPC Transformer monitoring stack
# Verifies all monitoring components are operational

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
ALERTMANAGER_URL="http://localhost:9093"
JAEGER_URL="http://localhost:16686"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        log_success "$service_name is healthy"
        return 0
    else
        log_error "$service_name is not responding"
        return 1
    fi
}

check_prometheus() {
    log_info "Checking Prometheus..."
    
    if check_service "Prometheus" "$PROMETHEUS_URL/-/healthy"; then
        # Check if Prometheus is scraping targets
        local targets_up=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=up" | jq -r '.data.result | length')
        if [[ "$targets_up" -gt 0 ]]; then
            log_success "Prometheus has $targets_up active targets"
        else
            log_warning "Prometheus has no active targets"
        fi
        
        # Check if rules are loaded
        local rules_count=$(curl -s "$PROMETHEUS_URL/api/v1/rules" | jq -r '.data.groups | length')
        log_info "Prometheus has $rules_count rule groups loaded"
        
        return 0
    else
        return 1
    fi
}

check_grafana() {
    log_info "Checking Grafana..."
    
    if check_service "Grafana" "$GRAFANA_URL/api/health"; then
        # Check datasources
        local datasources=$(curl -s -u admin:admin123 "$GRAFANA_URL/api/datasources" | jq -r '. | length' 2>/dev/null || echo "0")
        log_info "Grafana has $datasources datasources configured"
        
        # Check dashboards
        local dashboards=$(curl -s -u admin:admin123 "$GRAFANA_URL/api/search" | jq -r '. | length' 2>/dev/null || echo "0")
        log_info "Grafana has $dashboards dashboards available"
        
        return 0
    else
        return 1
    fi
}

check_alertmanager() {
    log_info "Checking Alertmanager..."
    
    if check_service "Alertmanager" "$ALERTMANAGER_URL/-/healthy"; then
        # Check active alerts
        local active_alerts=$(curl -s "$ALERTMANAGER_URL/api/v1/alerts" | jq -r '. | length' 2>/dev/null || echo "0")
        log_info "Alertmanager has $active_alerts active alerts"
        
        # Check silences
        local silences=$(curl -s "$ALERTMANAGER_URL/api/v1/silences" | jq -r '. | length' 2>/dev/null || echo "0")
        log_info "Alertmanager has $silences active silences"
        
        return 0
    else
        return 1
    fi
}

check_jaeger() {
    log_info "Checking Jaeger..."
    
    # Jaeger doesn't have a standard health endpoint, so we check the main page
    if check_service "Jaeger" "$JAEGER_URL"; then
        return 0
    else
        return 1
    fi
}

check_gpu_exporter() {
    log_info "Checking GPU Exporter..."
    
    if command -v nvidia-smi &> /dev/null; then
        if check_service "GPU Exporter" "http://localhost:9400/metrics"; then
            return 0
        else
            log_warning "GPU Exporter not responding (GPU monitoring disabled)"
            return 1
        fi
    else
        log_info "No GPU detected, skipping GPU exporter check"
        return 0
    fi
}

check_node_exporter() {
    log_info "Checking Node Exporter..."
    
    if check_service "Node Exporter" "http://localhost:9100/metrics"; then
        return 0
    else
        return 1
    fi
}

check_redis() {
    log_info "Checking Redis..."
    
    if redis-cli ping &> /dev/null; then
        log_success "Redis is responding to ping"
        
        # Check Redis metrics exporter
        if check_service "Redis Exporter" "http://localhost:9121/metrics"; then
            return 0
        else
            log_warning "Redis Exporter not responding"
            return 1
        fi
    else
        log_error "Redis is not responding"
        return 1
    fi
}

check_docker_containers() {
    log_info "Checking Docker containers..."
    
    local containers=("mpc-prometheus" "mpc-grafana" "mpc-alertmanager" "mpc-jaeger")
    local failed_containers=()
    
    for container in "${containers[@]}"; do
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            log_success "Container $container is running"
        else
            log_error "Container $container is not running"
            failed_containers+=("$container")
        fi
    done
    
    if [[ ${#failed_containers[@]} -eq 0 ]]; then
        return 0
    else
        log_error "Failed containers: ${failed_containers[*]}"
        return 1
    fi
}

check_disk_space() {
    log_info "Checking disk space for monitoring data..."
    
    local monitoring_dirs=("/var/lib/docker/volumes" "/tmp")
    
    for dir in "${monitoring_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local usage=$(df -h "$dir" | awk 'NR==2 {print $5}' | sed 's/%//')
            if [[ "$usage" -gt 80 ]]; then
                log_warning "Disk usage for $dir is ${usage}% (consider cleanup)"
            else
                log_success "Disk usage for $dir is ${usage}%"
            fi
        fi
    done
}

check_network_connectivity() {
    log_info "Checking network connectivity..."
    
    local endpoints=("$PROMETHEUS_URL" "$GRAFANA_URL" "$ALERTMANAGER_URL")
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s --connect-timeout 5 "$endpoint" > /dev/null; then
            log_success "Network connectivity to $endpoint is OK"
        else
            log_error "Cannot connect to $endpoint"
            return 1
        fi
    done
    
    return 0
}

generate_health_report() {
    local failed_checks=("$@")
    
    echo ""
    log_info "==================== HEALTH CHECK SUMMARY ===================="
    
    if [[ ${#failed_checks[@]} -eq 0 ]]; then
        log_success "All monitoring components are healthy! ✅"
        
        echo ""
        echo "Monitoring Stack Status:"
        echo "- Prometheus: ✅ Active"
        echo "- Grafana: ✅ Active" 
        echo "- Alertmanager: ✅ Active"
        echo "- Jaeger: ✅ Active"
        echo "- Node Exporter: ✅ Active"
        echo "- Redis: ✅ Active"
        
        if command -v nvidia-smi &> /dev/null; then
            echo "- GPU Exporter: ✅ Active"
        else
            echo "- GPU Exporter: ℹ️ N/A (No GPU)"
        fi
        
        echo ""
        echo "Access URLs:"
        echo "- Grafana Dashboard: $GRAFANA_URL (admin/admin123)"
        echo "- Prometheus: $PROMETHEUS_URL"
        echo "- Alertmanager: $ALERTMANAGER_URL"
        echo "- Jaeger: $JAEGER_URL"
        
    else
        log_error "Health check failed for ${#failed_checks[@]} component(s): ${failed_checks[*]}"
        
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check Docker containers: docker ps -a"
        echo "2. Check container logs: docker-compose -f monitoring/docker-compose.monitoring.yml logs [service]"
        echo "3. Restart monitoring stack: make docker-compose-down && make docker-compose-up"
        echo "4. Check network connectivity and firewall settings"
        
        return 1
    fi
}

main() {
    echo ""
    log_info "🏥 Starting MPC Transformer Monitoring Health Check..."
    echo ""
    
    local failed_checks=()
    
    # Check Docker containers first
    check_docker_containers || failed_checks+=("docker-containers")
    
    # Check core monitoring services
    check_prometheus || failed_checks+=("prometheus")
    check_grafana || failed_checks+=("grafana")
    check_alertmanager || failed_checks+=("alertmanager")
    check_jaeger || failed_checks+=("jaeger")
    
    # Check exporters
    check_node_exporter || failed_checks+=("node-exporter")
    check_gpu_exporter || failed_checks+=("gpu-exporter")
    check_redis || failed_checks+=("redis")
    
    # Check system resources
    check_disk_space
    check_network_connectivity || failed_checks+=("network")
    
    # Generate report
    generate_health_report "${failed_checks[@]}"
    
    # Exit with appropriate code
    if [[ ${#failed_checks[@]} -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Check dependencies
if ! command -v curl &> /dev/null; then
    log_error "curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    log_warning "jq is not installed - some checks will have limited functionality"
fi

# Run main function
main "$@"