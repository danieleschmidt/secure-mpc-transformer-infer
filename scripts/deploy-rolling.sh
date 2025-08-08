#!/bin/bash

# Rolling deployment script for MPC Transformer production
# Supports zero-downtime deployments with health checks and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
REGION=""
IMAGE=""
NAMESPACE=""
TIMEOUT="600s"
REPLICA_COUNT=""
DRY_RUN=false
VERBOSE=false
ROLLBACK_ON_FAILURE=true
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

# Help function
show_help() {
    cat << EOF
Rolling Deployment Script for MPC Transformer

Usage: $0 [OPTIONS]

Options:
    -r, --region REGION         Target region (americas, europe, apac)
    -i, --image IMAGE           Container image to deploy
    -n, --namespace NAMESPACE   Kubernetes namespace (optional)
    -t, --timeout TIMEOUT      Deployment timeout (default: 600s)
    -c, --replicas COUNT        Target replica count (optional)
    --dry-run                   Show what would be done without executing
    --no-rollback               Disable automatic rollback on failure
    --health-retries COUNT      Health check retry count (default: 10)
    --health-interval SECONDS   Health check interval (default: 30)
    -v, --verbose               Enable verbose output
    -h, --help                  Show this help message

Examples:
    # Rolling deployment to Americas region
    $0 --region=americas --image=ghcr.io/mpc-transformer/app:v1.2.3

    # Deployment with custom replica count and timeout
    $0 --region=europe --image=ghcr.io/mpc-transformer/app:v1.2.3 \\
       --replicas=5 --timeout=900s

    # Dry run to preview changes
    $0 --region=apac --image=ghcr.io/mpc-transformer/app:v1.2.3 --dry-run

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -c|--replicas)
            REPLICA_COUNT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        --health-retries)
            HEALTH_CHECK_RETRIES="$2"
            shift 2
            ;;
        --health-interval)
            HEALTH_CHECK_INTERVAL="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation
if [[ -z "$REGION" ]]; then
    log_error "Region is required"
    show_help
    exit 1
fi

if [[ -z "$IMAGE" ]]; then
    log_error "Image is required"
    show_help
    exit 1
fi

# Set namespace based on region if not provided
if [[ -z "$NAMESPACE" ]]; then
    NAMESPACE="mpc-transformer-${REGION}"
fi

# Validate region
case "$REGION" in
    americas|europe|apac)
        ;;
    *)
        log_error "Invalid region: $REGION. Must be one of: americas, europe, apac"
        exit 1
        ;;
esac

# Function to execute commands with dry-run support
execute() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: $*"
    else
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Executing: $*"
        fi
        "$@"
    fi
}

# Function to get current deployment image
get_current_image() {
    kubectl get deployment mpc-transformer -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo ""
}

# Function to get current replica count
get_current_replicas() {
    kubectl get deployment mpc-transformer -n "$NAMESPACE" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "3"
}

# Function to wait for deployment rollout
wait_for_rollout() {
    local deployment="$1"
    local namespace="$2"
    local timeout="$3"
    
    log_info "Waiting for deployment rollout to complete (timeout: $timeout)..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would wait for rollout of $deployment in $namespace"
        return 0
    fi
    
    if kubectl rollout status deployment/"$deployment" -n "$namespace" --timeout="$timeout"; then
        log_success "Deployment rollout completed successfully"
        return 0
    else
        log_error "Deployment rollout failed or timed out"
        return 1
    fi
}

# Function to perform health checks
perform_health_checks() {
    local region_endpoint=""
    
    case "$REGION" in
        americas)
            region_endpoint="https://us-east.mpc-transformer.com"
            ;;
        europe)
            region_endpoint="https://eu-west.mpc-transformer.com"
            ;;
        apac)
            region_endpoint="https://ap-northeast.mpc-transformer.com"
            ;;
    esac
    
    log_info "Performing health checks for $REGION region..."
    
    for attempt in $(seq 1 $HEALTH_CHECK_RETRIES); do
        log_info "Health check attempt $attempt/$HEALTH_CHECK_RETRIES"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would check health endpoint: $region_endpoint/health"
            return 0
        fi
        
        # Check internal Kubernetes service health
        if kubectl exec -n "$NAMESPACE" \
            deployment/mpc-transformer -- \
            curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "Internal health check passed"
        else
            log_warning "Internal health check failed on attempt $attempt"
            if [[ $attempt -lt $HEALTH_CHECK_RETRIES ]]; then
                sleep $HEALTH_CHECK_INTERVAL
                continue
            else
                log_error "Internal health checks failed after $HEALTH_CHECK_RETRIES attempts"
                return 1
            fi
        fi
        
        # Check external endpoint health
        if curl -f -s --max-time 30 "$region_endpoint/health" > /dev/null 2>&1; then
            log_success "External health check passed for $region_endpoint"
            return 0
        else
            log_warning "External health check failed on attempt $attempt"
            if [[ $attempt -lt $HEALTH_CHECK_RETRIES ]]; then
                sleep $HEALTH_CHECK_INTERVAL
                continue
            else
                log_error "External health checks failed after $HEALTH_CHECK_RETRIES attempts"
                return 1
            fi
        fi
    done
    
    return 1
}

# Function to validate MPC functionality
validate_mpc_functionality() {
    log_info "Validating MPC functionality..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would validate MPC functionality"
        return 0
    fi
    
    # Get a pod to run validation from
    local pod_name
    pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=mpc-transformer \
        --field-selector=status.phase=Running -o name | head -1 | cut -d'/' -f2)
    
    if [[ -z "$pod_name" ]]; then
        log_error "No running pods found for MPC validation"
        return 1
    fi
    
    # Run MPC connectivity test
    if kubectl exec -n "$NAMESPACE" "$pod_name" -- \
        python -c "
import requests
import json

# Test MPC protocol endpoint
try:
    response = requests.post(
        'http://localhost:8080/mpc/test-connectivity',
        json={'protocol': 'aby3', 'parties': 3},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
    
    if result.get('status') == 'success':
        print('✅ MPC connectivity test passed')
        exit(0)
    else:
        print('❌ MPC connectivity test failed')
        exit(1)
except Exception as e:
    print(f'❌ MPC validation error: {e}')
    exit(1)
"; then
        log_success "MPC functionality validation passed"
        return 0
    else
        log_error "MPC functionality validation failed"
        return 1
    fi
}

# Function to rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback deployment"
        return 0
    fi
    
    if kubectl rollout undo deployment/mpc-transformer -n "$NAMESPACE"; then
        log_info "Rollback initiated, waiting for completion..."
        if wait_for_rollout "mpc-transformer" "$NAMESPACE" "$TIMEOUT"; then
            log_success "Rollback completed successfully"
            return 0
        else
            log_error "Rollback failed"
            return 1
        fi
    else
        log_error "Failed to initiate rollback"
        return 1
    fi
}

# Function to check deployment prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check kubectl connection
    if ! kubectl cluster-info > /dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" > /dev/null 2>&1; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check deployment exists
    if ! kubectl get deployment mpc-transformer -n "$NAMESPACE" > /dev/null 2>&1; then
        log_error "Deployment mpc-transformer does not exist in namespace $NAMESPACE"
        exit 1
    fi
    
    # Check image accessibility
    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Validating image accessibility..."
        if ! docker manifest inspect "$IMAGE" > /dev/null 2>&1; then
            log_warning "Cannot inspect image manifest (may not affect deployment)"
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Function to create deployment backup
create_deployment_backup() {
    local backup_dir="$PROJECT_ROOT/backups/deployments"
    local backup_file="$backup_dir/mpc-transformer-${REGION}-$(date +%Y%m%d-%H%M%S).yaml"
    
    log_info "Creating deployment backup..."
    
    mkdir -p "$backup_dir"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl get deployment mpc-transformer -n "$NAMESPACE" -o yaml > "$backup_file"
        log_success "Deployment backup saved to $backup_file"
    else
        log_info "[DRY RUN] Would create backup at $backup_file"
    fi
}

# Function to update deployment annotations
update_deployment_annotations() {
    local deployment_timestamp
    deployment_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    log_info "Updating deployment annotations..."
    
    execute kubectl annotate deployment mpc-transformer -n "$NAMESPACE" \
        deployment.mpc-transformer.com/last-updated="$deployment_timestamp" \
        deployment.mpc-transformer.com/image="$IMAGE" \
        deployment.mpc-transformer.com/deployed-by="$(whoami)" \
        --overwrite
}

# Function to perform gradual rollout
perform_gradual_rollout() {
    local current_replicas
    current_replicas=$(get_current_replicas)
    
    local target_replicas="${REPLICA_COUNT:-$current_replicas}"
    
    log_info "Starting gradual rollout (current: $current_replicas, target: $target_replicas)"
    
    # Set deployment strategy to RollingUpdate with controlled parameters
    execute kubectl patch deployment mpc-transformer -n "$NAMESPACE" -p '{
        "spec": {
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": "25%",
                    "maxSurge": "25%"
                }
            }
        }
    }'
    
    # Update image
    execute kubectl set image deployment/mpc-transformer \
        mpc-transformer="$IMAGE" \
        -n "$NAMESPACE"
    
    # Update replica count if specified
    if [[ -n "$REPLICA_COUNT" ]]; then
        execute kubectl scale deployment mpc-transformer \
            --replicas="$REPLICA_COUNT" \
            -n "$NAMESPACE"
    fi
}

# Main deployment function
main() {
    local start_time
    start_time=$(date +%s)
    
    log_info "Starting rolling deployment for MPC Transformer"
    log_info "Region: $REGION"
    log_info "Namespace: $NAMESPACE" 
    log_info "Image: $IMAGE"
    log_info "Timeout: $TIMEOUT"
    log_info "Dry Run: $DRY_RUN"
    log_info "Rollback on Failure: $ROLLBACK_ON_FAILURE"
    
    # Store current state for potential rollback
    local current_image
    current_image=$(get_current_image)
    
    local current_replicas
    current_replicas=$(get_current_replicas)
    
    log_info "Current image: ${current_image:-'<none>'}"
    log_info "Current replicas: $current_replicas"
    
    # Check if we're trying to deploy the same image
    if [[ "$current_image" == "$IMAGE" ]]; then
        log_warning "Target image is the same as current image - no deployment needed"
        return 0
    fi
    
    # Execute deployment steps
    check_prerequisites
    create_deployment_backup
    update_deployment_annotations
    perform_gradual_rollout
    
    # Wait for rollout and perform validation
    if wait_for_rollout "mpc-transformer" "$NAMESPACE" "$TIMEOUT"; then
        log_success "Deployment rollout completed"
        
        if perform_health_checks; then
            log_success "Health checks passed"
            
            if validate_mpc_functionality; then
                log_success "MPC functionality validation passed"
                
                local end_time
                end_time=$(date +%s)
                local duration=$((end_time - start_time))
                
                log_success "Rolling deployment completed successfully in ${duration}s"
                log_success "New image: $IMAGE"
                
                # Send success notification
                if command -v curl > /dev/null 2>&1; then
                    curl -X POST "${SLACK_WEBHOOK_URL:-}" \
                        -H "Content-Type: application/json" \
                        -d "{\"text\":\"✅ Rolling deployment successful in $REGION (${duration}s)\"}" \
                        2>/dev/null || true
                fi
                
                return 0
            else
                log_error "MPC functionality validation failed"
                if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                    rollback_deployment
                fi
                return 1
            fi
        else
            log_error "Health checks failed"
            if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                rollback_deployment
            fi
            return 1
        fi
    else
        log_error "Deployment rollout failed"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment
        fi
        return 1
    fi
}

# Error handling
trap 'log_error "Deployment script interrupted or failed"' ERR INT TERM

# Execute main function
main "$@"