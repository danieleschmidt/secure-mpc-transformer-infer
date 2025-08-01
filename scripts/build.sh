#!/bin/bash
# Build script for Secure MPC Transformer project
# Supports multiple build targets and configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="secure-mpc-transformer"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost}"
VERSION="${VERSION:-$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo 'dev')}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')

# Default values
BUILD_TARGET="all"
PUSH=false
NO_CACHE=false
PLATFORM="linux/amd64"

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

show_help() {
    cat << EOF
Build script for Secure MPC Transformer project

Usage: $0 [OPTIONS]

Options:
    -t, --target TARGET     Build target (cpu, gpu, dev, all) [default: all]
    -p, --push             Push images to registry after build
    -r, --registry URL     Docker registry URL [default: localhost]
    -v, --version VERSION  Version tag [default: auto-detected]
    --no-cache             Build without using cache
    --platform PLATFORM    Target platform [default: linux/amd64]
    -h, --help             Show this help message

Build targets:
    cpu                    CPU-only production image
    gpu                    GPU-enabled production image  
    dev                    Development image with tools
    all                    Build all images

Examples:
    $0 -t cpu              Build CPU image only
    $0 -t gpu --push       Build and push GPU image
    $0 --no-cache          Build all images without cache
    $0 -r my-registry.com  Build and tag for custom registry

Environment variables:
    DOCKER_REGISTRY        Registry URL (overrides -r)
    VERSION               Version tag (overrides -v)
    DOCKER_BUILDKIT       Enable BuildKit (recommended: 1)
EOF
}

check_requirements() {
    log_info "Checking build requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check for multi-platform support if needed
    if [[ "$PLATFORM" != "linux/amd64" ]]; then
        if ! docker buildx version &> /dev/null; then
            log_error "Docker Buildx is required for multi-platform builds"
            exit 1
        fi
    fi
    
    log_success "Requirements check passed"
}

build_image() {
    local target=$1
    local dockerfile=$2
    local image_tag="${DOCKER_REGISTRY}/${PROJECT_NAME}:${target}-${VERSION}"
    local latest_tag="${DOCKER_REGISTRY}/${PROJECT_NAME}:${target}"
    
    log_info "Building ${target} image: ${image_tag}"
    
    # Build arguments
    local build_args=(
        --file "${dockerfile}"
        --tag "${image_tag}"
        --tag "${latest_tag}"
        --label "org.opencontainers.image.created=${BUILD_DATE}"
        --label "org.opencontainers.image.version=${VERSION}"
        --label "org.opencontainers.image.revision=${GIT_COMMIT}"
        --label "org.opencontainers.image.title=Secure MPC Transformer (${target})"
        --platform "${PLATFORM}"
    )
    
    # Add no-cache flag if requested
    if [[ "$NO_CACHE" == "true" ]]; then
        build_args+=(--no-cache)
    fi
    
    # Enable BuildKit for better performance
    export DOCKER_BUILDKIT=1
    
    # Build the image
    if docker build "${build_args[@]}" .; then
        log_success "Successfully built ${target} image"
        
        # Show image info
        log_info "Image size: $(docker images --format "table {{.Size}}" "${image_tag}" | tail -n 1)"
        
        # Push if requested
        if [[ "$PUSH" == "true" ]]; then
            log_info "Pushing ${image_tag}..."
            docker push "${image_tag}"
            docker push "${latest_tag}"
            log_success "Successfully pushed ${target} image"
        fi
        
        return 0
    else
        log_error "Failed to build ${target} image"
        return 1
    fi
}

build_cpu() {
    build_image "cpu" "docker/Dockerfile.cpu"
}

build_gpu() {
    # Check for NVIDIA Docker support
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_warning "NVIDIA Docker support not detected. GPU image may not work properly."
    fi
    
    build_image "gpu" "docker/Dockerfile.gpu"
}

build_dev() {
    build_image "dev" "docker/Dockerfile.dev"
}

validate_images() {
    log_info "Validating built images..."
    
    local images_to_check=()
    
    case "$BUILD_TARGET" in
        "cpu")
            images_to_check=("${DOCKER_REGISTRY}/${PROJECT_NAME}:cpu-${VERSION}")
            ;;
        "gpu")
            images_to_check=("${DOCKER_REGISTRY}/${PROJECT_NAME}:gpu-${VERSION}")
            ;;
        "dev")
            images_to_check=("${DOCKER_REGISTRY}/${PROJECT_NAME}:dev-${VERSION}")
            ;;
        "all")
            images_to_check=(
                "${DOCKER_REGISTRY}/${PROJECT_NAME}:cpu-${VERSION}"
                "${DOCKER_REGISTRY}/${PROJECT_NAME}:gpu-${VERSION}"
                "${DOCKER_REGISTRY}/${PROJECT_NAME}:dev-${VERSION}"
            )
            ;;
    esac
    
    local validation_failed=false
    
    for image in "${images_to_check[@]}"; do
        log_info "Validating ${image}..."
        
        # Basic smoke test
        if docker run --rm "${image}" python -c "import secure_mpc_transformer; print('Import successful')" &> /dev/null; then
            log_success "✓ ${image} validation passed"
        else
            log_error "✗ ${image} validation failed"
            validation_failed=true
        fi
    done
    
    if [[ "$validation_failed" == "true" ]]; then
        log_error "Image validation failed"
        return 1
    else
        log_success "All images validated successfully"
        return 0
    fi
}

generate_build_report() {
    log_info "Generating build report..."
    
    local report_file="build-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "${report_file}" << EOF
{
  "project": "${PROJECT_NAME}",
  "version": "${VERSION}",
  "build_date": "${BUILD_DATE}",
  "git_commit": "${GIT_COMMIT}",
  "build_target": "${BUILD_TARGET}",
  "platform": "${PLATFORM}",
  "registry": "${DOCKER_REGISTRY}",
  "images": [
EOF

    local first=true
    case "$BUILD_TARGET" in
        "cpu")
            echo "    \"${DOCKER_REGISTRY}/${PROJECT_NAME}:cpu-${VERSION}\"" >> "${report_file}"
            ;;
        "gpu")
            echo "    \"${DOCKER_REGISTRY}/${PROJECT_NAME}:gpu-${VERSION}\"" >> "${report_file}"
            ;;
        "dev")
            echo "    \"${DOCKER_REGISTRY}/${PROJECT_NAME}:dev-${VERSION}\"" >> "${report_file}"
            ;;
        "all")
            echo "    \"${DOCKER_REGISTRY}/${PROJECT_NAME}:cpu-${VERSION}\"," >> "${report_file}"
            echo "    \"${DOCKER_REGISTRY}/${PROJECT_NAME}:gpu-${VERSION}\"," >> "${report_file}"
            echo "    \"${DOCKER_REGISTRY}/${PROJECT_NAME}:dev-${VERSION}\"" >> "${report_file}"
            ;;
    esac

    cat >> "${report_file}" << EOF
  ],
  "build_successful": true
}
EOF

    log_success "Build report generated: ${report_file}"
}

cleanup() {
    log_info "Cleaning up dangling images..."
    docker image prune -f &> /dev/null || true
    log_success "Cleanup completed"
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            -p|--push)
                PUSH=true
                shift
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
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
    
    # Validate build target
    if [[ ! "$BUILD_TARGET" =~ ^(cpu|gpu|dev|all)$ ]]; then
        log_error "Invalid build target: $BUILD_TARGET"
        show_help
        exit 1
    fi
    
    log_info "Starting build process..."
    log_info "Project: ${PROJECT_NAME}"
    log_info "Version: ${VERSION}"
    log_info "Target: ${BUILD_TARGET}"
    log_info "Registry: ${DOCKER_REGISTRY}"
    log_info "Platform: ${PLATFORM}"
    log_info "Push: ${PUSH}"
    
    # Check requirements
    check_requirements
    
    # Build images
    case "$BUILD_TARGET" in
        "cpu")
            build_cpu || exit 1
            ;;
        "gpu")
            build_gpu || exit 1
            ;;
        "dev")
            build_dev || exit 1
            ;;
        "all")
            build_cpu || exit 1
            build_gpu || exit 1
            build_dev || exit 1
            ;;
    esac
    
    # Validate images
    validate_images || exit 1
    
    # Generate build report
    generate_build_report
    
    # Cleanup
    cleanup
    
    log_success "Build process completed successfully!"
    log_info "Images built:"
    
    case "$BUILD_TARGET" in
        "cpu")
            echo "  - ${DOCKER_REGISTRY}/${PROJECT_NAME}:cpu-${VERSION}"
            ;;
        "gpu")
            echo "  - ${DOCKER_REGISTRY}/${PROJECT_NAME}:gpu-${VERSION}"
            ;;
        "dev")
            echo "  - ${DOCKER_REGISTRY}/${PROJECT_NAME}:dev-${VERSION}"
            ;;
        "all")
            echo "  - ${DOCKER_REGISTRY}/${PROJECT_NAME}:cpu-${VERSION}"
            echo "  - ${DOCKER_REGISTRY}/${PROJECT_NAME}:gpu-${VERSION}"
            echo "  - ${DOCKER_REGISTRY}/${PROJECT_NAME}:dev-${VERSION}"
            ;;
    esac
    
    if [[ "$PUSH" == "true" ]]; then
        log_info "Images have been pushed to ${DOCKER_REGISTRY}"
    fi
}

# Handle script interruption
trap 'log_error "Build interrupted"; exit 1' INT TERM

# Run main function
main "$@"