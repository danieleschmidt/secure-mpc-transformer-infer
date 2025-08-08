#!/bin/bash

# Multi-architecture Docker build script for Secure MPC Transformer
# Supports: linux/amd64, linux/arm64, linux/arm/v7, windows/amd64

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-ghcr.io/mpc-transformer}"
IMAGE_NAME="${IMAGE_NAME:-secure-mpc-transformer}"
VERSION="${VERSION:-v0.2.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Platform configurations
PLATFORMS_LINUX="linux/amd64,linux/arm64,linux/arm/v7"
PLATFORMS_WINDOWS="windows/amd64"
PLATFORMS_ALL="$PLATFORMS_LINUX,$PLATFORMS_WINDOWS"

# Build variants
VARIANTS=("cpu" "gpu" "prod" "dev")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Help function
show_help() {
    cat << EOF
Multi-architecture Docker build script for Secure MPC Transformer

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -r, --registry REGISTRY Registry to push images to (default: $REGISTRY)
    -n, --name NAME         Image name (default: $IMAGE_NAME)
    -v, --version VERSION   Image version (default: $VERSION)
    -p, --platforms LIST    Comma-separated list of platforms
    -b, --variants LIST     Comma-separated list of variants to build
    --push                  Push images to registry after building
    --load                  Load images to local Docker (single platform only)
    --no-cache             Build without using cache
    --dry-run              Show commands without executing

Platforms:
    linux/amd64    - Linux x86_64 (supports GPU)
    linux/arm64    - Linux ARM64 (CPU only)
    linux/arm/v7   - Linux ARMv7 (CPU only)  
    windows/amd64  - Windows x86_64 (CPU only)

Variants:
    cpu     - CPU-only build (all platforms)
    gpu     - GPU-enabled build (linux/amd64 only)
    prod    - Production build with minimal dependencies
    dev     - Development build with additional tools

Examples:
    # Build all variants for Linux platforms
    $0 --platforms "$PLATFORMS_LINUX"
    
    # Build GPU variant for amd64 and push
    $0 --variants gpu --platforms linux/amd64 --push
    
    # Build production variant for all platforms
    $0 --variants prod --platforms "$PLATFORMS_ALL" --push
    
    # Build and load CPU variant locally
    $0 --variants cpu --platforms linux/amd64 --load

EOF
}

# Parse command line arguments
PLATFORMS="$PLATFORMS_LINUX"
BUILD_VARIANTS=""
PUSH=false
LOAD=false
NO_CACHE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -p|--platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        -b|--variants)
            BUILD_VARIANTS="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --load)
            LOAD=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set default variants if not specified
if [[ -z "$BUILD_VARIANTS" ]]; then
    BUILD_VARIANTS="prod"
fi

# Convert comma-separated lists to arrays
IFS=',' read -ra PLATFORM_ARRAY <<< "$PLATFORMS"
IFS=',' read -ra VARIANT_ARRAY <<< "$BUILD_VARIANTS"

# Validate platforms
for platform in "${PLATFORM_ARRAY[@]}"; do
    case "$platform" in
        linux/amd64|linux/arm64|linux/arm/v7|windows/amd64)
            ;;
        *)
            log_error "Unsupported platform: $platform"
            exit 1
            ;;
    esac
done

# Validate variants
for variant in "${VARIANT_ARRAY[@]}"; do
    case "$variant" in
        cpu|gpu|prod|dev)
            ;;
        *)
            log_error "Unsupported variant: $variant"
            exit 1
            ;;
    esac
done

# Validate GPU variant is only built for amd64
for variant in "${VARIANT_ARRAY[@]}"; do
    if [[ "$variant" == "gpu" ]]; then
        gpu_compatible=false
        for platform in "${PLATFORM_ARRAY[@]}"; do
            if [[ "$platform" == "linux/amd64" ]]; then
                gpu_compatible=true
                break
            fi
        done
        if [[ "$gpu_compatible" != "true" ]]; then
            log_error "GPU variant can only be built for linux/amd64 platform"
            exit 1
        fi
    fi
done

# Function to execute commands
execute() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] $*"
    else
        log_info "Executing: $*"
        "$@"
    fi
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if buildx is available
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is not available"
        exit 1
    fi
    
    # Check if experimental features are enabled for multi-arch
    if ! docker buildx ls | grep -q "docker-container\|kubernetes"; then
        log_warning "No multi-platform builder found. Creating one..."
        execute docker buildx create --name multiarch-builder --use --bootstrap
    fi
    
    log_success "Prerequisites check passed"
}

# Function to prepare build context
prepare_build_context() {
    log_info "Preparing build context..."
    
    # Create platform-specific binary directories
    mkdir -p docker/platform-binaries/{amd64,arm64,armv7}
    
    # Copy platform-specific binaries (placeholder for now)
    for arch in amd64 arm64 armv7; do
        echo "#!/bin/bash" > "docker/platform-binaries/${arch}/platform-info"
        echo "echo 'Architecture: ${arch}'" >> "docker/platform-binaries/${arch}/platform-info"
        chmod +x "docker/platform-binaries/${arch}/platform-info"
    done
    
    # Create health check script if it doesn't exist
    if [[ ! -f "docker/healthcheck.sh" ]]; then
        cat > docker/healthcheck.sh << 'EOF'
#!/bin/bash
set -e

# Health check for MPC Transformer service
curl -f http://localhost:8080/health >/dev/null 2>&1 || exit 1
EOF
        chmod +x docker/healthcheck.sh
    fi
    
    log_success "Build context prepared"
}

# Function to build images
build_images() {
    local variant=$1
    local platforms=$2
    
    log_info "Building variant '$variant' for platforms: $platforms"
    
    # Build arguments
    local build_args=(
        "--build-arg" "VARIANT=$variant"
        "--build-arg" "BUILD_DATE=$BUILD_DATE"
        "--build-arg" "GIT_COMMIT=$GIT_COMMIT"
        "--build-arg" "VERSION=$VERSION"
    )
    
    # Cache arguments
    if [[ "$NO_CACHE" == "true" ]]; then
        build_args+=("--no-cache")
    fi
    
    # Platform arguments
    build_args+=("--platform" "$platforms")
    
    # Output arguments
    if [[ "$PUSH" == "true" ]]; then
        build_args+=("--push")
    elif [[ "$LOAD" == "true" ]]; then
        if [[ "$platforms" == *","* ]]; then
            log_error "Cannot load multi-platform build. Use single platform with --load"
            exit 1
        fi
        build_args+=("--load")
    fi
    
    # Tags
    local base_tag="$REGISTRY/$IMAGE_NAME"
    local tags=(
        "-t" "$base_tag:$VERSION-$variant"
        "-t" "$base_tag:latest-$variant"
    )
    
    # Add platform-specific tags for single platform builds
    if [[ "$platforms" != *","* ]]; then
        local platform_slug=$(echo "$platforms" | sed 's|/|-|g')
        tags+=("-t" "$base_tag:$VERSION-$variant-$platform_slug")
    fi
    
    # Build command
    local build_cmd=(
        "docker" "buildx" "build"
        "${build_args[@]}"
        "${tags[@]}"
        "-f" "docker/Dockerfile.multiarch"
        "."
    )
    
    execute "${build_cmd[@]}"
    
    log_success "Built variant '$variant' successfully"
}

# Function to create and push manifest
create_manifest() {
    if [[ "$PUSH" != "true" ]]; then
        return
    fi
    
    log_info "Creating multi-architecture manifests..."
    
    for variant in "${VARIANT_ARRAY[@]}"; do
        local base_tag="$REGISTRY/$IMAGE_NAME"
        local manifest_tags=(
            "$base_tag:$VERSION-$variant"
            "$base_tag:latest-$variant"
        )
        
        for tag in "${manifest_tags[@]}"; do
            log_info "Creating manifest for $tag"
            
            # The buildx build with --push already creates the manifest
            # This step is mainly for verification
            if [[ "$DRY_RUN" != "true" ]]; then
                docker buildx imagetools inspect "$tag" > /dev/null || {
                    log_error "Failed to verify manifest for $tag"
                    exit 1
                }
            fi
        done
    done
    
    log_success "Manifests created successfully"
}

# Function to test built images
test_images() {
    if [[ "$LOAD" != "true" ]]; then
        log_info "Skipping image tests (images not loaded locally)"
        return
    fi
    
    log_info "Testing built images..."
    
    for variant in "${VARIANT_ARRAY[@]}"; do
        local test_image="$REGISTRY/$IMAGE_NAME:$VERSION-$variant"
        
        log_info "Testing image: $test_image"
        
        # Basic smoke test
        if [[ "$DRY_RUN" != "true" ]]; then
            docker run --rm "$test_image" python -c "import secure_mpc_transformer; print('✓ Import successful')" || {
                log_error "Smoke test failed for $test_image"
                exit 1
            }
        fi
    done
    
    log_success "Image tests passed"
}

# Function to cleanup
cleanup() {
    log_info "Cleaning up build artifacts..."
    
    # Remove platform-specific binaries
    rm -rf docker/platform-binaries
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting multi-architecture build for Secure MPC Transformer"
    log_info "Registry: $REGISTRY"
    log_info "Image: $IMAGE_NAME"
    log_info "Version: $VERSION"
    log_info "Platforms: $PLATFORMS"  
    log_info "Variants: $BUILD_VARIANTS"
    log_info "Push: $PUSH"
    log_info "Load: $LOAD"
    log_info "No cache: $NO_CACHE"
    log_info "Dry run: $DRY_RUN"
    
    # Check prerequisites
    check_prerequisites
    
    # Prepare build context
    prepare_build_context
    
    # Build each variant
    for variant in "${VARIANT_ARRAY[@]}"; do
        # Filter platforms for GPU variant
        local build_platforms="$PLATFORMS"
        if [[ "$variant" == "gpu" ]]; then
            build_platforms="linux/amd64"
        fi
        
        build_images "$variant" "$build_platforms"
    done
    
    # Create manifests
    create_manifest
    
    # Test images
    test_images
    
    # Cleanup
    cleanup
    
    log_success "Multi-architecture build completed successfully!"
    
    # Show final summary
    echo ""
    log_info "Built images:"
    for variant in "${VARIANT_ARRAY[@]}"; do
        echo "  • $REGISTRY/$IMAGE_NAME:$VERSION-$variant"
        echo "  • $REGISTRY/$IMAGE_NAME:latest-$variant"
    done
    
    if [[ "$PUSH" == "true" ]]; then
        echo ""
        log_info "Images have been pushed to registry: $REGISTRY"
    fi
}

# Trap for cleanup on exit
trap cleanup EXIT

# Execute main function
main "$@"