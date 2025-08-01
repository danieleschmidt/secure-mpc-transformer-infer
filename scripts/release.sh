#!/bin/bash
# Release automation script for Secure MPC Transformer
# Handles version bumping, tagging, and release preparation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="secure-mpc-transformer"
PYPROJECT_FILE="pyproject.toml"
CHANGELOG_FILE="CHANGELOG.md"

# Default values
RELEASE_TYPE="patch"
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false

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
Release automation script for Secure MPC Transformer

Usage: $0 [OPTIONS]

Options:
    -t, --type TYPE        Release type (major, minor, patch) [default: patch]
    -d, --dry-run          Show what would be done without making changes
    --skip-tests           Skip running tests before release
    --skip-build           Skip building artifacts
    -h, --help             Show this help message

Release types:
    major                  Increment major version (x.0.0)
    minor                  Increment minor version (x.y.0)
    patch                  Increment patch version (x.y.z)

Examples:
    $0 -t minor            Create minor version release
    $0 --dry-run           Preview release without changes
    $0 -t major --skip-tests  Major release without running tests

Environment variables:
    GITHUB_TOKEN          Required for GitHub release creation
    GPG_KEY_ID           Optional GPG key ID for signing
EOF
}

get_current_version() {
    python -c "import tomllib; print(tomllib.load(open('${PYPROJECT_FILE}', 'rb'))['project']['version'])" 2>/dev/null || {
        log_error "Could not read version from ${PYPROJECT_FILE}"
        exit 1
    }
}

bump_version() {
    local current_version=$1
    local release_type=$2
    
    # Parse version components
    IFS='.' read -ra VERSION_PARTS <<< "$current_version"
    local major=${VERSION_PARTS[0]}
    local minor=${VERSION_PARTS[1]}
    local patch=${VERSION_PARTS[2]}
    
    # Bump according to release type
    case "$release_type" in
        "major")
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        "minor")
            minor=$((minor + 1))
            patch=0
            ;;
        "patch")
            patch=$((patch + 1))
            ;;
        *)
            log_error "Invalid release type: $release_type"
            exit 1
            ;;
    esac
    
    echo "${major}.${minor}.${patch}"
}

update_version_file() {
    local new_version=$1
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would update version to ${new_version} in ${PYPROJECT_FILE}"
    else
        log_info "Updating version to ${new_version} in ${PYPROJECT_FILE}"
        
        # Use Python to update version in pyproject.toml
        python << EOF
import tomllib
import tomli_w

with open('${PYPROJECT_FILE}', 'rb') as f:
    data = tomllib.load(f)

data['project']['version'] = '${new_version}'

with open('${PYPROJECT_FILE}', 'wb') as f:
    tomli_w.dump(data, f)
EOF
        
        log_success "Version updated successfully"
    fi
}

update_changelog() {
    local new_version=$1
    local release_date=$(date +%Y-%m-%d)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would update changelog for version ${new_version}"
        return
    fi
    
    log_info "Updating changelog for version ${new_version}"
    
    # Create a temporary file with the new entry
    local temp_file=$(mktemp)
    
    # Write new changelog entry
    cat > "$temp_file" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [${new_version}] - ${release_date}

### Added
- Automated release process
- Enhanced build and containerization scripts

### Changed
- Updated dependencies to latest versions

### Fixed
- Various bug fixes and performance improvements

EOF
    
    # Append existing changelog content (skip the first few lines)
    if [[ -f "$CHANGELOG_FILE" ]]; then
        tail -n +4 "$CHANGELOG_FILE" >> "$temp_file"
    fi
    
    # Replace original changelog
    mv "$temp_file" "$CHANGELOG_FILE"
    
    log_success "Changelog updated successfully"
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running test suite..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would run: make ci-check"
        return 0
    fi
    
    # Run comprehensive test suite
    if make ci-check; then
        log_success "All tests passed"
    else
        log_error "Tests failed - aborting release"
        exit 1
    fi
}

build_artifacts() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping build as requested"
        return 0
    fi
    
    log_info "Building release artifacts..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would run: make build"
        log_info "Would run: ./scripts/build.sh -t all"
        return 0
    fi
    
    # Build Python package
    make build || {
        log_error "Python package build failed"
        exit 1
    }
    
    # Build Docker images
    ./scripts/build.sh -t all || {
        log_error "Docker image build failed"
        exit 1
    }
    
    log_success "Artifacts built successfully"
}

check_working_directory() {
    log_info "Checking working directory status..."
    
    # Check if git repo
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a Git repository"
        exit 1
    fi
    
    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        log_error "Working directory has uncommitted changes"
        log_info "Please commit or stash changes before creating a release"
        exit 1
    fi
    
    # Check current branch
    local current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "main" ]] && [[ "$current_branch" != "master" ]]; then
        log_warning "Not on main/master branch (currently on: $current_branch)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Release cancelled"
            exit 0
        fi
    fi
    
    log_success "Working directory check passed"
}

create_git_tag() {
    local new_version=$1
    local tag_name="v${new_version}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would create git tag: ${tag_name}"
        return 0
    fi
    
    log_info "Creating git tag: ${tag_name}"
    
    # Create signed tag if GPG key is available
    if [[ -n "${GPG_KEY_ID:-}" ]]; then
        git tag -s "$tag_name" -m "Release version ${new_version}" || {
            log_error "Failed to create signed tag"
            exit 1
        }
        log_success "Signed git tag created: ${tag_name}"
    else
        git tag "$tag_name" -m "Release version ${new_version}" || {
            log_error "Failed to create tag"
            exit 1
        }
        log_success "Git tag created: ${tag_name}"
    fi
}

push_changes() {
    local new_version=$1
    local tag_name="v${new_version}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would commit changes and push tag: ${tag_name}"
        return 0
    fi
    
    log_info "Committing release changes..."
    
    # Add changed files
    git add "$PYPROJECT_FILE" "$CHANGELOG_FILE"
    
    # Commit changes
    git commit -m "chore: release version ${new_version}

- Update version to ${new_version}
- Update changelog
- Prepare release artifacts

🤖 Generated with automated release script" || {
        log_error "Failed to commit release changes"
        exit 1
    }
    
    # Push commit and tag
    log_info "Pushing changes and tag to remote..."
    git push origin HEAD || {
        log_error "Failed to push commit"
        exit 1
    }
    
    git push origin "$tag_name" || {
        log_error "Failed to push tag"
        exit 1
    }
    
    log_success "Changes and tag pushed successfully"
}

create_github_release() {
    local new_version=$1
    local tag_name="v${new_version}"
    
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log_warning "GITHUB_TOKEN not set - skipping GitHub release creation"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would create GitHub release: ${tag_name}"
        return 0
    fi
    
    log_info "Creating GitHub release: ${tag_name}"
    
    # Extract changelog for this version
    local release_notes="Release version ${new_version}"
    
    # Create release using GitHub CLI if available
    if command -v gh &> /dev/null; then
        gh release create "$tag_name" \
            --title "Release ${new_version}" \
            --notes "$release_notes" \
            --latest || {
            log_warning "Failed to create GitHub release via CLI"
        }
    else
        log_warning "GitHub CLI not available - skipping release creation"
    fi
}

generate_release_summary() {
    local current_version=$1
    local new_version=$2
    
    log_info "Release Summary"
    echo "=================="
    echo "Project: $PROJECT_NAME"
    echo "Previous version: $current_version"
    echo "New version: $new_version"
    echo "Release type: $RELEASE_TYPE"
    echo "Tag: v${new_version}"
    echo ""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN - No changes were made"
    else
        echo "Release completed successfully!"
        echo ""
        echo "Next steps:"
        echo "1. Monitor CI/CD pipeline"
        echo "2. Verify Docker images are published"
        echo "3. Update documentation if needed"
        echo "4. Announce release to stakeholders"
    fi
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                RELEASE_TYPE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
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
    
    # Validate release type
    if [[ ! "$RELEASE_TYPE" =~ ^(major|minor|patch)$ ]]; then
        log_error "Invalid release type: $RELEASE_TYPE"
        show_help
        exit 1
    fi
    
    log_info "Starting release process..."
    log_info "Release type: $RELEASE_TYPE"
    log_info "Dry run: $DRY_RUN"
    
    # Pre-flight checks
    check_working_directory
    
    # Get current version and calculate new version
    local current_version=$(get_current_version)
    local new_version=$(bump_version "$current_version" "$RELEASE_TYPE")
    
    log_info "Current version: $current_version"
    log_info "New version: $new_version"
    
    # Confirm release
    if [[ "$DRY_RUN" != "true" ]]; then
        echo
        read -p "Proceed with release $current_version -> $new_version? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Release cancelled"
            exit 0
        fi
    fi
    
    # Execute release steps
    run_tests
    build_artifacts
    update_version_file "$new_version"
    update_changelog "$new_version"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        create_git_tag "$new_version"
        push_changes "$new_version"
        create_github_release "$new_version"
    fi
    
    # Generate summary
    generate_release_summary "$current_version" "$new_version"
    
    log_success "Release process completed!"
}

# Handle script interruption
trap 'log_error "Release process interrupted"; exit 1' INT TERM

# Check required dependencies
if ! python -c "import tomllib" &> /dev/null; then
    log_error "Python tomllib module not available (requires Python 3.11+)"
    log_info "Alternative: install tomli and tomli-w packages"
    exit 1
fi

# Run main function
main "$@"