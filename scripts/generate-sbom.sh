#!/bin/bash
# SBOM (Software Bill of Materials) generation for Secure MPC Transformer
# Generates comprehensive dependency tracking for security compliance

set -euo pipefail

# Configuration
PROJECT_NAME="secure-mpc-transformer"
OUTPUT_DIR="sbom"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Generating SBOM for ${PROJECT_NAME}${NC}"
echo "======================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Generate Python dependencies SBOM
generate_python_sbom() {
    echo -e "${YELLOW}Generating Python dependencies SBOM...${NC}"
    
    if command_exists pip; then
        # Generate requirements with versions
        pip freeze > "${OUTPUT_DIR}/requirements_frozen_${TIMESTAMP}.txt"
        
        # Generate pip-audit SBOM if available
        if command_exists pip-audit; then
            pip-audit --format=json --output="${OUTPUT_DIR}/pip_audit_${TIMESTAMP}.json"
            echo -e "${GREEN}✓ Generated pip-audit SBOM${NC}"
        else
            echo -e "${YELLOW}⚠ pip-audit not available, install with: pip install pip-audit${NC}"
        fi
        
        # Generate cyclonedx SBOM if available
        if command_exists cyclonedx-py; then
            cyclonedx-py --format json --output "${OUTPUT_DIR}/cyclonedx_python_${TIMESTAMP}.json"
            echo -e "${GREEN}✓ Generated CycloneDX Python SBOM${NC}"
        else
            echo -e "${YELLOW}⚠ cyclonedx-py not available, install with: pip install cyclonedx-bom${NC}"
        fi
    else
        echo -e "${RED}✗ pip not found${NC}"
    fi
}

# Generate container SBOM
generate_container_sbom() {
    echo -e "${YELLOW}Generating container SBOM...${NC}"
    
    # Check if Syft is available for container scanning
    if command_exists syft; then
        # Scan Docker images if they exist
        for dockerfile in docker/Dockerfile.*; do
            if [ -f "$dockerfile" ]; then
                image_name="${PROJECT_NAME}:$(basename ${dockerfile#*.})"
                
                # Build image if it doesn't exist
                if ! docker image inspect "$image_name" >/dev/null 2>&1; then
                    echo "Building $image_name for SBOM generation..."
                    docker build -f "$dockerfile" -t "$image_name" .
                fi
                
                # Generate SBOM
                syft "$image_name" -o json > "${OUTPUT_DIR}/syft_${image_name//[^a-zA-Z0-9]/_}_${TIMESTAMP}.json"
                echo -e "${GREEN}✓ Generated SBOM for $image_name${NC}"
            fi
        done
    else
        echo -e "${YELLOW}⚠ Syft not available for container scanning${NC}"
        echo "Install with: curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
    fi
}

# Generate system dependencies SBOM
generate_system_sbom() {
    echo -e "${YELLOW}Generating system dependencies SBOM...${NC}"
    
    # System package information
    if command_exists dpkg; then
        dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > "${OUTPUT_DIR}/dpkg_packages_${TIMESTAMP}.txt"
        echo -e "${GREEN}✓ Generated dpkg package list${NC}"
    fi
    
    if command_exists rpm; then
        rpm -qa --qf '%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n' > "${OUTPUT_DIR}/rpm_packages_${TIMESTAMP}.txt"
        echo -e "${GREEN}✓ Generated rpm package list${NC}"
    fi
    
    # Python environment info
    python3 -c "
import sys
import json
import platform

info = {
    'python_version': sys.version,
    'python_implementation': platform.python_implementation(),
    'platform': platform.platform(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'system': platform.system(),
    'timestamp': '${TIMESTAMP}'
}

with open('${OUTPUT_DIR}/python_env_${TIMESTAMP}.json', 'w') as f:
    json.dump(info, f, indent=2)
"
    echo -e "${GREEN}✓ Generated Python environment info${NC}"
}

# Generate project metadata
generate_project_metadata() {
    echo -e "${YELLOW}Generating project metadata...${NC}"
    
    # Git information
    if command_exists git && [ -d .git ]; then
        cat > "${OUTPUT_DIR}/project_metadata_${TIMESTAMP}.json" << EOF
{
    "project_name": "${PROJECT_NAME}",
    "timestamp": "${TIMESTAMP}",
    "git": {
        "commit": "$(git rev-parse HEAD)",
        "branch": "$(git rev-parse --abbrev-ref HEAD)",
        "tag": "$(git describe --tags --exact-match 2>/dev/null || echo 'none')",
        "remote": "$(git remote get-url origin 2>/dev/null || echo 'none')"
    },
    "build_environment": {
        "user": "${USER:-unknown}",
        "hostname": "${HOSTNAME:-unknown}",
        "pwd": "$(pwd)"
    }
}
EOF
        echo -e "${GREEN}✓ Generated project metadata${NC}"
    fi
}

# Generate security baseline
generate_security_baseline() {
    echo -e "${YELLOW}Generating security baseline...${NC}"
    
    # File checksums
    find src/ -type f -name "*.py" -exec sha256sum {} \; > "${OUTPUT_DIR}/source_checksums_${TIMESTAMP}.txt" 2>/dev/null || true
    
    # Security scan with bandit if available
    if command_exists bandit; then
        bandit -r src/ -f json -o "${OUTPUT_DIR}/bandit_scan_${TIMESTAMP}.json" 2>/dev/null || true
        echo -e "${GREEN}✓ Generated Bandit security scan${NC}"
    fi
    
    # Safety check if available
    if command_exists safety; then
        safety check --json --output "${OUTPUT_DIR}/safety_scan_${TIMESTAMP}.json" 2>/dev/null || true
        echo -e "${GREEN}✓ Generated Safety vulnerability scan${NC}"
    fi
}

# Generate comprehensive SBOM
generate_comprehensive_sbom() {
    echo -e "${YELLOW}Generating comprehensive SBOM summary...${NC}"
    
    cat > "${OUTPUT_DIR}/README_${TIMESTAMP}.md" << EOF
# Software Bill of Materials (SBOM)
## ${PROJECT_NAME}

Generated on: ${TIMESTAMP}

This directory contains comprehensive software bill of materials for the ${PROJECT_NAME} project.

## Files Generated

### Python Dependencies
- \`requirements_frozen_${TIMESTAMP}.txt\` - Frozen pip requirements
- \`pip_audit_${TIMESTAMP}.json\` - Pip audit security scan (if available)
- \`cyclonedx_python_${TIMESTAMP}.json\` - CycloneDX format SBOM (if available)

### Container Images
- \`syft_*.json\` - Container image SBOMs (if Syft available)

### System Dependencies
- \`dpkg_packages_${TIMESTAMP}.txt\` - Debian package list (if applicable)
- \`rpm_packages_${TIMESTAMP}.txt\` - RPM package list (if applicable)
- \`python_env_${TIMESTAMP}.json\` - Python environment information

### Project Metadata
- \`project_metadata_${TIMESTAMP}.json\` - Git and build environment info

### Security Information
- \`source_checksums_${TIMESTAMP}.txt\` - Source code checksums
- \`bandit_scan_${TIMESTAMP}.json\` - Bandit security scan (if available)
- \`safety_scan_${TIMESTAMP}.json\` - Safety vulnerability scan (if available)

## Usage

This SBOM can be used for:
- Supply chain security analysis
- Vulnerability tracking
- Compliance reporting
- Dependency management
- Security auditing

## Tools Used

The following tools were used to generate this SBOM:
- pip (Python package manager)
- pip-audit (Python vulnerability scanner)
- cyclonedx-py (CycloneDX SBOM generator)
- syft (Container image analyzer)
- bandit (Python security linter)
- safety (Python vulnerability checker)

Install missing tools as needed for complete SBOM generation.
EOF
    
    echo -e "${GREEN}✓ Generated comprehensive SBOM documentation${NC}"
}

# Main execution
main() {
    generate_python_sbom
    generate_container_sbom
    generate_system_sbom
    generate_project_metadata
    generate_security_baseline
    generate_comprehensive_sbom
    
    echo ""
    echo -e "${GREEN}SBOM generation complete!${NC}"
    echo -e "Output directory: ${BLUE}${OUTPUT_DIR}/${NC}"
    echo -e "Files generated: ${BLUE}$(ls -1 ${OUTPUT_DIR}/ | wc -l)${NC}"
    echo ""
    echo "To view the complete SBOM:"
    echo "  cat ${OUTPUT_DIR}/README_${TIMESTAMP}.md"
    echo ""
    echo "To install missing tools:"
    echo "  pip install pip-audit cyclonedx-bom bandit safety"
    echo "  curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
}

# Execute main function
main "$@"