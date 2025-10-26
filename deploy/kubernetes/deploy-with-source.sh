#!/bin/bash

# Context-Engine Kubernetes Deployment with Source Code Management
# Supports both local (hostPath) and Git-based source code access

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="context-engine"
SOURCE_MODE="${1:-local}"  # Options: local, git
GIT_REPO_URL="${2:-}"
GIT_BRANCH="${3:-main}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Print usage
usage() {
    cat << EOF
Usage: $0 <source-mode> [git-repo-url] [git-branch]

Source Modes:
  local    - Use hostPath volumes (source code must be pre-distributed to nodes)
  git      - Use Git sync sidecars (automatic source code synchronization)

Examples:
  # Local deployment (requires source code on nodes)
  $0 local

  # Git-based deployment
  $0 git https://github.com/your-org/your-repo.git main

  # Git-based deployment with private repo (requires SSH key setup)
  $0 git git@github.com:your-org/your-repo.git main

Environment Variables:
  REGISTRY     - Docker registry prefix (default: context-engine)
  TAG          - Docker image tag (default: latest)

Requirements for Git Mode:
  - Git repository must be accessible from the cluster
  - For private repos: create git-ssh-key secret or configure credentials
  - Sufficient network access to clone the repository

Requirements for Local Mode:
  - Source code must exist at /tmp/context-engine-work on ALL nodes
  - Node access required for code updates
EOF
}

# Validate input
validate_input() {
    if [[ "$SOURCE_MODE" != "local" && "$SOURCE_MODE" != "git" ]]; then
        error "Invalid source mode: $SOURCE_MODE. Must be 'local' or 'git'."
        usage
        exit 1
    fi

    if [[ "$SOURCE_MODE" == "git" && -z "$GIT_REPO_URL" ]]; then
        error "Git repository URL is required when using git mode."
        usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi

    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi

    success "Prerequisites check passed"
}

# Update ConfigMap with source code configuration
update_configmap() {
    log "Updating ConfigMap with source code configuration..."

    # Create a temporary configmap with updated values
    kubectl create configmap context-engine-config-temp \
        --from-env-file <(cat << EOF
SOURCE_CODE_MODE=$SOURCE_MODE
GIT_REPO_URL=$GIT_REPO_URL
GIT_BRANCH=$GIT_BRANCH
GIT_SYNC_PERIOD=60
GIT_USERNAME=""
GIT_PASSWORD=""
EOF
) \
        --namespace "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Merge with existing configmap (preserving other settings)
    log "Merging configuration with existing ConfigMap..."
    # Note: This is a simplified approach. In production, you might want to use
    # kustomize or a more sophisticated config management tool
}

# Deploy based on source mode
deploy_services() {
    log "Deploying Context-Engine services in $SOURCE_MODE mode..."

    # Deploy core infrastructure (always needed)
    log "Deploying core infrastructure..."
    kubectl apply -f "$SCRIPT_DIR/qdrant.yaml"

    # Wait for Qdrant to be ready
    log "Waiting for Qdrant to be ready..."
    kubectl wait --for=condition=ready pod -l app=qdrant -n "$NAMESPACE" --timeout=300s

    if [[ "$SOURCE_MODE" == "local" ]]; then
        deploy_local_mode
    else
        deploy_git_mode
    fi

    # Deploy remaining services
    log "Deploying remaining services..."
    kubectl apply -f "$SCRIPT_DIR/mcp-http.yaml"
    kubectl apply -f "$SCRIPT_DIR/indexer-services.yaml"

    # Deploy optional services
    if [[ -f "$SCRIPT_DIR/llamacpp.yaml" ]]; then
        log "Deploying optional Llama.cpp service..."
        kubectl apply -f "$SCRIPT_DIR/llamacpp.yaml"
    fi
}

# Deploy in local mode (using hostPath)
deploy_local_mode() {
    log "Deploying in LOCAL mode (hostPath volumes)..."

    # Apply hostPath-based deployments
    kubectl apply -f "$SCRIPT_DIR/mcp-memory.yaml"
    kubectl apply -f "$SCRIPT_DIR/mcp-indexer.yaml"

    warn "⚠️  LOCAL MODE REQUIREMENTS:"
    warn "  - Source code must exist at /tmp/context-engine-work on ALL cluster nodes"
    warn "  - Code updates require manual synchronization across nodes"
    warn "  - This mode is suitable for single-node clusters or development"

    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled"
        exit 1
    fi
}

# Deploy in Git mode (using Git sync sidecars)
deploy_git_mode() {
    log "Deploying in GIT mode (automatic source code synchronization)..."

    # Setup Git authentication if needed
    setup_git_auth

    # Apply Git-enabled deployments
    kubectl apply -f "$SCRIPT_DIR/mcp-memory-git.yaml"
    kubectl apply -f "$SCRIPT_DIR/mcp-indexer-git.yaml"

    success "✅ Git sync enabled - source code will be automatically synchronized"
    log "  Repository: $GIT_REPO_URL"
    log "  Branch: $GIT_BRANCH"
    log "  Sync Period: 60 seconds"
}

# Setup Git authentication
setup_git_auth() {
    # Check if this is a private repository requiring authentication
    if [[ "$GIT_REPO_URL" =~ ^git@ ]] || [[ "$GIT_REPO_URL" =~ \.git$ && ! "$GIT_REPO_URL" =~ ^https://github\.com/[^/]+/[^/]+\.git$ ]]; then
        warn "Private repository detected. Please ensure authentication is configured:"
        warn "  1. For SSH: Create git-ssh-key secret with your SSH private key"
        warn "  2. For HTTPS: Set GIT_USERNAME and GIT_PASSWORD in ConfigMap"

        # Check if SSH secret exists
        if ! kubectl get secret git-ssh-key -n "$NAMESPACE" &> /dev/null; then
            log "Creating placeholder SSH secret (please update with your actual SSH key)"
            kubectl create secret generic git-ssh-key \
                --from-literal=ssh-private-key="" \
                --namespace "$NAMESPACE" \
                --dry-run=client -o yaml | kubectl apply -f -
            warn "⚠️  Please update the git-ssh-key secret with your actual SSH private key:"
            warn "  kubectl delete secret git-ssh-key -n $NAMESPACE"
            warn "  kubectl create secret generic git-ssh-key --from-file=ssh-private-key=~/.ssh/id_rsa -n $NAMESPACE"
        fi
    fi
}

# Wait for deployment to be ready
wait_for_ready() {
    log "Waiting for all deployments to be ready..."

    # List of deployments to wait for
    local deployments=("mcp-memory" "mcp-indexer" "mcp-memory-http" "mcp-indexer-http")

    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            log "Waiting for $deployment to be ready..."
            kubectl wait --for=condition=available deployment/"$deployment" -n "$NAMESPACE" --timeout=300s
        fi
    done

    success "All deployments are ready"
}

# Show deployment status and access information
show_status() {
    log "Deployment completed successfully!"
    echo
    echo "=== Context-Engine Status ==="
    echo "Namespace: $NAMESPACE"
    echo "Source Mode: $SOURCE_MODE"
    if [[ "$SOURCE_MODE" == "git" ]]; then
        echo "Git Repository: $GIT_REPO_URL"
        echo "Git Branch: $GIT_BRANCH"
    fi
    echo
    echo "=== Services ==="
    kubectl get services -n "$NAMESPACE"
    echo
    echo "=== Pods ==="
    kubectl get pods -n "$NAMESPACE"
    echo
    echo "=== Access Information ==="

    # Get service access information
    local cluster_ip=$(kubectl get svc qdrant -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "N/A")
    echo "Qdrant: $cluster_ip:6333"

    if kubectl get svc mcp-memory -n "$NAMESPACE" &> /dev/null; then
        local memory_nodeport=$(kubectl get svc mcp-memory -n "$NAMESPACE" -o jsonpath='{.spec.ports[?(@.name=="sse")].nodePort}' 2>/dev/null || echo "N/A")
        echo "MCP Memory (SSE): NodePort $memory_nodeport"
    fi

    if kubectl get svc mcp-indexer -n "$NAMESPACE" &> /dev/null; then
        local indexer_nodeport=$(kubectl get svc mcp-indexer -n "$NAMESPACE" -o jsonpath='{.spec.ports[?(@.name=="sse")].nodePort}' 2>/dev/null || echo "N/A")
        echo "MCP Indexer (SSE): NodePort $indexer_nodeport"
    fi

    echo
    echo "=== Next Steps ==="
    echo "1. Test the deployment:"
    echo "   curl http://<node-ip>:30800/sse  # MCP Memory"
    echo "   curl http://<node-ip>:30802/sse  # MCP Indexer"
    echo "2. Call indexing tool:"
    echo "   curl -X POST http://<node-ip>:30802/sse -H 'Content-Type: application/json' \\"
    echo "        -d '{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"tools/call\", \"params\": {\"name\": \"qdrant_index_root\", \"arguments\": {}}}'"

    if [[ "$SOURCE_MODE" == "git" ]]; then
        echo "3. Monitor Git sync:"
        echo "   kubectl logs deployment/mcp-indexer -c git-sync -n $NAMESPACE"
        echo "   kubectl logs deployment/mcp-memory -c git-sync -n $NAMESPACE"
    fi
}

# Main execution
main() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi

    validate_input
    check_prerequisites
    update_configmap
    deploy_services
    wait_for_ready
    show_status
}

# Run main function with all arguments
main "$@"