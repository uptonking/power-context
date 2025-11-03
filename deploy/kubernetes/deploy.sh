#!/bin/bash

# Context-Engine Kubernetes Deployment Script
# This script deploys Context-Engine services to Kubernetes

set -e

# Configuration
NAMESPACE="context-engine"
IMAGE_REGISTRY="context-engine"
IMAGE_TAG="latest"

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

# Check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Kubernetes connection verified"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    kubectl apply -f namespace.yaml
    log_success "Namespace created/verified"
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration"
    kubectl apply -f configmap.yaml
    log_success "Configuration deployed"
}

# Deploy core services
deploy_core() {
    log_info "Deploying core services"

    # Deploy Qdrant
    log_info "Deploying Qdrant database..."
    kubectl apply -f qdrant.yaml

    # Wait for Qdrant to be ready
    log_info "Waiting for Qdrant to be ready..."
    kubectl wait --for=condition=ready pod -l component=qdrant -n $NAMESPACE --timeout=300s || log_warning "Qdrant may not be ready yet"

    log_success "Core services deployed"
}

# Deploy MCP servers
deploy_mcp_servers() {
    log_info "Deploying MCP servers"

    # Deploy SSE versions
    kubectl apply -f mcp-memory.yaml
    kubectl apply -f mcp-indexer.yaml

    # Wait for MCP servers to be ready
    log_info "Waiting for MCP servers to be ready..."
    kubectl wait --for=condition=ready pod -l component=mcp-memory -n $NAMESPACE --timeout=300s || log_warning "MCP Memory may not be ready yet"
    kubectl wait --for=condition=ready pod -l component=mcp-indexer -n $NAMESPACE --timeout=300s || log_warning "MCP Indexer may not be ready yet"

    log_success "MCP servers deployed"
}

# Deploy HTTP servers (optional)
deploy_http_servers() {
    log_info "Deploying HTTP servers (optional)"
    kubectl apply -f mcp-http.yaml

    # Wait for HTTP servers to be ready
    kubectl wait --for=condition=ready pod -l component=mcp-memory-http -n $NAMESPACE --timeout=300s || log_warning "MCP Memory HTTP may not be ready yet"
    kubectl wait --for=condition=ready pod -l component=mcp-indexer-http -n $NAMESPACE --timeout=300s || log_warning "MCP Indexer HTTP may not be ready yet"

    log_success "HTTP servers deployed"
}

# Deploy indexer services
deploy_indexer_services() {
    log_info "Deploying indexer services"
    kubectl apply -f indexer-services.yaml

    log_success "Indexer services deployed"
}

# Deploy optional Llama.cpp service
deploy_llamacpp() {
    if [[ "$SKIP_LLAMACPP" != "true" ]]; then
        log_info "Deploying Llama.cpp service (optional)"
        kubectl apply -f llamacpp.yaml
        log_success "Llama.cpp service deployed"
    else
        log_warning "Skipping Llama.cpp deployment"
    fi
}

# Deploy Ingress (optional)
deploy_ingress() {
    if [[ "$DEPLOY_INGRESS" == "true" ]]; then
        log_info "Deploying Ingress"
        kubectl apply -f ingress.yaml
        log_success "Ingress deployed"
    else
        log_warning "Skipping Ingress deployment (set --deploy-ingress to enable)"
    fi
}

# Show deployment status
show_status() {
    log_info "Deployment status:"
    echo
    echo "Namespace: $NAMESPACE"
    echo
    echo "Pods:"
    kubectl get pods -n $NAMESPACE -o wide
    echo
    echo "Services:"
    kubectl get services -n $NAMESPACE
    echo

    log_success "Deployment complete!"
    echo
    log_info "Access URLs:"
    echo "  Qdrant: http://<node-ip>:30333"
    echo "  MCP Memory (SSE): http://<node-ip>:30800"
    echo "  MCP Memory (HTTP): http://<node-ip>:30804"
    echo "  MCP Indexer (SSE): http://<node-ip>:30802"
    echo "  MCP Indexer (HTTP): http://<node-ip>:30806"
    if [[ "$SKIP_LLAMACPP" != "true" ]]; then
        echo "  Llama.cpp: http://<node-ip>:30808"
    fi
}

# Main deployment function
main() {
    log_info "Starting Context-Engine Kubernetes deployment"

    # Check prerequisites
    check_kubectl

    # Deploy in order
    create_namespace
    deploy_config
    deploy_core
    deploy_mcp_servers
    deploy_http_servers
    deploy_indexer_services
    deploy_llamacpp
    deploy_ingress

    # Show status
    show_status
}

# Help function
show_help() {
    echo "Context-Engine Kubernetes Deployment Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -r, --registry REGISTRY       Docker image registry (default: context-engine)"
    echo "  -t, --tag TAG                 Docker image tag (default: latest)"
    echo "  --skip-llamacpp               Skip Llama.cpp deployment"
    echo "  --deploy-ingress              Deploy Ingress configuration"
    echo "  --namespace NAMESPACE         Kubernetes namespace (default: context-engine)"
    echo
    echo "Examples:"
    echo "  $0                            # Basic deployment"
    echo "  $0 --skip-llamacpp            # Skip Llama.cpp"
    echo "  $0 --deploy-ingress           # Deploy with Ingress"
    echo "  $0 -r myregistry.com -t v1.0  # Use custom image"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--registry)
            IMAGE_REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --skip-llamacpp)
            SKIP_LLAMACPP=true
            shift
            ;;
        --deploy-ingress)
            DEPLOY_INGRESS=true
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "qdrant.yaml" ]]; then
    log_error "Please run this script from the deploy/kubernetes directory"
    exit 1
fi

# Run main deployment
main

