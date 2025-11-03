#!/bin/bash

# Context-Engine Kubernetes Cleanup Script
# This script removes all Context-Engine resources from Kubernetes

set -e

# Configuration
NAMESPACE="context-engine"
FORCE=false

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

# Confirm cleanup
confirm_cleanup() {
    if [[ "$FORCE" != "true" ]]; then
        log_warning "This will delete all Context-Engine resources in namespace: $NAMESPACE"
        read -p "Are you sure you want to continue? (yes/no): " -r
        echo
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log_info "Cleanup cancelled"
            exit 0
        fi
    fi
}

# Delete resources
cleanup_resources() {
    log_info "Cleaning up Context-Engine resources..."

    # Delete deployments
    log_info "Deleting deployments..."
    kubectl delete deployment --all -n $NAMESPACE --ignore-not-found=true

    # Delete statefulsets
    log_info "Deleting statefulsets..."
    kubectl delete statefulset --all -n $NAMESPACE --ignore-not-found=true

    # Delete jobs
    log_info "Deleting jobs..."
    kubectl delete job --all -n $NAMESPACE --ignore-not-found=true

    # Delete services
    log_info "Deleting services..."
    kubectl delete service --all -n $NAMESPACE --ignore-not-found=true

    # Delete ingress
    log_info "Deleting ingress..."
    kubectl delete ingress --all -n $NAMESPACE --ignore-not-found=true

    # Delete configmaps
    log_info "Deleting configmaps..."
    kubectl delete configmap --all -n $NAMESPACE --ignore-not-found=true

    # Delete secrets
    log_info "Deleting secrets..."
    kubectl delete secret --all -n $NAMESPACE --ignore-not-found=true

    # Delete PVCs
    log_info "Deleting persistent volume claims..."
    kubectl delete pvc --all -n $NAMESPACE --ignore-not-found=true

    # Delete namespace
    log_info "Deleting namespace..."
    kubectl delete namespace $NAMESPACE --ignore-not-found=true

    log_success "Cleanup complete!"
}

# Help function
show_help() {
    echo "Context-Engine Kubernetes Cleanup Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  --namespace NAMESPACE     Kubernetes namespace (default: context-engine)"
    echo "  --force                   Skip confirmation prompt"
    echo
    echo "Examples:"
    echo "  $0                        # Interactive cleanup"
    echo "  $0 --force                # Force cleanup without confirmation"
    echo "  $0 --namespace my-ns      # Cleanup specific namespace"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main cleanup function
main() {
    log_info "Starting Context-Engine Kubernetes cleanup"

    # Check prerequisites
    check_kubectl

    # Confirm cleanup
    confirm_cleanup

    # Cleanup resources
    cleanup_resources
}

# Run main cleanup
main

