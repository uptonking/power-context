# Git Sync Source Code Management for Context-Engine

This guide explains how to set up and configure Git-based source code synchronization for Context-Engine in Kubernetes deployments.

## Overview

The Git sync solution uses **Git sync sidecar containers** that automatically pull source code from a Git repository into the application pods. This solves the critical issue of source code distribution in remote Kubernetes clusters.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Pod                          │
│  ┌─────────────┐  ┌───────────────┐  ┌───────────────────┐ │
│  │ Main App    │  │ Git Sync      │  │ Shared Volume     │ │
│  │ Container   │  │ Sidecar       │  │ (emptyDir)        │ │
│  │             │  │               │  │                   │ │
│  │ /work       │←─→│ /git → /work  │←─→│ Source Code       │ │
│  └─────────────┘  └───────────────┘  └───────────────────┘ │
│         ↕                                                    │
│    Git Repository (GitHub/GitLab/Bitbucket)                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Deploy with Git Sync

```bash
# Public repository
./deploy-with-source.sh git https://github.com/your-org/your-repo.git main

# Private repository with HTTPS
./deploy-with-source.sh git https://github.com/your-org/your-repo.git main

# Private repository with SSH
./deploy-with-source.sh git git@github.com:your-org/your-repo.git main
```

### 2. Deploy with Local Mode (Alternative)

```bash
# Use this if source code is already on cluster nodes
./deploy-with-source.sh local
```

## Configuration Options

### ConfigMap Settings

Update `deploy/kubernetes/configmap.yaml` with your Git configuration:

```yaml
# Source Code Configuration
SOURCE_CODE_MODE: "git"  # Options: "local" or "git"

# Git repository configuration (only used when SOURCE_CODE_MODE=git)
GIT_REPO_URL: "https://github.com/your-org/your-repo.git"
GIT_BRANCH: "main"
GIT_SYNC_PERIOD: "60"  # Sync every 60 seconds
GIT_USERNAME: ""  # For private repos (optional)
GIT_PASSWORD: ""  # For private repos (optional)
```

### Git Sync Sidecar Configuration

The Git sync sidecar uses the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GITSYNC_REPO` | Git repository URL | From ConfigMap |
| `GITSYNC_BRANCH` | Git branch to checkout | From ConfigMap |
| `GITSYNC_ROOT` | Directory to clone into | `/git` |
| `GITSYNC_SYNC_PERIOD` | Sync frequency in seconds | From ConfigMap |
| `GITSYNC_ONE_TIME` | Sync once and exit | `false` |
| `GITSYNC_LINK` | Create symlink to latest commit | `latest` |
| `GITSYNC_MAX_FAILURES` | Max sync failures before giving up | `5` |
| `GITSYNC_USERNAME` | Username for HTTP basic auth | From ConfigMap |
| `GITSYNC_PASSWORD` | Password/token for HTTP basic auth | From ConfigMap |

## Authentication Setup

### Public Repositories

No additional setup required. The Git sync sidecar will clone public repositories automatically.

### Private Repositories (HTTPS)

1. **Using Personal Access Token:**

```bash
# Create ConfigMap with credentials
kubectl patch configmap context-engine-config -n context-engine --patch '{"data":{"GIT_USERNAME":"your-username","GIT_PASSWORD":"your-personal-access-token"}}'
```

2. **Alternative: Create Secret:**

```bash
kubectl create secret generic git-https-credentials \
  --from-literal=username=your-username \
  --from-literal=password=your-personal-access-token \
  -n context-engine
```

### Private Repositories (SSH)

1. **Generate SSH Key:**

```bash
ssh-keygen -t rsa -b 4096 -C "git-sync@context-engine" -f ~/.ssh/context_engine_git
```

2. **Add SSH Key to Git Repository:**

   - Copy the public key (`~/.ssh/context_engine_git.pub`)
   - Add it as a deploy key in your Git repository settings

3. **Create Kubernetes Secret:**

```bash
kubectl create secret generic git-ssh-key \
  --from-file=ssh-private-key=~/.ssh/context_engine_git \
  -n context-engine
```

4. **Update Git Sync Configuration:**

The manifests are already configured to use SSH when the `git-ssh-key` secret exists.

## Deployment Options

### Option 1: Automated Deployment Script

```bash
# Deploy with automated script
cd deploy/kubernetes
./deploy-with-source.sh git https://github.com/your-org/your-repo.git main
```

### Option 2: Manual Deployment with Kustomize

1. **Update ConfigMap:**

```yaml
# kustomization.yaml patches
apiVersion: v1
kind: ConfigMap
metadata:
  name: context-engine-config
data:
  SOURCE_CODE_MODE: "git"
  GIT_REPO_URL: "https://github.com/your-org/your-repo.git"
  GIT_BRANCH: "main"
```

2. **Apply Git-enabled manifests:**

```bash
kubectl apply -f mcp-memory-git.yaml
kubectl apply -f mcp-indexer-git.yaml
```

### Option 3: Switching Between Modes

To switch from local to Git mode (or vice versa):

```bash
# Update ConfigMap
kubectl patch configmap context-engine-config -n context-engine --type merge --patch '{"data":{"SOURCE_CODE_MODE":"git"}}'

# Redeploy affected services
kubectl rollout restart deployment/mcp-memory -n context-engine
kubectl rollout restart deployment/mcp-indexer -n context-engine
```

## Monitoring and Troubleshooting

### Check Git Sync Status

```bash
# Check Git sync logs for indexer
kubectl logs deployment/mcp-indexer -c git-sync -n context-engine

# Check Git sync logs for memory server
kubectl logs deployment/mcp-memory -c git-sync -n context-engine
```

### Common Issues

#### 1. Authentication Failures

**Error:** `authentication failed`

**Solution:**
- Verify SSH key is correctly configured
- Check that the deploy key has read access
- Ensure the SSH key format is correct

#### 2. Network Connectivity

**Error:** `unable to access '...'`

**Solution:**
- Check cluster network policies
- Verify firewall rules allow Git access
- Test connectivity from a pod in the cluster

#### 3. Repository Not Found

**Error:** `repository not found`

**Solution:**
- Verify the repository URL is correct
- Check that the repository exists
- Ensure the Git branch exists

#### 4. Sync Loop Issues

**Error:** Continuous sync failures

**Solution:**
- Check `GITSYNC_MAX_FAILURES` setting
- Examine Git sync logs for specific errors
- Verify repository permissions

### Health Checks

The Git sync sidecar doesn't have built-in health endpoints, but you can monitor:

```bash
# Check if source code is present
kubectl exec deployment/mcp-indexer -c mcp-indexer -n context-engine -- ls -la /work

# Check Git sync status
kubectl exec deployment/mcp-indexer -c git-sync -n context-engine -- cat /git/.git-sync
```

## Best Practices

### 1. Repository Management

- **Use specific branches:** Pin to specific branches for production
- **Tag releases:** Use Git tags for release deployments
- **Clean repository:** Avoid including large binary files in the repository

### 2. Security

- **Use read-only deploy keys:** Don't use SSH keys with write access
- **Rotate credentials:** Regularly rotate personal access tokens
- **Network policies:** Restrict pod network access as needed

### 3. Performance

- **Optimize sync frequency:** Adjust `GIT_SYNC_PERIOD` based on update frequency
- **Repository size:** Keep repository size reasonable for faster clones
- **Shallow clones:** Consider using `--depth 1` for large repositories

### 4. High Availability

- **Multiple replicas:** Git sync works with multiple pod replicas
- **Regional repositories:** Use Git mirrors for global deployments
- **Fallback strategies:** Consider local mode as fallback

## Advanced Configuration

### Custom Git Sync Options

You can customize the Git sync sidecar by editing the manifests:

```yaml
env:
- name: GITSYNC_DEPTH
  value: "1"  # Shallow clone for faster sync
- name: GITSYNC_GARBAGE_COLLECTION
  value: "true"  # Clean up old commits
- name: GITSYNC_ADD_USER
  value: "true"  # Set .gitconfig user info
```

### Webhook Integration

For instant updates, consider using webhooks with a custom controller:

```yaml
# This would require a custom webhook receiver
apiVersion: v1
kind: Service
metadata:
  name: git-webhook-receiver
spec:
  selector:
    app: git-webhook-receiver
  ports:
  - port: 8080
    targetPort: 8080
```

### Multi-Repository Setup

For complex projects requiring multiple repositories:

```yaml
# Add multiple Git sync sidecars
- name: git-sync-main
  env:
  - name: GITSYNC_REPO
    value: "https://github.com/your-org/main-repo.git"
  volumeMounts:
  - name: main-volume
    mountPath: /git-main

- name: git-sync-config
  env:
  - name: GITSYNC_REPO
    value: "https://github.com/your-org/config-repo.git"
  volumeMounts:
  - name: config-volume
    mountPath: /git-config
```

## Migration Guide

### From Local Mode to Git Mode

1. **Backup current data:**
```bash
kubectl exec deployment/mcp-indexer -c mcp-indexer -n context-engine -- tar czf /tmp/backup.tar.gz -C /work .
```

2. **Update configuration:**
```bash
kubectl patch configmap context-engine-config -n context-engine --type merge --patch '{"data":{"SOURCE_CODE_MODE":"git","GIT_REPO_URL":"https://github.com/your-org/your-repo.git"}}'
```

3. **Restart deployments:**
```bash
kubectl rollout restart deployment/mcp-indexer -n context-engine
kubectl rollout restart deployment/mcp-memory -n context-engine
```

4. **Verify sync:**
```bash
kubectl logs deployment/mcp-indexer -c git-sync -n context-engine -f
```

## Support

For issues with Git sync setup:

1. Check the [Git sync documentation](https://github.com/kubernetes/git-sync)
2. Review Kubernetes pod logs
3. Verify network connectivity to Git repository
4. Check authentication configuration
5. Validate ConfigMap settings

Remember that Git sync provides **automatic source code distribution** for remote Kubernetes deployments, eliminating the need for manual code synchronization across cluster nodes.