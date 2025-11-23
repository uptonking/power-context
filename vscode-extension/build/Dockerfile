# Docker build for VSCode extension
FROM node:18-alpine

ARG BUNDLE_DEPS=false

# Install bash/coreutils and Python for staging logic and optional dependency bundling, plus vsce for packaging
RUN apk add --no-cache bash coreutils python3 py3-pip \
    && npm install -g @vscode/vsce

# Copy entire repository into the image
WORKDIR /workspace
COPY ../.. /workspace

# Run the shared build script to produce a VSIX into /workspace/vscode-extension/out
WORKDIR /workspace/vscode-extension/build
RUN if [ "$BUNDLE_DEPS" = "true" ]; then bash build.sh --bundle-deps; else bash build.sh; fi

# Expose the output directory as the final working directory
WORKDIR /workspace/vscode-extension/out