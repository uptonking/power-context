# Docker build for clean VSCode extension
FROM node:18-alpine

# Install bash/coreutils for staging logic and vsce for packaging
RUN apk add --no-cache bash coreutils \
    && npm install -g @vscode/vsce

# Copy entire repository into the image
WORKDIR /workspace
COPY ../.. /workspace

# Run the shared build script to produce a clean VSIX into /workspace/vscode-extension/out
WORKDIR /workspace/vscode-extension/build
RUN bash build.sh

# Expose the output directory as the final working directory
WORKDIR /workspace/vscode-extension/out