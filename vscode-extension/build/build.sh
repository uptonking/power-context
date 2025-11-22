#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$SCRIPT_DIR/../context-engine-uploader"
OUT_DIR="$SCRIPT_DIR/../out"
SRC_SCRIPT="$SCRIPT_DIR/../../scripts/standalone_upload_client.py"
CLIENT="standalone_upload_client.py"
STAGE_DIR="$OUT_DIR/extension-stage"

cleanup() {
    rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "Building clean Context Engine Uploader extension..."

mkdir -p "$OUT_DIR"

# Ensure extension directory is clean
rm -f "$EXT_DIR/$CLIENT"

# Copy upload client to the distributable out directory
cp "$SRC_SCRIPT" "$OUT_DIR/$CLIENT"

# Prepare staging directory
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp -a "$EXT_DIR/." "$STAGE_DIR/"

# Inject the upload client into the staged extension for packaging
cp "$OUT_DIR/$CLIENT" "$STAGE_DIR/$CLIENT"
chmod +x "$STAGE_DIR/$CLIENT"

pushd "$STAGE_DIR" >/dev/null
echo "Packaging extension..."
npx @vscode/vsce package --no-dependencies --out "$OUT_DIR"
popd >/dev/null

echo "Build complete! Check the /out directory for .vsix and .py files."
ls -la "$OUT_DIR"