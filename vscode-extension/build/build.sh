#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$SCRIPT_DIR/../context-engine-uploader"
OUT_DIR="$SCRIPT_DIR/../out"
SRC_SCRIPT="$SCRIPT_DIR/../../scripts/standalone_upload_client.py"
CLIENT="standalone_upload_client.py"
STAGE_DIR="$OUT_DIR/extension-stage"
BUNDLE_DEPS="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOOK_SRC="$SCRIPT_DIR/../../ctx-hook-simple.sh"
CTX_SRC="$SCRIPT_DIR/../../scripts/ctx.py"

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

# Bundle ctx hook script and ctx CLI into the staged extension for reference
if [[ -f "$HOOK_SRC" ]]; then
    cp "$HOOK_SRC" "$STAGE_DIR/ctx-hook-simple.sh"
    chmod +x "$STAGE_DIR/ctx-hook-simple.sh"
fi
if [[ -f "$CTX_SRC" ]]; then
    cp "$CTX_SRC" "$STAGE_DIR/ctx.py"
fi

# Optional: bundle Python deps into the staged extension when requested
if [[ "$BUNDLE_DEPS" == "--bundle-deps" ]]; then
    echo "Bundling Python dependencies into staged extension using $PYTHON_BIN..."
    "$PYTHON_BIN" -m pip install -t "$STAGE_DIR/python_libs" requests urllib3 charset_normalizer
fi

pushd "$STAGE_DIR" >/dev/null
echo "Packaging extension..."
npx @vscode/vsce package --no-dependencies --out "$OUT_DIR"
popd >/dev/null

echo "Build complete! Check the /out directory for .vsix and .py files."
ls -la "$OUT_DIR"