#!/bin/bash

# Simplified Claude Code UserPromptSubmit hook for ctx.py
# Takes JSON input from Claude Code and outputs enhanced prompt

# Read JSON input from stdin
INPUT=$(cat)

# Extract the user message using jq
if command -v jq >/dev/null 2>&1; then
    USER_MESSAGE=$(echo "$INPUT" | jq -r '.user_message')
else
    echo "$INPUT"
    exit 0
fi

# Skip if empty message
if [ -z "$USER_MESSAGE" ] || [ "$USER_MESSAGE" = "null" ]; then
    echo "$INPUT"
    exit 0
fi

# Easy bypass patterns - any of these will skip ctx enhancement
if [[ "$USER_MESSAGE" =~ ^(noctx|raw|bypass|skip|no-enhance): ]] || \
   [[ "$USER_MESSAGE" =~ ^\\ ]] || \
   [[ "$USER_MESSAGE" =~ ^\< ]] || \
   [[ "$USER_MESSAGE" =~ ^(/help|/clear|/exit|/quit) ]] || \
   [[ "$USER_MESSAGE" =~ ^\?\s*$ ]] || \
   [ ${#USER_MESSAGE} -lt 12 ]; then
    echo "$INPUT"
    exit 0
fi

# Set working directory to where the hook script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Read all settings from ctx_config.json
CONFIG_FILE="ctx_config.json"
if [ -f "$CONFIG_FILE" ]; then
    CTX_COLLECTION=$(grep -o '"default_collection"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"default_collection"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    REFRAG_RUNTIME=$(grep -o '"refrag_runtime"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"refrag_runtime"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "glm")
    GLM_API_KEY=$(grep -o '"glm_api_key"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"glm_api_key"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    GLM_API_BASE=$(grep -o '"glm_api_base"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"glm_api_base"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    GLM_MODEL=$(grep -o '"glm_model"[[:space:]]*:[[:space:]]*"[^\"]*"' "$CONFIG_FILE" | sed 's/.*"glm_model"[[:space:]]*:[[:space:]]*"\([^\"]*\)".*/\1/' || echo "glm-4.6")
    CTX_DEFAULT_MODE=$(grep -o '"default_mode"[[:space:]]*:[[:space:]]*"[^\"]*"' "$CONFIG_FILE" | sed 's/.*"default_mode"[[:space:]]*:[[:space:]]*"\([^\"]*\)".*/\1/')
    CTX_REQUIRE_CONTEXT=$(grep -o '"require_context"[[:space:]]*:[[:space:]]*\(true\|false\)' "$CONFIG_FILE" | sed 's/.*"require_context"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/')
    CTX_RELEVANCE_GATE=$(grep -o '"relevance_gate_enabled"[[:space:]]*:[[:space:]]*\(true\|false\)' "$CONFIG_FILE" | sed 's/.*"relevance_gate_enabled"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/')
    CTX_MIN_RELEVANCE=$(grep -o '"min_relevance"[[:space:]]*:[[:space:]]*[0-9.][0-9.]*' "$CONFIG_FILE" | sed 's/.*"min_relevance"[[:space:]]*:[[:space:]]*\([0-9.][0-9.]*\).*/\1/')
    CTX_REWRITE_MAX_TOKENS=$(grep -o '"rewrite_max_tokens"[[:space:]]*:[[:space:]]*[0-9][0-9]*' "$CONFIG_FILE" | sed 's/.*"rewrite_max_tokens"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/')
fi

# Set defaults if not found in config
CTX_COLLECTION=${CTX_COLLECTION:-"codebase"}
REFRAG_RUNTIME=${REFRAG_RUNTIME:-"glm"}
GLM_API_KEY=${GLM_API_KEY:-}
GLM_API_BASE=${GLM_API_BASE:-}
GLM_MODEL=${GLM_MODEL:-"glm-4.6"}
CTX_DEFAULT_MODE=${CTX_DEFAULT_MODE:-"default"}
CTX_REQUIRE_CONTEXT=${CTX_REQUIRE_CONTEXT:-true}
CTX_RELEVANCE_GATE=${CTX_RELEVANCE_GATE:-false}
CTX_MIN_RELEVANCE=${CTX_MIN_RELEVANCE:-0.1}
CTX_REWRITE_MAX_TOKENS=${CTX_REWRITE_MAX_TOKENS:-320}

# Export GLM/context environment variables from config
export REFRAG_RUNTIME GLM_API_KEY GLM_API_BASE GLM_MODEL CTX_REQUIRE_CONTEXT CTX_RELEVANCE_GATE CTX_MIN_RELEVANCE CTX_REWRITE_MAX_TOKENS

# Build ctx command with optional mode flag
CTX_CMD=(python3 scripts/ctx.py)
case "${CTX_DEFAULT_MODE,,}" in
	unicorn)
		CTX_CMD+=("--unicorn")
		;;
	detail)
		CTX_CMD+=("--detail")
		;;
esac
CTX_CMD+=("$USER_MESSAGE" --collection "$CTX_COLLECTION")

# Run ctx with collection (extended timeout for multi-pass unicorn mode)
ENHANCED=$(timeout 60 "${CTX_CMD[@]}" 2>/dev/null || echo "$USER_MESSAGE")

# Replace user message with enhanced version using jq
echo "$INPUT" | jq --arg enhanced "$ENHANCED" '.user_message = $enhanced'