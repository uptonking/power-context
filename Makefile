SHELL := /bin/bash

# Avoid inheriting Docker context from shells/venvs (e.g., DOCKER_HOST=unix:///Users/...)
# An empty export forces docker to use its default context/socket.
export DOCKER_HOST =

.PHONY: help up down logs ps restart rebuild index reindex watch env hybrid bootstrap history rerank-local setup-reranker prune warm health
.PHONY: venv venv-install

.PHONY: qdrant-status qdrant-list qdrant-prune qdrant-index-root

venv: ## create local virtualenv .venv
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip

venv-install: ## install project dependencies into .venv
	[ -d .venv ] || $(MAKE) venv
	. .venv/bin/activate && pip install -r requirements.txt



# Show available targets
help: ## show targets and their descriptions
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' Makefile | sed 's/:.*##/: /' | column -t

# Guard for required env vars: usage `make guard-VAR`
guard-%:
	@if [ -z "${${*}}" ]; then echo "Missing env: $*"; exit 1; fi

up: ## docker compose up (build if needed)
	docker compose up -d --build

down: ## docker compose down
	docker compose down

logs: ## follow logs
	docker compose logs -f --tail=100

ps: ## show container status
	docker compose ps

restart: ## restart stack (rebuild)
	docker compose down && docker compose up -d --build

rebuild: ## rebuild images without cache
	docker compose build --no-cache

index: ## index code into Qdrant without dropping the collection
	docker compose run --rm indexer --root /work

reindex: ## recreate collection then index from scratch (will remove existing points!)
	docker compose run --rm indexer --root /work --recreate

# Index an arbitrary local path without cloning into this repo
index-path: ## index an arbitrary repo: make index-path REPO_PATH=/abs/path [RECREATE=1] [REPO_NAME=name] [COLLECTION=name]
	@if [ -z "$(REPO_PATH)" ]; then \
		echo "Usage: make index-path REPO_PATH=/abs/path [RECREATE=1] [REPO_NAME=name] [COLLECTION=name]"; exit 1; \
	fi
	@NAME=$${REPO_NAME:-$$(basename "$(REPO_PATH)")}; \
	COLL=$${COLLECTION:-$$NAME}; \
	HOST_INDEX_PATH="$(REPO_PATH)" COLLECTION_NAME="$$COLL" REPO_NAME="$$NAME" \
	docker compose run --rm -v "$$PWD":/app:ro --entrypoint python indexer /app/scripts/ingest_code.py --root /work $${RECREATE:+--recreate}

# Index the current working directory quickly
index-here: ## index the current directory: make index-here [RECREATE=1] [REPO_NAME=name] [COLLECTION=name]
	@RP=$$(pwd); \
	NAME=$${REPO_NAME:-$$(basename "$$RP")}; \
	COLL=$${COLLECTION:-$$NAME}; \
	HOST_INDEX_PATH="$$RP" COLLECTION_NAME="$$COLL" REPO_NAME="$$NAME" \
	docker compose run --rm indexer --root /work $${RECREATE:+--recreate}


watch: ## watch mode: reindex changed files on save (Ctrl+C to stop)
	docker compose run --rm --entrypoint python indexer /work/scripts/watch_index.py

rerank: ## multi-query re-ranker helper example
	docker compose run --rm --entrypoint python indexer /work/scripts/rerank_query.py \
	  --query "chunk code by lines with overlap for indexing" \
	  --query "function to split code into overlapping line chunks" \
	  --language python --under /work/scripts --limit 5

warm: ## prime ANN/search caches with a few queries
	docker compose run --rm --entrypoint python indexer /work/scripts/warm_start.py --ef 256 --limit 3

health: ## run health checks for collection/model settings
	docker compose run --rm --entrypoint python indexer /work/scripts/health_check.py


# Check llama.cpp decoder health on localhost:8080 (200 OK expected)
decoder-health: ## ping llama.cpp server
	@URL=$${LLAMACPP_HEALTH_URL:-http://localhost:8080}; \
	CODE=$$(curl -s -o /dev/null -w "%{http_code}" $$URL); \
	echo "llamacpp @ $$URL -> $$CODE"; \
	[ "$$CODE" = "200" ] && echo "OK" || echo "WARN: non-200"

env: ## create .env from example if missing
	[ -f .env ] || cp .env.example .env

hybrid: ## hybrid search: dense + lexical RRF fuse (respects --language/--under/--kind)
	docker compose run --rm --entrypoint python indexer /work/scripts/hybrid_search.py \
	  --query "chunk code by lines" --query "overlapping line chunks" --limit 8

bootstrap: env up ## one-shot: up -> wait -> index -> warm -> health
	./scripts/wait-for-qdrant.sh
	$(MAKE) reindex
	$(MAKE) warm || true
	$(MAKE) health

history: ## ingest Git history (messages + file lists)
	docker compose run --rm --entrypoint python indexer /work/scripts/ingest_history.py --max-commits 200

prune-path: ## prune a repo by path: make prune-path REPO_PATH=/abs/path
	@if [ -z "$(REPO_PATH)" ]; then \
		echo "Usage: make prune-path REPO_PATH=/abs/path"; exit 1; \
	fi
	HOST_INDEX_PATH="$(REPO_PATH)" PRUNE_ROOT=/work \
	docker compose run --rm --entrypoint python indexer /work/scripts/prune.py

rerank-local: ## local cross-encoder reranker (requires RERANKER_ONNX_PATH, RERANKER_TOKENIZER_PATH)
	@if [ -z "$(RERANKER_ONNX_PATH)" ] || [ -z "$(RERANKER_TOKENIZER_PATH)" ]; then \
		echo "RERANKER_ONNX_PATH and RERANKER_TOKENIZER_PATH must be set in .env"; exit 1; \
	fi
	docker compose run --rm --entrypoint python indexer /work/scripts/rerank_local.py --query "search symbols" --topk 50 --limit 12

setup-reranker: ## download ONNX reranker + tokenizer, update .env, then smoke-test
	@if [ -z "$(ONNX_URL)" ] || [ -z "$(TOKENIZER_URL)" ]; then \
		echo "Provide ONNX_URL and TOKENIZER_URL, e.g."; \
		echo "  make setup-reranker ONNX_URL=https://.../model.onnx TOKENIZER_URL=https://.../tokenizer.json"; \
		exit 1; \
	fi
	python3 scripts/setup_reranker.py --onnx-url "$(ONNX_URL)" --tokenizer-url "$(TOKENIZER_URL)" --dest "$(or $(DEST),models)" && \
	$(MAKE) rerank-local

prune: ## remove points for missing files or mismatched file_hash
	docker compose run --rm --entrypoint python indexer /work/scripts/prune.py



# Convenience: full no-cache rebuild and bring-up sequences
up-nc: ## up with full no-cache rebuild
	docker compose build --no-cache && docker compose up -d

restart-nc: ## down, no-cache rebuild, up
	docker compose down && docker compose build --no-cache && docker compose up -d

reset-dev: ## full dev reset: qdrant -> wait -> init payload -> reindex -> bring up services (incl. decoder)
	docker compose down || true
	docker compose build --no-cache indexer mcp mcp_indexer mcp_http mcp_indexer_http watcher llamacpp
	docker compose up -d qdrant
	./scripts/wait-for-qdrant.sh
	docker compose run --rm init_payload || true
	$(MAKE) tokenizer

	docker compose run --rm -e INDEX_MICRO_CHUNKS -e MAX_MICRO_CHUNKS_PER_FILE -e TOKENIZER_PATH -e TOKENIZER_URL indexer --root /work --recreate
	$(MAKE) llama-model
	docker compose up -d mcp mcp_indexer watcher llamacpp
	# Ensure watcher is up even if a prior step or manual bring-up omitted it
	docker compose up -d watcher
	docker compose ps


reset-dev-codex: ## bring up Qdrant + Streamable HTTP MCPs only (for OpenAI Codex RMCP)
	docker compose down || true
	docker compose build --no-cache indexer mcp_http mcp_indexer_http watcher llamacpp
	docker compose up -d qdrant
	./scripts/wait-for-qdrant.sh
	# Seed Qdrant and create a fresh index for Codex
	docker compose run --rm init_payload || true
	$(MAKE) tokenizer
	docker compose run --rm -e INDEX_MICRO_CHUNKS -e MAX_MICRO_CHUNKS_PER_FILE -e TOKENIZER_PATH -e TOKENIZER_URL indexer --root /work --recreate
	$(MAKE) llama-model

	docker compose up -d mcp_http mcp_indexer_http watcher llamacpp
	docker compose ps


quantize-reranker: ## Quantize reranker ONNX to INT8 (set RERANKER_ONNX_PATH, optional OUTPUT_ONNX_PATH)
	@[ -n "$(RERANKER_ONNX_PATH)" ] || { echo "Set RERANKER_ONNX_PATH to your ONNX file"; exit 1; }
	python3 scripts/quantize_reranker.py



reset-dev-dual: ## bring up BOTH legacy SSE and Streamable HTTP MCPs (dual-compat mode)
	docker compose down || true
	docker compose build --no-cache indexer mcp mcp_indexer mcp_http mcp_indexer_http watcher llamacpp
	docker compose up -d qdrant
	./scripts/wait-for-qdrant.sh
	docker compose run --rm init_payload || true
	$(MAKE) tokenizer
	docker compose run --rm -e INDEX_MICRO_CHUNKS -e MAX_MICRO_CHUNKS_PER_FILE -e TOKENIZER_PATH -e TOKENIZER_URL indexer --root /work --recreate
	$(MAKE) llama-model
	docker compose up -d mcp mcp_indexer mcp_http mcp_indexer_http watcher llamacpp
	# Ensure watcher is up even if a prior step or manual bring-up omitted it
	docker compose up -d watcher
	docker compose ps

# --- llama.cpp tiny model provisioning ---
LLAMACPP_MODEL_URL ?= https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF/resolve/main/granite-4.0-micro-Q4_K_M.gguf
LLAMACPP_MODEL_PATH ?= models/model.gguf

llama-model: ## download tiny GGUF model into ./models/model.gguf (override with LLAMACPP_MODEL_URL/LLAMACPP_MODEL_PATH)
	@mkdir -p $(dir $(LLAMACPP_MODEL_PATH))
	@echo "Downloading: $(LLAMACPP_MODEL_URL) -> $(LLAMACPP_MODEL_PATH)" && \
	curl -L --fail --retry 3 -C - "$(LLAMACPP_MODEL_URL)" -o "$(LLAMACPP_MODEL_PATH)"

llamacpp-up: llama-model ## fetch tiny model (if missing) and start llama.cpp sidecar
	docker compose up -d llamacpp && sleep 2 && curl -sI http://localhost:8080 | head -n1 || true

# Optional: build a custom image that bakes the model into the image (no host volume needed)
llamacpp-build-image: ## build custom llama.cpp image with baked model (override LLAMACPP_MODEL_URL)
	docker build -f Dockerfile.llamacpp --build-arg MODEL_URL="$(LLAMACPP_MODEL_URL)" -t context-llamacpp:tiny .

# Download a tokenizer.json for micro-chunking (default: BAAI/bge-base-en-v1.5)
TOKENIZER_URL ?= https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json
TOKENIZER_PATH ?= models/tokenizer.json

tokenizer: ## download tokenizer.json to models/tokenizer.json (override with TOKENIZER_URL/TOKENIZER_PATH)
	@mkdir -p $(dir $(TOKENIZER_PATH))
	@echo "Downloading: $(TOKENIZER_URL) -> $(TOKENIZER_PATH)" && \
	curl -L --fail --retry 3 -C - "$(TOKENIZER_URL)" -o "$(TOKENIZER_PATH)"


# Router helpers
Q ?= what is hybrid search?
route-plan: ## plan-only route for a query: make route-plan Q="your question"
	python3 scripts/mcp_router.py --plan "$(Q)"

route-run: ## execute routed tool(s) over HTTP: make route-run Q="your question"
	python3 scripts/mcp_router.py --run "$(Q)"
router-eval: ## run the mock-based router eval harness
	python3 scripts/router_eval.py


# Live orchestration smoke test (no CI): bring up stack, reindex, run router
router-smoke: ## spin up compose, reindex, store a memory via router, then answer; exits nonzero on failure
	set -e; \
	docker compose down || true; \
	docker compose up -d qdrant; \
	./scripts/wait-for-qdrant.sh; \
	$(MAKE) llama-model; \
	docker compose up -d mcp_http mcp_indexer_http llamacpp; \
	echo "Waiting for MCP HTTP health..."; \
	for i in $$(seq 1 30); do \
	  code1=$$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$${FASTMCP_HTTP_HEALTH_PORT:-18002}/readyz || true); \
	  code2=$$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$${FASTMCP_INDEXER_HTTP_HEALTH_PORT:-18003}/readyz || true); \
	  if [ "$$code1" = "200" ] && [ "$$code2" = "200" ]; then echo "MCP HTTP ready"; break; fi; \
	  sleep 1; \
	  if [ $$i -eq 30 ]; then echo "MCP HTTP health timeout"; exit 1; fi; \
	done; \
	$(MAKE) reindex; \
	echo "Storing a smoke memory via router..."; \
	python3 scripts/mcp_router.py --run "remember this: router smoke memory"; \
	echo "Running a router answer..."; \
	python3 scripts/mcp_router.py --run "recap our architecture decisions for the indexer"; \
	echo "router-smoke: PASS"



# Qdrant via MCP router convenience targets
qdrant-status:
	python3 scripts/mcp_router.py --run "status"

qdrant-list:
	python3 scripts/mcp_router.py --run "list collections"

qdrant-prune:
	python3 scripts/mcp_router.py --run "prune"

qdrant-index-root:
	python3 scripts/mcp_router.py --run "reindex repo"


