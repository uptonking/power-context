SHELL := /bin/bash

.PHONY: help up down logs ps restart rebuild index reindex watch env hybrid bootstrap history rerank-local setup-reranker prune warm health

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

