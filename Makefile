SHELL := /bin/bash

.PHONY: up down logs ps restart rebuild index reindex env hybrid bootstrap history rerank-local

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

restart:
	docker compose down && docker compose up -d --build

rebuild:
	docker compose build --no-cache

# Index code into Qdrant without dropping the collection
index:
	docker compose run --rm indexer --root /work

# Recreate collection then index from scratch (will remove existing points!)
reindex:
	docker compose run --rm indexer --root /work --recreate

# Watch mode: reindex changed files on save (Ctrl+C to stop)
watch:
	docker compose run --rm --entrypoint python indexer /work/scripts/watch_index.py

# Multi-query re-ranker helper example
rerank:
	docker compose run --rm --entrypoint python indexer /work/scripts/rerank_query.py \
	  --query "chunk code by lines with overlap for indexing" \
	  --query "function to split code into overlapping line chunks" \
	  --language python --under /work/scripts --limit 5

warm:
	docker compose run --rm --entrypoint python indexer /work/scripts/warm_start.py --ef 256 --limit 3

health:
	docker compose run --rm --entrypoint python indexer /work/scripts/health_check.py

# Create .env from example if missing
env:
	[ -f .env ] || cp .env.example .env

# Hybrid search: dense + lexical RRF fuse
hybrid:
	docker compose run --rm --entrypoint python indexer /work/scripts/hybrid_search.py \
	  --query "chunk code by lines" --query "overlapping line chunks" --limit 8

# One-shot bootstrap to a production-ready local MCP + index + checks
bootstrap: env up
	$(MAKE) reindex
	$(MAKE) warm || true
	$(MAKE) health

# Ingest Git history (messages + file lists) into the collection
history:
	docker compose run --rm --entrypoint python indexer /work/scripts/ingest_history.py --max-commits 200

# Local cross-encoder reranker (requires RERANKER_ONNX_PATH and RERANKER_TOKENIZER_PATH)
rerank-local:
	docker compose run --rm --entrypoint python indexer /work/scripts/rerank_local.py --query "search symbols" --topk 50 --limit 12


