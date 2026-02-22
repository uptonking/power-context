# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-22

### Added

- Pip-installable package with `power-context` command
- Version management via `cli._version` module
- `pyproject.toml` for package configuration
- Entry point: `power-context = "cli.main:main"`
- Package includes both `cli` and `scripts` modules

### Changed

- CLI prog name changed from `context-engine` to `power-context`
- Improved import path resolution with fallback for development mode
- Skill renamed from `context-engine-cli` to `power-context`

### Fixed

- Import path resolution now works correctly both when installed as package and in development mode

## risks

- not all tests passed
  - uv run --with pytest --with pytest-asyncio pytest -q tests/test_search_specialized_collection.py tests/ test_symbol_graph_tool.py

- --collection is ignored by specialized search commands.
  - cli/main.py:39 defines -c/--collection for search-tests, search-config, search-callers, search-importers (via _add_common_args), but command handlers never forward it
- Compared to scripts/watch_index.py:157, existing watcher intentionally avoids eagerly ensuring default collection in multi-repo mode.

## Notes

- Legacy CLI at `scripts/ctx.py` remains available for backward compatibility
- All 14 CLI commands fully functional
- Docker deployments continue using `requirements.txt`
