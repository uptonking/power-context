#!/usr/bin/env python3
"""
Repository Manager for SWE-bench Evaluation.

Handles cloning, caching, and checking out specific commits
for SWE-bench repository evaluation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "swe-bench" / "repos"

# Repository URLs
REPO_URLS = {
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "django/django": "https://github.com/django/django.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
}


class RepoManager:
    """Manages repository clones for SWE-bench evaluation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._current_checkouts: dict[str, str] = {}  # repo -> current commit
    
    def get_repo_path(self, repo: str) -> Path:
        """Get the local path for a repository."""
        # Convert "django/django" to "django__django"
        safe_name = repo.replace("/", "__")
        return self.cache_dir / safe_name
    
    def ensure_repo(self, repo: str) -> Path:
        """Ensure repository is cloned locally.
        
        Returns the local path to the repository.
        """
        repo_path = self.get_repo_path(repo)
        
        if repo_path.exists() and (repo_path / ".git").exists():
            # Already cloned, fetch latest
            print(f"  Updating {repo}...")
            try:
                subprocess.run(
                    ["git", "fetch", "--all", "--quiet"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                pass  # Non-fatal, might be offline
            return repo_path
        
        # Clone fresh
        url = REPO_URLS.get(repo)
        if not url:
            # Try to construct GitHub URL
            url = f"https://github.com/{repo}.git"
        
        print(f"  Cloning {repo}...")
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            ["git", "clone", "--quiet", url, str(repo_path)],
            check=True,
            capture_output=True,
        )
        
        return repo_path
    
    def checkout_commit(self, repo: str, commit: str) -> Path:
        """Checkout a specific commit in the repository.
        
        Returns the repo path after checkout.
        """
        repo_path = self.ensure_repo(repo)
        
        # Check if already at this commit
        if self._current_checkouts.get(repo) == commit:
            return repo_path
        
        # Clean and checkout
        try:
            subprocess.run(
                ["git", "checkout", "--force", commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "clean", "-fdx", "--quiet"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            self._current_checkouts[repo] = commit
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to checkout {commit} in {repo}: {e.stderr.decode()}"
            )
        
        return repo_path
    
    def get_current_commit(self, repo: str) -> Optional[str]:
        """Get the currently checked out commit for a repo."""
        return self._current_checkouts.get(repo)
    
    def list_cached_repos(self) -> list[str]:
        """List all cached repositories."""
        repos = []
        for path in self.cache_dir.iterdir():
            if path.is_dir() and (path / ".git").exists():
                # Convert "django__django" back to "django/django"
                repos.append(path.name.replace("__", "/"))
        return sorted(repos)
    
    def clear_cache(self, repo: Optional[str] = None):
        """Clear cached repositories.
        
        If repo is specified, only clear that repo.
        Otherwise, clear all cached repos.
        """
        if repo:
            repo_path = self.get_repo_path(repo)
            if repo_path.exists():
                shutil.rmtree(repo_path)
                self._current_checkouts.pop(repo, None)
        else:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._current_checkouts.clear()

