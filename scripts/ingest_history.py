#!/usr/bin/env python3
import os
import argparse
import subprocess
import shlex
import hashlib
from typing import List, Dict, Any
import re
import time

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")
REPO_NAME = os.environ.get("REPO_NAME", "workspace")


from scripts.utils import sanitize_vector_name as _sanitize_vector_name


def run(cmd: str) -> str:
    p = subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{p.stderr}")
    return p.stdout


def try_run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def ensure_base_ref(args) -> str:
    # Prefer local HEAD when present
    proc = try_run("git rev-parse --verify HEAD")
    if proc.returncode == 0 and proc.stdout.strip():
        return "HEAD"
    # Fallback: fetch shallow history from remote and use its HEAD
    remote = args.remote or "origin"
    depth = args.fetch_depth or 1000
    try_run(f"git fetch --all --tags --prune --depth {depth}")
    ref = try_run(f"git symbolic-ref -q {remote}/HEAD")
    if ref.returncode == 0 and ref.stdout.strip():
        return ref.stdout.strip()
    # Last resort: try common branch names
    for b in (f"{remote}/main", f"{remote}/master"):
        if try_run(f"git rev-parse --verify {b}").returncode == 0:
            return b
    raise RuntimeError("Could not determine a base revision (HEAD or remote HEAD)")


def list_commits(args) -> List[str]:
    cmd = ["git", "rev-list", "--no-merges"]
    if args.since:
        cmd += [f"--since={args.since}"]
    if args.until:
        cmd += [f"--until={args.until}"]
    if args.author:
        cmd += [f"--author={args.author}"]
    if args.grep:
        cmd += [f"--grep={args.grep}"]
    base = ensure_base_ref(args)
    if args.path:
        cmd += [base, "--", args.path]
    else:
        cmd += [base]
    out = run(" ".join(shlex.quote(c) for c in cmd))
    commits = [l.strip() for l in out.splitlines() if l.strip()]
    if args.max_commits and len(commits) > args.max_commits:
        commits = commits[: args.max_commits]
    return commits


def _redact_emails(text: str) -> str:
    return re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<redacted>", text or ""
    )


def commit_metadata(commit: str) -> Dict[str, Any]:
    fmt = "%H%x1f%an%x1f%ae%x1f%ad%x1f%s%x1f%b"
    out = run(f"git show -s --format={fmt} {commit}")
    parts = out.strip().split("\x1f")
    sha, an, ae, ad, subj, body = (parts + [""] * 6)[:6]
    files_out = run(f"git diff-tree --no-commit-id --name-only -r {commit}")
    files = [f for f in files_out.splitlines() if f]
    message = _redact_emails((subj + ("\n" + body if body else "")).strip())
    if len(message) > 2000:
        message = message[:2000] + "…"
    return {
        "commit_id": sha,
        "author_name": an,
        # email stripped for privacy
        "authored_date": ad,
        "message": message,
        "files": files,
    }


def stable_id(commit_id: str) -> int:
    h = hashlib.sha1(commit_id.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def build_text(
    md: Dict[str, Any], max_files: int = 200, include_body: bool = True
) -> str:
    msg = md.get("message", "")
    files = md.get("files", [])
    if not include_body:
        msg = msg.splitlines()[0] if msg else ""
    head = msg.strip()
    files_part = "\n".join(files[:max_files])
    return (head + "\n\nFiles:\n" + files_part).strip()


def main():
    ap = argparse.ArgumentParser(
        description="Ingest Git history into Qdrant deterministically"
    )
    ap.add_argument(
        "--since", type=str, default=None, help="e.g., '2 years ago' or '2023-01-01'"
    )
    ap.add_argument("--until", type=str, default=None)
    ap.add_argument("--author", type=str, default=None)
    ap.add_argument(
        "--grep", type=str, default=None, help="Filter commit messages by regex"
    )
    ap.add_argument(
        "--path", type=str, default=None, help="Limit history to a path subtree"
    )
    ap.add_argument("--max-commits", type=int, default=500)
    ap.add_argument("--per-batch", type=int, default=128)
    ap.add_argument(
        "--include-body", action="store_true", help="Include full body in text to embed"
    )
    ap.add_argument(
        "--remote",
        type=str,
        default="origin",
        help="Remote to fetch from if no local HEAD is present",
    )
    ap.add_argument(
        "--fetch-depth",
        type=int,
        default=1000,
        help="Shallow fetch depth for history, when needed",
    )
    args = ap.parse_args()

    model = TextEmbedding(model_name=MODEL_NAME)
    vec_name = _sanitize_vector_name(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    commits = list_commits(args)
    if not commits:
        print("No commits matched filters.")
        return

    points: List[models.PointStruct] = []
    for sha in commits:
        md = commit_metadata(sha)
        text = build_text(md, include_body=args.include_body)
        vec = next(model.embed([text])).tolist()
        payload = {
            "document": (
                md.get("message", "").splitlines()[0]
                if md.get("message")
                else md["commit_id"]
            ),
            "information": text[:512],
            "metadata": {
                "language": "git",
                "kind": "git_message",
                "symbol": md["commit_id"],
                "symbol_path": md["commit_id"],
                "repo": REPO_NAME,
                "commit_id": md["commit_id"],
                "author_name": md["author_name"],
                "authored_date": md["authored_date"],
                "message": md["message"],
                "files": md["files"],
                "path": ".git",
                "path_prefix": ".git",
                "ingested_at": int(time.time()),
            },
        }
        pid = stable_id(md["commit_id"])  # deterministic per-commit
        point = models.PointStruct(id=pid, vector={vec_name: vec}, payload=payload)
        points.append(point)
        if len(points) >= args.per_batch:
            client.upsert(collection_name=COLLECTION, points=points)
            points.clear()
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(commits)} commits into {COLLECTION}.")


if __name__ == "__main__":
    main()
