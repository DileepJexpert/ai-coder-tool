#!/usr/bin/env python3
"""
codebase_rag.py — local semantic code search over a repo, using Ollama embeddings.

Two commands:
    python codebase_rag.py index <dir>
    python codebase_rag.py search "<query>" [-k 5]

The index is a single pickle (.watchdog_index.pkl) holding an L2-normalized
matrix of embedding vectors plus the chunk metadata. Search is a cosine
similarity matmul. No vector DB, no framework.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import requests


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
INDEX_FILE = ".watchdog_index.pkl"

# Chunking: ~50-line windows with ~10-line overlap so that symbols near a
# window boundary still appear in at least one chunk.
CHUNK_LINES = 50
CHUNK_OVERLAP = 10

# Hard size cap. Files larger than this almost always hurt retrieval quality
# (vendored bundles, generated code) and inflate the index.
MAX_FILE_BYTES = 200 * 1024  # 200KB

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", "target", ".idea", ".dart_tool",
    ".mypy_cache", ".pytest_cache", ".tox", ".next", ".nuxt",
}

CODE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".m", ".mm",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".proto",
    ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    ".md", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".tf", ".hcl",
    ".dart", ".lua", ".pl", ".r", ".jl", ".ex", ".exs",
}

# Files without an extension we still want to index.
NAMED_FILES = {"dockerfile", "makefile", "rakefile", "vagrantfile"}


def embed(text: str) -> np.ndarray:
    """Single embedding via Ollama's /api/embeddings."""
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def should_index(p: Path) -> bool:
    if p.suffix.lower() in CODE_EXTS:
        return True
    if p.name.lower() in NAMED_FILES:
        return True
    return False


def walk_files(root: Path):
    """Yield file Paths to index, skipping junk dirs, large files, and binary blobs."""
    for dirpath, dirnames, filenames in os.walk(root):
        # In-place mutate dirnames so os.walk skips them.
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            if not should_index(p):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p


def read_utf8(p: Path):
    """Strict UTF-8 read; anything that isn't clean text is skipped."""
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def chunk_file(rel_path: str, text: str):
    lines = text.splitlines()
    if not lines:
        return
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    i = 0
    n = len(lines)
    while i < n:
        end = min(i + CHUNK_LINES, n)
        yield {
            "path": rel_path,
            "start": i + 1,
            "end": end,
            "text": "\n".join(lines[i:end]),
        }
        if end == n:
            break
        i += step


def cmd_index(root_dir: str):
    root = Path(root_dir).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(2)
    print(f"Indexing {root}...")

    chunks = []
    n_files = 0
    for fp in walk_files(root):
        text = read_utf8(fp)
        if text is None:
            continue
        rel = str(fp.relative_to(root)).replace("\\", "/")
        for c in chunk_file(rel, text):
            chunks.append(c)
        n_files += 1

    if not chunks:
        print("No code chunks found to index.", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding {len(chunks)} chunks from {n_files} files via {EMBED_MODEL}...")
    vectors = []
    for i, c in enumerate(chunks):
        # Prefix with the relative path so identifiers like "auth/login.py"
        # contribute lexically to the embedding.
        prefixed = f"{c['path']}\n{c['text']}"
        try:
            vectors.append(embed(prefixed))
        except Exception as e:
            print(f"\nEmbed failed at chunk {i} ({c['path']}): {e}", file=sys.stderr)
            sys.exit(1)
        if (i + 1) % 20 == 0 or i + 1 == len(chunks):
            print(f"  {i+1}/{len(chunks)}", end="\r", flush=True)
    print()

    matrix = normalize_rows(np.vstack(vectors).astype(np.float32))
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"vectors": matrix, "meta": chunks, "root": str(root)}, f)
    print(f"Wrote {INDEX_FILE} ({matrix.shape[0]} chunks, dim={matrix.shape[1]})")


def _load_index():
    if not os.path.exists(INDEX_FILE):
        return None
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def search_code(query: str, k: int = 5) -> str:
    """Public entry point used by watchdog.py.

    Returns either formatted top-k results or a clear error message — never
    raises in normal operation, so the agent can read the result like any
    other tool output.
    """
    idx = _load_index()
    if idx is None:
        return (
            f"ERROR: no semantic index found ({INDEX_FILE} missing in cwd). "
            f"Build one first: python codebase_rag.py index ."
        )
    try:
        q = embed(query)
    except Exception as e:
        return f"ERROR: embedding the query failed: {e}"
    n = np.linalg.norm(q) or 1.0
    q = q / n
    sims = idx["vectors"] @ q
    if k < 1:
        k = 1
    k = min(k, len(idx["meta"]))
    top = np.argsort(-sims)[:k]

    out = []
    for i in top:
        i = int(i)
        m = idx["meta"][i]
        out.append(
            f"--- {m['path']}:{m['start']}-{m['end']}  (score={float(sims[i]):.3f}) ---\n"
            f"{m['text']}"
        )
    return "\n\n".join(out) if out else "No results."


def cmd_search(query: str, k: int):
    print(search_code(query, k=k))


def main():
    ap = argparse.ArgumentParser(prog="codebase_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Build the semantic index")
    p_idx.add_argument("dir", nargs="?", default=".")

    p_s = sub.add_parser("search", help="Search the index")
    p_s.add_argument("query")
    p_s.add_argument("-k", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "index":
        cmd_index(args.dir)
    else:
        cmd_search(args.query, args.k)


if __name__ == "__main__":
    main()
