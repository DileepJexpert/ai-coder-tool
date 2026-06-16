#!/usr/bin/env python3
"""
watchdog v3 — a local CLI coding agent on Ollama.

A small, self-contained alternative to Claude Code that runs entirely
against a local Ollama instance. Two deps only: requests + numpy.
"""
import argparse
import base64
import glob as _glob
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

# Semantic code search is optional: the agent still runs without it.
try:
    from codebase_rag import search_code as _rag_search
except Exception:
    _rag_search = None


# --- Configuration -----------------------------------------------------------

OLLAMA_HOST          = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
A1111_HOST           = os.environ.get("A1111_HOST",  "http://localhost:7860")
DEFAULT_MODEL        = "qwen3:8b"            # fits 8GB fully; fast tool-calling
DEFAULT_VISION_MODEL = "moondream"           # light VLM for 8GB
EMBED_MODEL          = "nomic-embed-text"    # ~270MB, used by codebase_rag
NUM_CTX              = 8192
MAX_TOOL_ROUNDS      = 25                    # safety cap on the agent loop
MAX_SUBTASK_DEPTH    = 1                     # subtasks cannot spawn subtasks

# Output / IO caps. These exist because a single oversized tool result can
# blow past NUM_CTX and ruin the rest of the turn.
MAX_READ_BYTES   = 60 * 1024       # read_file
MAX_CMD_OUTPUT   = 12 * 1024       # run_command, run_tests, git
MAX_GREP_HITS    = 200
CMD_TIMEOUT      = 120

CODE_EXTS_FOR_GREP = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".scala",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".sh", ".bash", ".ps1", ".bat",
    ".sql", ".html", ".css", ".scss", ".vue", ".svelte",
    ".md", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".dart", ".lua", ".pl", ".r", ".jl", ".ex", ".exs",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", "target", ".idea", ".dart_tool",
    ".mypy_cache", ".pytest_cache", ".tox", ".next", ".nuxt",
}


# --- Terminal colors ---------------------------------------------------------

_CSI = "\033["
_USE_COLOR = sys.stdout.isatty()


def _c(s, code):
    return f"{_CSI}{code}m{s}{_CSI}0m" if _USE_COLOR else str(s)


def DIM(s):    return _c(s, "2")
def RED(s):    return _c(s, "31")
def GREEN(s):  return _c(s, "32")
def YELLOW(s): return _c(s, "33")
def BLUE(s):   return _c(s, "34")
def CYAN(s):   return _c(s, "36")
def BOLD(s):   return _c(s, "1")


# --- System prompts ----------------------------------------------------------

SYSTEM_PROMPT = """You are watchdog, a local AI assistant running entirely on the user's own machine.
You help with anything — writing and editing code, writing documents, tutorials, blog posts
and learning material for any audience, creating illustrations to go with them, exploring
data, debugging, and other day-to-day tasks.

For CODE:
- Use search_code (semantic) or grep/glob FIRST to find where something lives. Do not guess filenames.
- Read a file before editing it. Make the smallest change that works. Use edit_file/multi_edit
  with exact, unique snippets.
- After changing code, run_tests to verify when a test command is available.
- Use git (read-only) to inspect diffs and history; never assume what changed.

For WRITING (tutorials, docs, blog posts, study material, stories, etc.):
- Match the language and depth to the audience. Ask if it isn't clear who it's for.
  Kids (class 1-5) need short sentences and concrete examples; adults can handle nuance.
- Useful structure: short intro, worked examples, summary or practice questions.
- When an illustration would help, call generate_image with a vivid prompt and the
  STYLE HINTS appropriate to the audience and topic — e.g. "photorealistic, 4k" for
  product shots, "digital painting, dramatic lighting" for art, "children's book
  illustration, cartoon, flat colors" for kid material, "technical diagram, clean
  vector style" for explainers. Don't default to a kid style unless that's the goal.
- Save the final material as a markdown file with write_file. Reference any generated
  images inline using ![alt](images/<file>) so they render in the chat.

General rules:
- Work step by step. Use ONE tool at a time and read the result before the next step.
- For a large, self-contained sub-job, call run_subtask so the main context stays clean.
- To inspect a local image/diagram, call analyze_image with a specific question.
- Keep explanations short. When done, give a one or two line summary and stop calling tools.
- You operate on the user's real filesystem. Be careful with destructive commands.
"""

SUBTASK_SYSTEM_PROMPT = """You are watchdog running as a SUBTASK. Complete ONE focused subtask using your tools, then stop and report a SHORT plain-text result summary to the parent agent.

Same rules as the main agent: use search_code/grep/glob to locate code, read before editing, make the smallest change that works, run tests if you changed code. Do not chat. Do not ask follow-ups. When the subtask is done, output a one or two line summary and stop calling tools.
"""


# --- Permission gate ---------------------------------------------------------

class Permissions:
    """Wraps the y/N confirmation prompt for [confirm] tools.

    --yolo flips ask() to always-allow, including inside subtasks (the same
    Permissions instance is threaded through).
    """

    def __init__(self, yolo: bool):
        self.yolo = yolo

    def ask(self, tool_name: str, args: dict) -> bool:
        if self.yolo:
            return True
        try:
            preview = json.dumps(args, ensure_ascii=False)
        except Exception:
            preview = str(args)
        if len(preview) > 400:
            preview = preview[:400] + "...(truncated)"
        print(YELLOW(f"[confirm] {tool_name}({preview})"))
        try:
            ans = input(YELLOW("  Allow? [y/N] ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return ans in ("y", "yes")


# --- Event sink --------------------------------------------------------------

class EventSink:
    """Receives agent-loop events. The default implementation prints to stdout
    in the same look as the original CLI. Override the on_* methods to plug a
    different frontend in (e.g. the Rich TUI in watchdog_tui.py)."""

    def on_assistant_text(self, text: str, label: str = "agent") -> None:
        if label == "agent":
            print(BOLD(GREEN("\nwatchdog:")) + " " + text + "\n")

    def on_tool_call(self, name: str, args: dict) -> None:
        print(CYAN(f"-> {name}({_short_args(args)})"))

    def on_tool_result(self, name: str, result: str) -> None:
        print(DIM(f"   {_short_preview(result)}"))

    def on_subtask_start(self, goal: str) -> None:
        print(DIM(f"   [subtask start] goal={goal[:120]!r}"))

    def on_subtask_end(self) -> None:
        print(DIM("   [subtask end]"))

    def on_error(self, message: str) -> None:
        print(RED(message))

    def on_round_cap(self, label: str, max_rounds: int) -> None:
        print(YELLOW(f"[{label}] hit round limit ({max_rounds}) — stopping."))


# --- File operation tools ----------------------------------------------------

def tool_read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    try:
        data = p.read_bytes()
    except OSError as e:
        return f"ERROR: {e}"
    truncated = len(data) > MAX_READ_BYTES
    if truncated:
        data = data[:MAX_READ_BYTES]
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n\n[...truncated at {MAX_READ_BYTES} bytes...]"
    return text


def tool_write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {path}"


def tool_edit_file(path: str, old: str, new: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path} is not valid UTF-8 text."
    if not old:
        return "ERROR: `old` is empty. Provide an exact snippet copied from the file."
    count = text.count(old)
    if count == 0:
        return (
            "ERROR: `old` snippet was not found in the file. "
            "Copy an exact snippet (including whitespace) from the file."
        )
    if count > 1:
        return (
            f"ERROR: `old` snippet matches {count} places. "
            "Provide a larger snippet so it is uniquely present in the file."
        )
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"OK: replaced 1 occurrence in {path}"


def tool_multi_edit(path: str, edits) -> str:
    if not isinstance(edits, list) or not edits:
        return "ERROR: `edits` must be a non-empty list of {old, new} objects."
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path} is not valid UTF-8 text."

    # Apply on a working copy so a failure leaves the file untouched.
    work = text
    for i, e in enumerate(edits, 1):
        if not isinstance(e, dict):
            return f"ERROR: edit #{i}: must be an object with `old` and `new`."
        old = e.get("old", "")
        new = e.get("new", "")
        if not old:
            return f"ERROR: edit #{i}: `old` is empty."
        c = work.count(old)
        if c == 0:
            return f"ERROR: edit #{i}: `old` snippet not found (after prior edits)."
        if c > 1:
            return f"ERROR: edit #{i}: `old` snippet matches {c} places; make it unique."
        work = work.replace(old, new, 1)

    p.write_text(work, encoding="utf-8")
    return f"OK: applied {len(edits)} edits to {path}"


def tool_list_dir(path: str = ".") -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: not found: {path}"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    entries = []
    for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = item.name
        if name.startswith(".") or name in SKIP_DIRS:
            continue
        entries.append(name + ("/" if item.is_dir() else ""))
    return "\n".join(entries) if entries else "(empty)"


# --- Search tools ------------------------------------------------------------

def tool_search_code(query: str) -> str:
    if _rag_search is None:
        return (
            "ERROR: codebase_rag is not available. "
            "Make sure codebase_rag.py sits next to watchdog.py and an index exists."
        )
    if not query:
        return "ERROR: empty query."
    return _rag_search(query, k=5)


def _walk_grepable(root: Path):
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            fp = Path(dirpath) / fn
            if fp.suffix.lower() in CODE_EXTS_FOR_GREP:
                yield fp


def tool_grep(pattern: str, path: str = ".", regex: bool = False) -> str:
    if not pattern:
        return "ERROR: empty pattern."
    base = Path(path)
    if not base.exists():
        return f"ERROR: not found: {path}"
    pat = None
    if regex:
        try:
            pat = re.compile(pattern)
        except re.error as e:
            return f"ERROR: bad regex: {e}"

    hits = []
    for fp in _walk_grepable(base):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    matched = pat.search(line) if pat else (pattern in line)
                    if matched:
                        snippet = line.rstrip("\n")
                        if len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        hits.append(f"{fp}:{i}: {snippet}")
                        if len(hits) >= MAX_GREP_HITS:
                            break
        except OSError:
            continue
        if len(hits) >= MAX_GREP_HITS:
            break

    if not hits:
        return "No matches."
    suffix = f"\n...(capped at {MAX_GREP_HITS} hits)" if len(hits) >= MAX_GREP_HITS else ""
    return "\n".join(hits) + suffix


def tool_glob(pattern: str) -> str:
    if not pattern:
        return "ERROR: empty pattern."
    paths = sorted(_glob.glob(pattern, recursive=True))
    if not paths:
        return "No matches."
    if len(paths) > 200:
        return "\n".join(paths[:200]) + f"\n...({len(paths) - 200} more)"
    return "\n".join(paths)


# --- Execution tools ---------------------------------------------------------

def tool_run_command(command: str) -> str:
    if not command or not command.strip():
        return "ERROR: empty command."
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {CMD_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > MAX_CMD_OUTPUT:
        out = out[:MAX_CMD_OUTPUT] + f"\n...[truncated, total {len(out)} chars]"
    return f"exit={proc.returncode}\n{out}".rstrip() + "\n"


def _guess_test_command() -> str:
    """Pick a sensible test command based on the files in the cwd."""
    cwd = Path(".")
    if (cwd / "pytest.ini").exists() or (cwd / "tests").exists() \
            or any(cwd.glob("test_*.py")) or any(cwd.glob("*_test.py")):
        return "pytest -q"
    if (cwd / "pyproject.toml").exists():
        return "pytest -q"
    if (cwd / "pom.xml").exists():
        return "mvn -q test"
    if (cwd / "build.gradle.kts").exists() or (cwd / "build.gradle").exists():
        # Use the wrapper if it exists; fall back to a plain gradle invocation.
        if (cwd / "gradlew").exists() or (cwd / "gradlew.bat").exists():
            return "./gradlew test" if os.name != "nt" else "gradlew.bat test"
        return "gradle test"
    if (cwd / "package.json").exists():
        return "npm test"
    if (cwd / "Cargo.toml").exists():
        return "cargo test"
    if (cwd / "go.mod").exists():
        return "go test ./..."
    return ""


def tool_run_tests(command=None) -> str:
    cmd = command or _guess_test_command()
    if not cmd:
        return (
            "ERROR: no test command provided and could not guess one from the "
            "files in this directory. Call again with an explicit `command`."
        )
    return tool_run_command(cmd)


# --- Version-control tool (read-only) ---------------------------------------

GIT_ALLOWED = {"diff", "log", "status", "blame", "show"}


def tool_git(subcommand: str) -> str:
    if not subcommand or not subcommand.strip():
        return "ERROR: empty subcommand."
    # Tokenize naively — read-only commands don't need shell metacharacters,
    # and avoiding the shell removes a whole class of mistake.
    parts = subcommand.strip().split()
    sub = parts[0]
    if sub not in GIT_ALLOWED:
        return (
            f"ERROR: only read-only git subcommands are allowed. "
            f"Allowed: {sorted(GIT_ALLOWED)}"
        )
    try:
        proc = subprocess.run(
            ["git"] + parts,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return "ERROR: `git` is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return "ERROR: git timed out."
    except Exception as e:
        return f"ERROR: {e}"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > MAX_CMD_OUTPUT:
        out = out[:MAX_CMD_OUTPUT] + "\n...[truncated]"
    return f"exit={proc.returncode}\n{out}".rstrip() + "\n"


# --- Vision tool -------------------------------------------------------------

def tool_analyze_image(path: str, question: str, vision_model: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    try:
        b = p.read_bytes()
    except OSError as e:
        return f"ERROR: {e}"
    b64 = base64.b64encode(b).decode("ascii")
    body = {
        "model": vision_model,
        "messages": [{"role": "user", "content": question or "Describe this image.", "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": NUM_CTX},
    }
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=body, timeout=300)
    except requests.RequestException as e:
        return f"ERROR: Ollama request failed: {e}"
    if r.status_code != 200:
        text = (r.text or "").lower()
        if "not found" in text or "no such" in text or r.status_code == 404:
            return (
                f"ERROR: vision model '{vision_model}' isn't pulled. "
                f"Run: ollama pull {vision_model}"
            )
        return f"ERROR: vision call failed ({r.status_code}): {(r.text or '')[:300]}"
    try:
        return (r.json().get("message", {}) or {}).get("content", "") or "(empty response)"
    except Exception as e:
        return f"ERROR: parsing vision response: {e}"


# --- Image generation tool ---------------------------------------------------

IMAGES_DIR = Path("images")


def _safe_image_slug(name: str) -> str:
    """Filename-safe ASCII slug. No path separators, no dotfiles, max 60 chars."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name or "").strip("_").lower()
    return s[:60] or "image"


def _a1111_txt2img(prompt: str, width: int, height: int, steps: int):
    """Call AUTOMATIC1111 / Forge text-to-image. Returns PNG bytes or None on any failure."""
    try:
        r = requests.post(
            f"{A1111_HOST}/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "sampler_name": "Euler a",
            },
            timeout=300,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        imgs = r.json().get("images") or []
        return base64.b64decode(imgs[0]) if imgs else None
    except Exception:
        return None


def _stub_svg(prompt: str, width: int, height: int) -> bytes:
    """Placeholder image: gradient + the prompt text. Used when SD isn't reachable."""
    digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    c1, c2 = "#" + digest[:6], "#" + digest[6:12]
    body = html.escape(prompt)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{c1}"/><stop offset="100%" stop-color="{c2}"/>'
        f'</linearGradient></defs>'
        f'<rect width="{width}" height="{height}" fill="url(#g)"/>'
        f'<text x="{width // 2}" y="36" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20" font-weight="700" fill="white" opacity="0.85">[ stub image ]</text>'
        f'<foreignObject x="20" y="60" width="{width - 40}" height="{height - 80}">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" '
        f'style="color:white;font-family:sans-serif;font-size:18px;line-height:1.4;'
        f'text-shadow:0 1px 4px rgba(0,0,0,0.4);word-wrap:break-word;">{body}</div>'
        f'</foreignObject></svg>'
    )
    return svg.encode("utf-8")


def tool_generate_image(prompt: str, filename: str = "",
                        width: int = 512, height: int = 512, steps: int = 20) -> str:
    if not prompt or not prompt.strip():
        return "ERROR: empty prompt."
    # Snap to multiples of 8 — most SD samplers require it; safe no-op otherwise.
    width  = (max(64, min(int(width  or 512), 2048)) // 8) * 8
    height = (max(64, min(int(height or 512), 2048)) // 8) * 8
    steps  = max(1, min(int(steps or 20), 150))
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = _safe_image_slug(filename or prompt)

    png = _a1111_txt2img(prompt, width, height, steps)
    if png is not None:
        out = IMAGES_DIR / f"{ts}-{slug}.png"
        out.write_bytes(png)
        return f"OK: image saved to {out.as_posix()} (generated via {A1111_HOST})"

    out = IMAGES_DIR / f"{ts}-{slug}.svg"
    out.write_bytes(_stub_svg(prompt, width, height))
    return (
        f"OK: stub image saved to {out.as_posix()}\n"
        f"NOTE: AUTOMATIC1111 was not reachable at {A1111_HOST} — wrote a placeholder SVG. "
        f"Start AUTOMATIC1111 / Forge with `--api` to get real images."
    )


# --- Subtask tool ------------------------------------------------------------

def tool_run_subtask(goal: str, model: str, vision_model: str,
                     perms: "Permissions", depth: int, sink: "EventSink") -> str:
    if not goal or not goal.strip():
        return "ERROR: empty goal."
    if depth >= MAX_SUBTASK_DEPTH:
        return (
            "ERROR: subtask depth limit reached "
            f"(MAX_SUBTASK_DEPTH={MAX_SUBTASK_DEPTH}); subtasks cannot spawn subtasks."
        )
    # The subtask sees every tool EXCEPT run_subtask itself.
    sub_specs = [t for t in TOOL_SPECS if t["function"]["name"] != "run_subtask"]
    messages = [
        {"role": "system", "content": SUBTASK_SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    sink.on_subtask_start(goal)
    summary = agent_loop(
        messages=messages,
        model=model,
        vision_model=vision_model,
        perms=perms,
        tool_specs=sub_specs,
        depth=depth + 1,
        max_rounds=10,
        label="subtask",
        sink=sink,
    )
    sink.on_subtask_end()
    return summary or "(subtask produced no text)"


# --- Tool registry (the JSON schemas sent to the model) ---------------------

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file and return its contents (capped at ~60KB). "
                "ALWAYS call this on a file before editing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file. Parent directories are created. "
                "Use only for NEW files or whole-file rewrites; for small changes use edit_file. "
                "Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace EXACTLY ONE occurrence of `old` with `new` in a file. "
                "`old` must be an exact snippet copied from the file and must appear there "
                "exactly once. If it is missing or ambiguous the call errors and the file is untouched. "
                "Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact unique snippet copied from the file."},
                    "new": {"type": "string", "description": "Replacement snippet."},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": (
                "Apply several edits to ONE file atomically. `edits` is a list of {old,new} pairs "
                "applied in order; each `old` must be uniquely present at the moment it is applied. "
                "If any edit fails, NO edit is written. Use this when one file needs several small "
                "non-overlapping changes. Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                            },
                            "required": ["old", "new"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List files and directories in a directory. Skips dotfiles and junk dirs "
                "(.git, node_modules, __pycache__, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path; defaults to '.'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Semantic code search over the indexed repo. Use this FIRST to find where something "
                "lives by meaning (e.g. 'where is auth handled', 'function that parses CSV'). "
                "Returns the top matching code chunks with file:line headers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of what to find."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Literal or regex text search across code files. Returns 'file:line: matched line' hits. "
                "Use this for exact strings, identifiers, or symbols when you know what to look for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory or file; defaults to '.'."},
                    "regex": {"type": "boolean", "description": "If true, treat pattern as a regex."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "Find files by filename pattern (e.g. '**/*Controller.java', 'src/**/*.ts'). "
                "Returns sorted matching paths. Use for filename lookups, not content search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command via the system shell (PowerShell/cmd on Windows, sh on Linux/macOS). "
                "Captures stdout+stderr and the exit code; timeout ~120s. Use for one-off builds, "
                "scripts, or inspection. Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the project's test suite. If `command` is omitted, the agent guesses based on the "
                "files present (pytest, mvn, gradle, npm, cargo, go test). Call this AFTER editing code "
                "to verify. Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Optional explicit test command."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git",
            "description": (
                "Run a READ-ONLY git subcommand. Allowed: diff, log, status, blame, show. "
                "Anything else errors. No permission prompt because nothing is mutated. "
                "Use to inspect the working tree, history, and diffs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subcommand": {
                        "type": "string",
                        "description": "e.g. 'status', 'log -n 5', 'diff HEAD~1', 'show abc1234'",
                    },
                },
                "required": ["subcommand"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": (
                "Ask a vision model a SPECIFIC question about a local image "
                "(screenshot, diagram, photo). Returns the model's text answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["path", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image from a text prompt and save it under images/. "
                "Use whenever a picture would help — diagrams, characters, scenes, charts, "
                "covers, art for posts and tutorials, anything visual. Reference the saved "
                "path in markdown as ![alt](images/<file>) so it renders inline in the chat. "
                "Backend is AUTOMATIC1111 / Forge at $A1111_HOST (fully local); if it isn't "
                "running, a labeled placeholder SVG is written so the rest of the workflow still runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Vivid description of the image. Use concrete language and include "
                            "STYLE HINTS that match the goal — e.g. 'photorealistic, 4k, soft light' "
                            "for realism, 'digital painting, dramatic lighting' for art, "
                            "'children's book illustration, cartoon, flat colors' for kid material, "
                            "'clean technical diagram, vector, labeled' for explainers."
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": "Short slug for the filename (no extension). Optional.",
                    },
                    "width":  {"type": "integer", "description": "Width in pixels, 64-2048. Default 512."},
                    "height": {"type": "integer", "description": "Height in pixels, 64-2048. Default 512."},
                    "steps":  {"type": "integer", "description": "Sampler steps, 1-150. Default 20."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_subtask",
            "description": (
                "Run an ISOLATED inner agent loop on a focused sub-job, with a fresh context. "
                "Use for a large self-contained piece of work (e.g. 'find every caller of X and "
                "convert them to Y'). Returns only a short summary to you. Cannot be called from "
                "inside a subtask."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Concrete description of the sub-job."},
                },
                "required": ["goal"],
            },
        },
    },
]

CONFIRM_TOOLS = {"write_file", "edit_file", "multi_edit", "run_command", "run_tests"}


# --- Tool dispatch -----------------------------------------------------------

def _parse_args(raw):
    """Tool arguments arrive as a dict OR as a stringified JSON object."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _call_tool(name, args, model, vision_model, perms, depth, sink):
    if name == "read_file":    return tool_read_file(args.get("path", ""))
    if name == "write_file":   return tool_write_file(args.get("path", ""), args.get("content", ""))
    if name == "edit_file":    return tool_edit_file(args.get("path", ""), args.get("old", ""), args.get("new", ""))
    if name == "multi_edit":   return tool_multi_edit(args.get("path", ""), args.get("edits", []))
    if name == "list_dir":     return tool_list_dir(args.get("path", "."))
    if name == "search_code":  return tool_search_code(args.get("query", ""))
    if name == "grep":         return tool_grep(args.get("pattern", ""), args.get("path", "."), bool(args.get("regex", False)))
    if name == "glob":         return tool_glob(args.get("pattern", ""))
    if name == "run_command":  return tool_run_command(args.get("command", ""))
    if name == "run_tests":    return tool_run_tests(args.get("command"))
    if name == "git":          return tool_git(args.get("subcommand", ""))
    if name == "analyze_image":return tool_analyze_image(args.get("path", ""), args.get("question", ""), vision_model)
    if name == "generate_image":
        return tool_generate_image(
            args.get("prompt", ""),
            args.get("filename", ""),
            int(args.get("width", 512) or 512),
            int(args.get("height", 512) or 512),
            int(args.get("steps", 20) or 20),
        )
    if name == "run_subtask":  return tool_run_subtask(args.get("goal", ""), model, vision_model, perms, depth, sink)
    return f"ERROR: unknown tool '{name}'"


# --- Ollama HTTP -------------------------------------------------------------

def ollama_chat(model, messages, tools):
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
    }
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=body, timeout=600)
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    if r.status_code != 200:
        raise RuntimeError(f"Ollama /api/chat returned {r.status_code}: {(r.text or '')[:300]}")
    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Ollama returned non-JSON: {e}")


# --- Pretty-printing helpers -------------------------------------------------

def _short_args(args) -> str:
    if not isinstance(args, dict):
        return str(args)[:120]
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            s = v.replace("\n", "\\n")
            if len(s) > 60:
                s = s[:60] + "..."
            parts.append(f"{k}={s!r}")
        elif isinstance(v, (int, float, bool)) or v is None:
            parts.append(f"{k}={v}")
        else:
            try:
                s = json.dumps(v, ensure_ascii=False)
            except Exception:
                s = str(v)
            if len(s) > 60:
                s = s[:60] + "..."
            parts.append(f"{k}={s}")
    s = ", ".join(parts)
    if len(s) > 200:
        s = s[:200] + "..."
    return s


def _short_preview(s, limit=160) -> str:
    s = str(s).replace("\n", " | ")
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


# --- The agent loop ----------------------------------------------------------

def agent_loop(messages, model, vision_model, perms, tool_specs,
               depth=0, max_rounds=None, label="agent", sink=None) -> str:
    """One turn of the agent. Mutates `messages` in place.

    Returns the final assistant text (or an empty string on early failure).
    Output is routed through `sink` (defaults to the print-based EventSink).
    """
    if max_rounds is None:
        max_rounds = MAX_TOOL_ROUNDS
    if sink is None:
        sink = EventSink()

    last_text = ""
    for _ in range(1, max_rounds + 1):
        try:
            resp = ollama_chat(model, messages, tool_specs)
        except RuntimeError as e:
            sink.on_error(str(e))
            return last_text

        msg = resp.get("message", {}) or {}
        # Echo the assistant message into history verbatim, including any
        # tool_calls so the model sees its own prior calls on the next round.
        assistant_msg = {"role": "assistant", "content": msg.get("content", "") or ""}
        if msg.get("tool_calls"):
            assistant_msg["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            content = msg.get("content", "") or ""
            last_text = content
            sink.on_assistant_text(content, label)
            return content

        # Execute each requested tool call sequentially.
        for call in tool_calls:
            fn = (call.get("function") or {})
            name = fn.get("name", "") or ""
            args = _parse_args(fn.get("arguments"))

            sink.on_tool_call(name, args)

            if name in CONFIRM_TOOLS and not perms.ask(name, args):
                result = "DENIED: user refused this action."
            else:
                try:
                    result = _call_tool(name, args, model, vision_model, perms, depth, sink)
                except Exception as e:
                    result = f"ERROR: tool '{name}' raised: {e}"

            sink.on_tool_result(name, result)
            messages.append({
                "role": "tool",
                "name": name,
                "content": str(result),
            })

    sink.on_round_cap(label, max_rounds)
    return last_text or f"[{label}] hit round limit ({max_rounds}) — stopping."


# --- REPL --------------------------------------------------------------------

def _new_messages():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def repl(model, vision_model, perms):
    print(BOLD(BLUE(
        f"watchdog v3  —  model={model}, vision={vision_model}, "
        f"yolo={'on' if perms.yolo else 'off'}, host={OLLAMA_HOST}"
    )))
    print(DIM("Type /reset to clear history, /exit or /quit to leave."))

    messages = _new_messages()
    while True:
        try:
            user = input(BOLD("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/reset":
            messages = _new_messages()
            print(DIM("(history cleared)"))
            continue

        messages.append({"role": "user", "content": user})
        try:
            agent_loop(messages, model, vision_model, perms, TOOL_SPECS, depth=0, sink=EventSink())
        except KeyboardInterrupt:
            print(YELLOW("\n(interrupted)"))
            continue


def main():
    ap = argparse.ArgumentParser(
        prog="watchdog",
        description="Local CLI coding agent on Ollama.",
    )
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help=f"Chat model (default: {DEFAULT_MODEL})")
    ap.add_argument("--vision-model", default=DEFAULT_VISION_MODEL,
                    help=f"Vision model (default: {DEFAULT_VISION_MODEL})")
    ap.add_argument("--yolo", action="store_true",
                    help="Skip all permission prompts.")
    ap.add_argument("--web", action="store_true",
                    help="Launch the browser chat UI instead of the CLI REPL "
                         "(requires `pip install fastapi uvicorn`).")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address for --web (default: 127.0.0.1).")
    ap.add_argument("--port", type=int, default=8765,
                    help="Port for --web (default: 8765).")
    args = ap.parse_args()

    try:
        if args.web:
            try:
                from watchdog_web import serve
            except ImportError as e:
                print(
                    "--web requires `fastapi` and `uvicorn`. Install with:\n"
                    "  pip install fastapi uvicorn\n"
                    f"  ({e})",
                    file=sys.stderr,
                )
                sys.exit(1)
            serve(args.model, args.vision_model, args.yolo, args.host, args.port)
        else:
            perms = Permissions(args.yolo)
            repl(args.model, args.vision_model, perms)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
