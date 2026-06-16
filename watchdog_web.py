#!/usr/bin/env python3
"""Browser chat UI for watchdog — FastAPI + Server-Sent Events.

Run:
    pip install -r requirements.txt
    python watchdog.py --web
    # open http://127.0.0.1:8765

Architecture
------------
- Each user turn runs `agent_loop` on a background thread (the loop is sync).
- A `WebSink` pushes events from the loop onto a thread-safe queue.
- POST /api/message returns a streaming SSE response that drains the queue.
- Permission prompts (write_file / edit_file / run_command / ...) pause the
  agent thread on a threading.Event; POST /api/confirm releases it with the
  user's choice.

v1 is single-user, single-session. The session object is module-global.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from watchdog import (
    OLLAMA_HOST,
    SYSTEM_PROMPT,
    TOOL_SPECS,
    EventSink,
    Permissions,
    agent_loop,
)


# --- Per-turn rendezvous primitives -----------------------------------------

class _PendingConfirm:
    """Rendezvous between the agent thread and POST /api/confirm.

    The agent emits a `confirm` event carrying `id`, then waits on `event`.
    The HTTP handler writes `result` and sets `event`.
    """

    __slots__ = ("id", "event", "result")

    def __init__(self, id: str):
        self.id = id
        self.event = threading.Event()
        self.result = False


class WebSession:
    def __init__(self, model: str, vision_model: str, yolo: bool):
        self.model = model
        self.vision_model = vision_model
        self.yolo = yolo
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.confirms: dict[str, _PendingConfirm] = {}
        self.lock = threading.Lock()
        self.busy = False

    def reset(self) -> None:
        with self.lock:
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


class WebSink(EventSink):
    """Emits agent-loop events into the session's queue as plain dicts."""

    def __init__(self, session: WebSession):
        self.session = session

    def _emit(self, type: str, **payload) -> None:
        self.session.events.put({"type": type, **payload})

    def on_assistant_text(self, text: str, label: str = "agent") -> None:
        # Subtask final text becomes the run_subtask tool result already; the
        # parent agent's `tool_result` event carries it. Suppress here.
        if label != "agent":
            return
        self._emit("assistant", text=text)

    def on_tool_call(self, name: str, args: dict) -> None:
        self._emit("tool_call", name=name, args=args)

    def on_tool_result(self, name: str, result: str) -> None:
        first = (result or "").split("\n", 1)[0]
        is_error = first.startswith("ERROR") or first.startswith("DENIED")
        self._emit("tool_result", name=name, result=result, is_error=is_error)

    def on_subtask_start(self, goal: str) -> None:
        self._emit("subtask_start", goal=goal)

    def on_subtask_end(self) -> None:
        self._emit("subtask_end")

    def on_error(self, message: str) -> None:
        self._emit("error", message=message)

    def on_round_cap(self, label: str, max_rounds: int) -> None:
        self._emit("round_cap", label=label, max_rounds=max_rounds)


class WebPermissions(Permissions):
    """Blocks the agent thread on a per-call Event until the user clicks
    Allow / Deny in the browser. --yolo short-circuits to allow."""

    def __init__(self, yolo: bool, session: WebSession):
        super().__init__(yolo)
        self.session = session
        self._counter = 0
        self._counter_lock = threading.Lock()

    def ask(self, tool_name: str, args: dict) -> bool:
        if self.yolo:
            return True
        with self._counter_lock:
            self._counter += 1
            cid = f"c{self._counter}"
        pc = _PendingConfirm(cid)
        self.session.confirms[cid] = pc
        self.session.events.put({
            "type": "confirm",
            "id": cid,
            "name": tool_name,
            "args": args,
        })
        # Cap at 10 min. If the browser closes without responding, deny.
        signalled = pc.event.wait(timeout=600)
        self.session.confirms.pop(cid, None)
        return pc.result if signalled else False


# --- FastAPI app -------------------------------------------------------------

_BASE = Path(__file__).resolve().parent
_WEB_DIR = _BASE / "web"

app = FastAPI(title="watchdog")
_session: Optional[WebSession] = None


class MessageBody(BaseModel):
    text: str


class ConfirmBody(BaseModel):
    id: str
    allow: bool


@app.get("/")
def root():
    return FileResponse(_WEB_DIR / "index.html")


@app.get("/api/info")
def info():
    s = _session
    return {
        "model": s.model,
        "vision_model": s.vision_model,
        "yolo": s.yolo,
        "host": OLLAMA_HOST,
    }


@app.post("/api/reset")
def reset():
    _session.reset()
    return {"ok": True}


@app.post("/api/confirm")
def confirm(body: ConfirmBody):
    pc = _session.confirms.get(body.id)
    if pc is None:
        raise HTTPException(404, "no such pending confirm (already resolved or expired)")
    pc.result = bool(body.allow)
    pc.event.set()
    return {"ok": True}


def _run_turn(session: WebSession, text: str) -> None:
    sink = WebSink(session)
    perms = WebPermissions(session.yolo, session)
    session.messages.append({"role": "user", "content": text})
    try:
        agent_loop(
            session.messages,
            session.model,
            session.vision_model,
            perms,
            TOOL_SPECS,
            depth=0,
            sink=sink,
        )
    except Exception as e:
        sink.on_error(f"agent crashed: {e!r}")
    finally:
        session.events.put({"type": "done"})
        with session.lock:
            session.busy = False


@app.post("/api/message")
def message(body: MessageBody):
    s = _session
    with s.lock:
        if s.busy:
            raise HTTPException(409, "agent is busy with the previous message")
        s.busy = True
    # Echo to anyone listening to the stream (handy if reconnects are added later).
    s.events.put({"type": "user", "text": body.text})
    threading.Thread(target=_run_turn, args=(s, body.text), daemon=True).start()
    return StreamingResponse(
        _event_stream(s),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering if any
        },
    )


def _event_stream(s: WebSession):
    """SSE generator: drain the session queue until 'done'.

    Sends a comment every second when idle so middleboxes don't kill the
    connection mid-LLM-call.
    """
    yield ": stream open\n\n"
    while True:
        try:
            ev = s.events.get(timeout=1.0)
        except queue.Empty:
            yield ": ping\n\n"
            continue
        type_ = ev.get("type", "message")
        data = json.dumps(ev, ensure_ascii=False)
        yield f"event: {type_}\ndata: {data}\n\n"
        if type_ == "done":
            break


def serve(model: str, vision_model: str, yolo: bool, host: str, port: int) -> None:
    global _session
    if not _WEB_DIR.exists():
        raise FileNotFoundError(
            f"web/ directory missing: expected at {_WEB_DIR}. "
            "It should ship next to watchdog_web.py."
        )
    _session = WebSession(model, vision_model, yolo)
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "uvicorn is not installed. Run: pip install uvicorn"
        ) from e
    yolo_tag = "  [YOLO mode — no confirmations]" if yolo else ""
    print(f"watchdog web UI on http://{host}:{port}{yolo_tag}")
    print("Open it in your browser. Ctrl-C to stop.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
