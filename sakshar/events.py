"""In-process pub/sub for pipeline stage events (consumed by the SSE endpoint)."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field


@dataclass
class Event:
    stage: str
    label: str
    status: str = "running"  # running | done | error
    progress: float = 0.0
    detail: str = ""


@dataclass
class _Channel:
    subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list)
    history: list[Event] = field(default_factory=list)


class EventBus:
    def __init__(self) -> None:
        self._channels: dict[str, _Channel] = {}
        self._lock = threading.Lock()

    def publish(self, doc_id: str, event: Event) -> None:
        """Safe to call from the event loop or from worker threads."""
        with self._lock:
            ch = self._channels.setdefault(doc_id, _Channel())
            ch.history.append(event)
            subs = list(ch.subscribers)
        for loop, q in subs:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                q.put_nowait(event)
            else:
                loop.call_soon_threadsafe(q.put_nowait, event)

    def subscribe(self, doc_id: str) -> tuple[asyncio.Queue, list[Event]]:
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            ch = self._channels.setdefault(doc_id, _Channel())
            ch.subscribers.append((loop, q))
            return q, list(ch.history)

    def unsubscribe(self, doc_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            ch = self._channels.get(doc_id)
            if ch:
                ch.subscribers = [(l, x) for l, x in ch.subscribers if x is not q]


bus = EventBus()
