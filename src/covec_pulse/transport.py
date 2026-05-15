"""Async transport to Scope server — non-blocking, fire-and-forget."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError


class ScopeTransport:
    """Send probe snapshots to Scope server in background threads.

    Never blocks training. Failures are silent.
    """

    def __init__(
        self,
        endpoint: str,
        verdict_path: str | Path = "pulse_outputs/scope_verdict.jsonl",
        timeout: float = 3.0,
        display: bool = True,
    ):
        self.endpoint = endpoint
        self.verdict_path = Path(verdict_path)
        self.verdict_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.display = display

    def send(self, probe_dict: dict) -> None:
        """Fire-and-forget POST in background thread."""
        t = threading.Thread(
            target=self._post,
            args=(probe_dict,),
            daemon=True,
        )
        t.start()

    def _post(self, data: dict) -> None:
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            req = Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            with open(self.verdict_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if self.display:
                status = result.get("status", "?")
                conf = result.get("confidence", "")
                msg = result.get("message", "")
                step = data.get("step", "?")
                print(f"[Scope] step={step:>5}  {status} ({conf}) -- {msg}")
                sys.stdout.flush()

        except (URLError, OSError, ValueError, KeyError):
            pass
