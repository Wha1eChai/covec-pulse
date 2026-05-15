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
        api_key: str | None = None,
        verdict_path: str | Path = "pulse_outputs/scope_verdict.jsonl",
        timeout: float = 3.0,
        display: bool = True,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
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
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = Request(
                self.endpoint,
                data=body,
                headers=headers,
                method="POST",
            )
            _MAX_RESPONSE = 16384
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read(_MAX_RESPONSE)
                result = json.loads(raw.decode("utf-8"))

            if not isinstance(result, dict) or "verdict" not in result:
                return

            safe_record = {
                "step": data.get("step"),
                "verdict": result.get("verdict"),
                "summary": result.get("summary", ""),
            }
            with open(self.verdict_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

            if self.display:
                verdict = result.get("verdict", "?")
                summary = result.get("summary", "")
                step = data.get("step", "?")
                print(f"[Scope] step={step:>5}  {verdict} -- {summary}")
                sys.stdout.flush()

        except (URLError, OSError) as e:
            if "SSL" in str(e) or "CERTIFICATE" in str(e).upper():
                print(f"[Scope] WARNING: TLS error — {e}", file=sys.stderr)
            # Other network errors: silent (don't disrupt training)
        except (ValueError, KeyError):
            pass
