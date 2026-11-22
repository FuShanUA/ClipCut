from __future__ import annotations

from contextlib import contextmanager
import subprocess
import threading
from typing import Any, Iterator


class JobCancelledError(Exception):
    """Raised when a running pipeline/planning job is manually cancelled."""


class JobCancelToken:
    """Thread-safe cooperative + preemptive cancellation token for jobs."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen[Any]] = set()
        self._clients: set[Any] = set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self, msg: str = "任务已手动中止") -> None:
        if self._event.is_set():
            raise JobCancelledError(msg)

    def register_proc(self, proc: subprocess.Popen[Any]) -> None:
        cancelled = False
        with self._lock:
            if self._event.is_set():
                cancelled = True
            else:
                self._procs.add(proc)
        if cancelled:
            _stop_proc(proc)
            raise JobCancelledError("任务已手动中止")

    def unregister_proc(self, proc: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._procs.discard(proc)

    def register_client(self, client: Any) -> None:
        cancelled = False
        with self._lock:
            if self._event.is_set():
                cancelled = True
            else:
                self._clients.add(client)
        if cancelled:
            try:
                client.close()
            except Exception:
                pass
            raise JobCancelledError("任务已手动中止")

    def unregister_client(self, client: Any) -> None:
        with self._lock:
            self._clients.discard(client)

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            clients = list(self._clients)
            procs = list(self._procs)
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        for proc in procs:
            _stop_proc(proc)


def _stop_proc(proc: subprocess.Popen[Any]) -> None:
    try:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=0.3)
            except Exception:
                pass
    except Exception:
        pass


_LOCAL = threading.local()


def get_current_cancel_token() -> JobCancelToken | None:
    return getattr(_LOCAL, "cancel_token", None)


def check_cancelled(msg: str = "任务已手动中止") -> None:
    token = get_current_cancel_token()
    if token is not None:
        token.check(msg)


@contextmanager
def use_cancel_token(token: JobCancelToken | None) -> Iterator[JobCancelToken | None]:
    if token is None:
        yield get_current_cancel_token()
        return
    prev = getattr(_LOCAL, "cancel_token", None)
    _LOCAL.cancel_token = token
    try:
        yield token
    finally:
        _LOCAL.cancel_token = prev


def run_cancellable_subprocess(
    cmd: list[str],
    *,
    capture_output: bool = True,
    text: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[Any]:
    token = get_current_cancel_token()
    if token is None:
        return subprocess.run(cmd, capture_output=capture_output, text=text, timeout=timeout)

    token.check()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
    )
    token.register_proc(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_proc(proc)
        raise
    except Exception:
        if token.is_cancelled:
            _stop_proc(proc)
            raise JobCancelledError("任务已手动中止") from None
        _stop_proc(proc)
        raise
    finally:
        token.unregister_proc(proc)

    if token.is_cancelled:
        _stop_proc(proc)
        raise JobCancelledError("任务已手动中止")
    return subprocess.CompletedProcess(cmd, int(proc.returncode or 0), stdout, stderr)
