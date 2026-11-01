"""Switching off the machine the demo runs on once the demo has been unused for a while, so that a paid server is
not left running. Nothing about it is shown on the demo's pages.

Off unless ``power_off_command`` (demo.yaml or RB_DEMO_POWER_OFF_COMMAND) and ``idle_stop_minutes``
(RB_DEMO_IDLE_STOP_MINUTES) are both set; for example ``sudo -n /usr/sbin/shutdown -h now`` and 15 on a cloud server
whose shutdown stops the instance. Unused means no job running or waiting, and no page of the demo opened, for that
long. Before the command runs, a running tool is stopped and every job folder is deleted, as when the app stops.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
import time
from collections.abc import Callable

from . import jobs as J

log = logging.getLogger("demo")

CHECK_EVERY_S = 20


class Power:
    def __init__(self, command: str, idle_stop_minutes: float, queue: J.JobQueue, *,
                 clock: Callable[[], float] = time.time, run: Callable[[list[str]], object] | None = None,
                 watch: bool = True):
        self.argv = shlex.split(command) if command else []
        self.idle_limit_s = max(0.0, idle_stop_minutes) * 60
        self.queue = queue
        self.clock = clock
        self._run = run or (lambda argv: subprocess.run(argv, check=False, stdin=subprocess.DEVNULL))
        self.last_use = clock()
        self.stopping = False
        if watch and self.enabled:
            threading.Thread(target=self._watch, name="demo-idle-stop", daemon=True).start()

    @property
    def enabled(self) -> bool:
        return bool(self.argv) and self.idle_limit_s > 0

    def busy(self) -> bool:
        """A job is running or waiting."""
        return any(j.state in (J.QUEUED, J.RUNNING) for j in list(self.queue.jobs.values()))

    def used(self) -> None:
        """Someone opened a page of the demo."""
        self.last_use = self.clock()

    def stops_in(self) -> float | None:
        """Seconds until the machine is switched off, as things stand; None when the idle stop is off or a job is
        running or waiting."""
        if not self.enabled or self.busy():
            return None
        return max(0.0, self.last_use + self.idle_limit_s - self.clock())

    def check(self) -> bool:
        """One look by the watcher: a running or waiting job counts as use; switch off when unused long enough."""
        if self.stopping:
            return False
        if self.busy():
            self.used()
            return False
        left = self.stops_in()
        if left is None or left > 0:
            return False
        self.stopping = True
        log.warning("switching the server off: unused for %g minutes", self.idle_limit_s / 60)
        self.queue.shutdown()
        self._run(self.argv)
        return True

    def _watch(self) -> None:
        while not self.stopping:
            time.sleep(CHECK_EVERY_S)
            try:
                self.check()
            except Exception:  # noqa: BLE001 - the watcher must keep running
                log.exception("idle check failed")
