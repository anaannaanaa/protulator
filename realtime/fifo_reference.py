import os
import threading
import time
import numpy as np


class FIFOReference:
    """Read a changing reference value from a Linux FIFO."""

    def __init__(self, fifo_path, initial_value=0.0):
        self.fifo_path = fifo_path
        self.value = float(initial_value)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)

        self._thread = threading.Thread(
            target=self._reader,
            daemon=True,
        )
        self._thread.start()

    def _reader(self):
        while not self._stop_event.is_set():
            try:
                # O_RDWR prevents open() from waiting for another process.
                fd = os.open(self.fifo_path, os.O_RDWR | os.O_NONBLOCK)

                try:
                    while not self._stop_event.is_set():
                        try:
                            data = os.read(fd, 256)

                            if data:
                                text = data.decode().strip()

                                # Accept the latest numeric value.
                                for line in text.splitlines():
                                    try:
                                        value = float(line)
                                    except ValueError:
                                        continue

                                    with self._lock:
                                        self.value = value

                        except BlockingIOError:
                            time.sleep(0.001)

                finally:
                    os.close(fd)

            except OSError:
                time.sleep(0.01)

    def __call__(self, t):
        """Return the latest reference; compatible with ref_fcn(t)."""
        with self._lock:
            return self.value

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
