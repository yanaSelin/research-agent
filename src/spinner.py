"""Terminal spinner — shows a braille animation on stderr while work runs."""
import itertools
import sys
import threading
import time


class Spinner:
    """Context manager that shows a spinner on stderr during a blocking operation.

    Args:
        message: Label shown next to the spinner frames.
    """

    def __init__(self, message: str = "Thinking") -> None:
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        while not self._stop.is_set():
            sys.stderr.write(f"\r{next(frames)} {self._message}...")
            sys.stderr.flush()
            time.sleep(0.1)
        sys.stderr.write(f"\r{' ' * (len(self._message) + 12)}\r")
        sys.stderr.flush()

    def __enter__(self) -> "Spinner":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
