"""A per-process LRU of language bundles, pinned for the duration of each passage."""

import gc
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, RLock


@dataclass
class Bundle:
    parser: object = None
    coref: object = None
    users: int = 0
    loading: bool = True
    lock: object = field(default_factory=RLock)


class LanguageCache:
    def __init__(self, limit):
        if limit < 1:
            raise ValueError("max loaded languages must be a positive integer")
        self.limit = limit
        self.entries = OrderedDict()
        self.condition = Condition()

    def resident(self):
        with self.condition:
            return [k for k in reversed(self.entries) if not self.entries[k].loading]

    @contextmanager
    def acquire(self, language, load):
        evicted = False
        with self.condition:
            while True:
                if language in self.entries:
                    if self.entries[language].loading:
                        self.condition.wait()
                        continue
                    entry = self.entries[language]
                    self.entries.move_to_end(language)
                    break
                if len(self.entries) == self.limit:
                    idle = next(
                        (k for k, v in self.entries.items() if not v.users), None
                    )
                    if idle is None:
                        self.condition.wait()
                        continue
                    old = self.entries.pop(idle)
                    old.coref = old.parser = None
                    evicted = True
                entry = self.entries[language] = Bundle()
                break
            entry.users += 1
        try:
            if entry.loading:
                try:
                    if evicted:
                        gc.collect()  # Release provider reference cycles before the next allocation.
                    entry.parser = load(language)
                except BaseException:
                    with self.condition:
                        del self.entries[language]
                        self.condition.notify_all()
                    raise
                with self.condition:
                    entry.loading = False
                    self.condition.notify_all()
            with entry.lock:
                yield entry
        finally:
            with self.condition:
                entry.users -= 1
                self.condition.notify_all()


def default_limit():
    return int(os.environ.get("TEXT_TO_GRAPH_MAX_LOADED_LANGUAGES", "4"))
