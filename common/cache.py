import time,threading
from collections import OrderedDict

# thread safe
import time
from collections import OrderedDict
import threading


class ThreadSafeCache:
  def __init__(self, max_size=100, default_expiration=60):
    self.cache = OrderedDict()
    self.max_size = max_size
    self.default_expiration = default_expiration
    self.lock = self.ReadWriteLock()

  def _get_current_time(self):
    return time.time()

  class ReadWriteLock:
    def __init__(self):
      self.rlock = threading.RLock()
      self.wlock = threading.Lock()
      self.readers = 0

    def acquire_read(self):
      with self.rlock:
        self.readers += 1
        if self.readers == 1:
          self.wlock.acquire()

    def release_read(self):
      with self.rlock:
        self.readers -= 1
        if self.readers == 0:
          self.wlock.release()

    def acquire_write(self):
      self.wlock.acquire()

    def release_write(self):
      self.wlock.release()

  def get(self, key):
    self.lock.acquire_read()
    try:
      if key in self.cache:
        value, expiration = self.cache[key]
        if self._get_current_time() < expiration:
          self.cache.move_to_end(key)
          return value
        else:
          del self.cache[key]
      return None
    finally:
      self.lock.release_read()

  def set(self, key, value, expiration=None):
    self.lock.acquire_write()
    try:
      if expiration is None:
        expiration = self._get_current_time() + self.default_expiration
      if len(self.cache) >= self.max_size:
        self.cache.popitem(last=False)
      self.cache[key] = (value, expiration)
    finally:
      self.lock.release_write()

  def clear_expired(self):
    self.lock.acquire_write()
    try:
      current_time = self._get_current_time()
      keys_to_delete = []
      for key, (_, expiration) in self.cache.items():
        if current_time >= expiration:
          keys_to_delete.append(key)
      for key in keys_to_delete:
        del self.cache[key]
    finally:
      self.lock.release_write()